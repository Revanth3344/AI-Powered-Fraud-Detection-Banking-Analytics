import os
from datetime import date, datetime
from decimal import Decimal
from flask import Flask, render_template, request, jsonify
import mysql.connector
import joblib
import pandas as pd

app = Flask(__name__)
DB = {
"host": os.getenv("DB_HOST", "localhost"),
"user": os.getenv("DB_USER", "root"),
"password": os.getenv("DB_PASSWORD", ""),"database": os.getenv("DB_NAME", "banking_fraud_system"),
"port": int(os.getenv("DB_PORT", "3306"))
}

MODEL_PATH = "trained_models_v2/random_forest.pkl"
FEATURE_PATH = "trained_models_v2/feature_columns.pkl"

def conn(): return mysql.connector.connect(**DB)
def ident(s): return "`" + s.replace("`","``") + "`"
def rows(sql, params=()):
    db=conn()
    try:
        cur=db.cursor()
        cur.execute(sql,params)
        names=[x[0] for x in cur.description] if cur.description else []
        result=[]
        for row in cur.fetchall():
            item={}
            for k,v in zip(names,row):
                if isinstance(v,Decimal): v=float(v)
                elif isinstance(v,datetime): v=v.isoformat(sep=" ",timespec="seconds")
                elif isinstance(v,date): v=v.isoformat()
                item[k]=v
            result.append(item)
        cur.close()
        return result
    finally: db.close()
def one(sql,p=()):
    r=rows(sql,p)
    return r[0] if r else {}
def exists(t):
    return bool(rows("SELECT TABLE_NAME FROM information_schema.tables WHERE table_schema=%s AND table_name=%s",(DB["database"],t)))
def cols(t):
    return [r["COLUMN_NAME"] for r in rows("SELECT COLUMN_NAME FROM information_schema.columns WHERE table_schema=%s AND table_name=%s",(DB["database"],t))] if exists(t) else []
def pick(cs,*opts):
    d={c.lower():c for c in cs}
    return next((d[o.lower()] for o in opts if o.lower() in d),None)
def count(t): return int(one(f"SELECT COUNT(*) n FROM {ident(t)}").get("n",0)) if exists(t) else 0

def tx_meta():
    c=cols("transactions")
    return {"cols":c,"id":pick(c,"transaction_id","id"),"type":pick(c,"transaction_type","type"),
            "time":pick(c,"transaction_time","timestamp","created_at","date")}
def transactions(limit=250,q="",risk="all"):
    m=tx_meta()
    if not m["cols"]: return []
    order=m["time"] or m["id"] or m["cols"][0]
    sql="SELECT * FROM `transactions`"; params=[]
    if q and m["id"]:
        sql+=f" WHERE CAST({ident(m['id'])} AS CHAR) LIKE %s"; params.append("%"+q+"%")
    sql+=f" ORDER BY {ident(order)} DESC LIMIT %s"; params.append(limit)
    result=rows(sql,tuple(params))
    # Fetch predictions for the visible page in one query instead of one query per row.
    # This avoids hundreds of sequential database round trips on the Transactions page.
    pc=cols("ai_predictions"); ptx=pick(pc,"transaction_id")
    if result and ptx and m["id"]:
        pfields=[pick(pc,"risk_level","risk_category"),
                 pick(pc,"prediction","predicted_class","is_fraud","prediction_result"),
                 pick(pc,"fraud_score","fraud_probability","probability","risk_score")]
        pfields=[x for x in pfields if x]
        if pfields:
            ids=[r[m["id"]] for r in result]
            pid=pick(pc,"prediction_id","id")
            select_cols=[ident(ptx)] + [ident(x) for x in pfields]
            order_sql=f" ORDER BY {ident(pid)} DESC" if pid else ""
            placeholders=",".join(["%s"]*len(ids))
            preds=rows(f"SELECT {','.join(select_cols)} FROM `ai_predictions` WHERE {ident(ptx)} IN ({placeholders}){order_sql}",tuple(ids))
            latest={}
            for pred in preds:
                latest.setdefault(pred[ptx],pred)
            for r in result:
                pred=latest.get(r[m["id"]])
                if pred:
                    r.update({"ai_"+k:v for k,v in pred.items() if k != ptx})
    if risk!="all":
        def matches(r):
            val=str(r.get("ai_risk_level",r.get("ai_prediction",""))).lower()
            return ("high" in val or "fraud" in val) if risk=="high" else ("medium" in val if risk=="medium" else ("low" in val or "legitimate" in val))
        result=[r for r in result if matches(r)]
    return result

def alerts(limit=250,status="all"):
    c=cols("fraud_alerts")
    if not c:return []
    sc=pick(c,"status","alert_status"); order=pick(c,"created_at","alert_time","triggered_at","alert_id","id")
    sql="SELECT * FROM `fraud_alerts`"; p=[]
    if status!="all" and sc:sql+=f" WHERE {ident(sc)}=%s";p.append(status)
    if order:sql+=f" ORDER BY {ident(order)} DESC"
    sql+=" LIMIT %s";p.append(limit)
    return rows(sql,tuple(p))
def stats():
    s={"transactions":count("transactions"),"predictions":count("ai_predictions"),"alerts":count("fraud_alerts"),"open_alerts":0,"high_risk":0,"fraud_predictions":0}
    c=cols("fraud_alerts"); sc=pick(c,"status","alert_status")
    if sc:s["open_alerts"]=int(one(f"SELECT COUNT(*) n FROM fraud_alerts WHERE LOWER({ident(sc)}) NOT IN ('resolved','closed')").get("n",0))
    elif s["alerts"]:s["open_alerts"]=s["alerts"]
    c=cols("ai_predictions"); rc=pick(c,"risk_level","risk_category"); fc=pick(c,"prediction","predicted_class","is_fraud","prediction_result")
    if rc:s["high_risk"]=int(one(f"SELECT COUNT(*) n FROM ai_predictions WHERE LOWER({ident(rc)})='high'").get("n",0))
    if fc:s["fraud_predictions"]=int(one(f"SELECT COUNT(*) n FROM ai_predictions WHERE LOWER(CAST({ident(fc)} AS CHAR)) IN ('fraud','1','true','suspicious')").get("n",0))
    s["coverage"]=round(s["predictions"]/s["transactions"]*100,1) if s["transactions"] else 0
    return s

def build_features(row, feature_columns):
    amount=float(row["amount"]); old_origin=float(row["oldbalanceOrg"]); new_origin=float(row["newbalanceOrig"])
    old_destination=float(row["oldbalanceDest"]); new_destination=float(row["newbalanceDest"])
    data=pd.DataFrame([{"amount":amount,"oldbalanceOrg":old_origin,"newbalanceOrig":new_origin,
        "oldbalanceDest":old_destination,"newbalanceDest":new_destination,
        "errorBalanceOrig":old_origin-amount-new_origin,
        "errorBalanceDest":old_destination+amount-new_destination,
        "type":str(row["transaction_type"]).strip().upper()}])
    data=pd.get_dummies(data,columns=["type"],dtype=int)
    return data.reindex(columns=feature_columns,fill_value=0)

def shap_positive_values(explainer,X):
    values=explainer.shap_values(X)
    if isinstance(values,list): return values[1][0]
    if getattr(values,"ndim",0)==3: return values[0,:,1]
    return values[0]

@app.route("/")
def dashboard():
    error=None;s={k:0 for k in ["transactions","predictions","alerts","open_alerts","high_risk","fraud_predictions","coverage"]};recent=[]
    try:s=stats();recent=transactions(8)
    except Exception as e:error=str(e)
    return render_template("dashboard.html",stats=s,recent=recent,error=error,page="dashboard")
@app.route("/transactions")
def transaction_page():
    error=None;data=[];q=request.args.get("q","").strip();risk=request.args.get("risk","all")
    try:data=transactions(50,q,risk)
    except Exception as e:error=str(e)
    return render_template("transactions.html",rows=data,error=error,query=q,risk_filter=risk,page="transactions")
@app.route("/alerts")
def alert_page():
    error=None;data=[];status=request.args.get("status","all")
    try:data=alerts(100,status)
    except Exception as e:error=str(e)
    return render_template("alerts.html",rows=data,error=error,status_filter=status,page="alerts")
@app.route("/analytics")
def analytics():
    error=None;s={};types=[];days=[]
    try:
        s=stats();m=tx_meta()
        if m["type"]:types=rows(f"SELECT {ident(m['type'])} label,COUNT(*) value FROM transactions GROUP BY {ident(m['type'])} ORDER BY value DESC LIMIT 10")
        if m["time"]:
            days=rows(f"SELECT DATE({ident(m['time'])}) label,COUNT(*) value FROM transactions WHERE {ident(m['time'])} IS NOT NULL GROUP BY DATE({ident(m['time'])}) ORDER BY DATE({ident(m['time'])}) DESC LIMIT 14");days.reverse()
    except Exception as e:error=str(e)
    return render_template("analytics.html",stats=s,type_data=types,daily_data=days,error=error,page="analytics")

@app.route("/explain",methods=["GET","POST"])
def explain():
    error=None;result=None;transaction_id=request.values.get("transaction_id","").strip()
    if request.method=="POST":
        try:
            if not transaction_id.isdigit(): raise ValueError("Enter a numeric transaction ID.")
            model=joblib.load(MODEL_PATH); feature_columns=joblib.load(FEATURE_PATH)
            db=conn()
            try:
                cur=db.cursor(dictionary=True)
                cur.execute("""SELECT t.transaction_id,t.amount,t.transaction_type,
                    t.oldbalance_org AS oldbalanceOrg,t.newbalance_orig AS newbalanceOrig,
                    t.oldbalance_dest AS oldbalanceDest,t.newbalance_dest AS newbalanceDest,
                    p.fraud_score AS stored_fraud_score,p.risk_level AS stored_risk_level,
                    p.prediction AS stored_prediction,p.model_version
                    FROM transactions t LEFT JOIN ai_predictions p ON p.transaction_id=t.transaction_id
                    WHERE t.transaction_id=%s ORDER BY p.prediction_id DESC LIMIT 1""",(int(transaction_id),))
                row=cur.fetchone();cur.close()
            finally:db.close()
            if row is None: raise ValueError("Transaction ID was not found.")
            keys=["oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]
            if any(row.get(k) is None for k in keys): raise ValueError("This transaction is missing one or more balance features required by the V3 model.")
            X=build_features(row,feature_columns)
            probability=float(model.predict_proba(X)[0,1]); predicted=int(model.predict(X)[0])
            import shap
            explainer=shap.TreeExplainer(model)
            contributions=shap_positive_values(explainer,X)
            features=[]
            for name,value,impact in zip(feature_columns,X.iloc[0].to_numpy(),contributions):
                features.append({"name":str(name),"value":float(value),"impact":float(impact)})
            features.sort(key=lambda x:abs(x["impact"]),reverse=True)
            result={"id":row["transaction_id"],"amount":float(row["amount"]),"type":row["transaction_type"],
                "probability":probability,"predicted":"Fraud" if predicted else "Legitimate",
                "risk":"High" if probability>=.70 else "Medium" if probability>=.30 else "Low",
                "stored_prediction":row.get("stored_prediction"),"stored_risk":row.get("stored_risk_level"),
                "model_version":row.get("model_version"),"features":features[:10]}
        except Exception as e:error=str(e)
    return render_template("explain.html",result=result,error=error,transaction_id=transaction_id,page="explain")

@app.route("/api/alerts/<int:aid>/status",methods=["POST"])
def update_status(aid):
    status=(request.get_json(silent=True) or {}).get("status","")
    if status not in {"Open","Investigating","Resolved","Closed"}:return jsonify(ok=False,error="Invalid status"),400
    c=cols("fraud_alerts");idc=pick(c,"alert_id","id");sc=pick(c,"status","alert_status")
    if not idc or not sc:return jsonify(ok=False,error="Could not detect alert ID/status columns"),400
    db=conn()
    try:
        cur=db.cursor();cur.execute(f"UPDATE fraud_alerts SET {ident(sc)}=%s WHERE {ident(idc)}=%s",(status,aid));db.commit();changed=cur.rowcount;cur.close()
        return jsonify(ok=changed>0,updated=changed)
    finally:db.close()
if __name__=="__main__":app.run(debug=True)
