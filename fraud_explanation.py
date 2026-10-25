import joblib
import shap
import mysql.connector
import pandas as pd

MODEL_PATH = "trained_models_v2/random_forest.pkl"
FEATURE_PATH = "trained_models_v2/feature_columns.pkl"

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "3344",
    "database": "banking_fraud_system",
}


def build_features(row, feature_columns):
    amount = float(row["amount"])
    old_origin = float(row["oldbalanceOrg"])
    new_origin = float(row["newbalanceOrig"])
    old_destination = float(row["oldbalanceDest"])
    new_destination = float(row["newbalanceDest"])

    data = pd.DataFrame([{
        "amount": amount,
        "oldbalanceOrg": old_origin,
        "newbalanceOrig": new_origin,
        "oldbalanceDest": old_destination,
        "newbalanceDest": new_destination,
        "errorBalanceOrig": old_origin - amount - new_origin,
        "errorBalanceDest": old_destination + amount - new_destination,
        "type": str(row["transaction_type"]).strip().upper(),
    }])
    data = pd.get_dummies(data, columns=["type"], dtype=int)
    return data.reindex(columns=feature_columns, fill_value=0)


def get_positive_class_shap(explainer, X):
    values = explainer.shap_values(X)

    if isinstance(values, list):
        return values[1][0]

    # SHAP versions may return samples x features x classes.
    if getattr(values, "ndim", 0) == 3:
        return values[0, :, 1]

    # Binary model output may be samples x features.
    return values[0]


def main():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
    explainer = shap.TreeExplainer(model)

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    try:
        try:
            transaction_id = int(input("Enter transaction ID to explain: ").strip())
        except ValueError:
            print("Transaction ID must be an integer.")
            return

        cursor.execute("""
            SELECT
                t.transaction_id,
                t.amount,
                t.transaction_type,
                t.oldbalance_org AS oldbalanceOrg,
                t.newbalance_orig AS newbalanceOrig,
                t.oldbalance_dest AS oldbalanceDest,
                t.newbalance_dest AS newbalanceDest,
                p.fraud_score AS stored_fraud_score,
                p.risk_level AS stored_risk_level,
                p.prediction AS stored_prediction,
                p.model_version
            FROM transactions t
            LEFT JOIN ai_predictions p
                ON p.transaction_id = t.transaction_id
            WHERE t.transaction_id = %s
        """, (transaction_id,))
        row = cursor.fetchone()

        if row is None:
            print("Transaction not found.")
            return

        balance_keys = [
            "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest"
        ]
        if any(row[key] is None for key in balance_keys):
            print(
                "Cannot explain this transaction with V3: balance features are missing. "
                "Update the transaction importer and populate these columns first."
            )
            return

        X = build_features(row, feature_columns)
        probability = float(model.predict_proba(X)[0, 1])
        predicted_class = int(model.predict(X)[0])
        contributions = get_positive_class_shap(explainer, X)

        print("\n========== V3 TRANSACTION EXPLANATION ==========")
        print(f"Transaction ID: {transaction_id}")
        print(f"Amount: {row['amount']}")
        print(f"Type: {row['transaction_type']}")
        print(f"Current V3 fraud probability: {probability:.4f}")
        print(f"Current V3 prediction: {'Fraud' if predicted_class else 'Legitimate'}")
        print(f"Stored prediction: {row['stored_prediction']}")
        print(f"Stored risk level: {row['stored_risk_level']}")
        print(f"Stored score: {row['stored_fraud_score']}")
        print(f"Stored model version: {row['model_version']}")

        result = pd.DataFrame({
            "feature": feature_columns,
            "value": X.iloc[0].to_numpy(),
            "shap_contribution": contributions,
        })
        result["absolute_impact"] = result["shap_contribution"].abs()
        result = result.sort_values("absolute_impact", ascending=False)

        print("\n========== TOP SHAP CONTRIBUTIONS ==========")
        for _, item in result.head(10).iterrows():
            contribution = float(item["shap_contribution"])
            direction = (
                "pushed toward fraud" if contribution > 0
                else "pushed away from fraud" if contribution < 0
                else "had approximately no effect"
            )
            print(
                f"{item['feature']} = {item['value']} | "
                f"SHAP: {contribution:.6f} | {direction}"
            )

        print(
            "\nSHAP explains the current model output; it does not prove that "
            "a transaction is genuinely fraudulent."
        )
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
