import joblib
import mysql.connector
import pandas as pd

MODEL_PATH = "trained_models_v2/random_forest.pkl"
FEATURE_PATH = "trained_models_v2/feature_columns.pkl"
BATCH_SIZE = 1000

HIGH_RISK_THRESHOLD = 0.70
MEDIUM_RISK_THRESHOLD = 0.30

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "3344",
    "database": "banking_fraud_system",
}


def build_features(rows, feature_columns):
    """Use exactly the features and formulas used during V3 training."""
    records = []

    for row in rows:
        amount = float(row["amount"])
        old_origin = float(row["oldbalanceOrg"])
        new_origin = float(row["newbalanceOrig"])
        old_destination = float(row["oldbalanceDest"])
        new_destination = float(row["newbalanceDest"])
        transaction_type = str(row["transaction_type"]).strip().upper()

        records.append({
            "amount": amount,
            "oldbalanceOrg": old_origin,
            "newbalanceOrig": new_origin,
            "oldbalanceDest": old_destination,
            "newbalanceDest": new_destination,
            "errorBalanceOrig": old_origin - amount - new_origin,
            "errorBalanceDest": old_destination + amount - new_destination,
            "type": transaction_type,
        })

    frame = pd.DataFrame(records)
    frame = pd.get_dummies(frame, columns=["type"], dtype=int)
    return frame.reindex(columns=feature_columns, fill_value=0)


def main():
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
    print("V3 Random Forest model loaded.")

    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    processed = high = medium = low = 0

    try:
        # Requires the balance columns added by the SQL migration.
        cursor.execute("""
            SELECT
                t.transaction_id,
                t.amount,
                t.transaction_type,
                t.oldbalance_org AS oldbalanceOrg,
                t.newbalance_orig AS newbalanceOrig,
                t.oldbalance_dest AS oldbalanceDest,
                t.newbalance_dest AS newbalanceDest
            FROM transactions t
            LEFT JOIN ai_predictions p
                ON p.transaction_id = t.transaction_id
            WHERE p.transaction_id IS NULL
              AND t.oldbalance_org IS NOT NULL
              AND t.newbalance_orig IS NOT NULL
              AND t.oldbalance_dest IS NOT NULL
              AND t.newbalance_dest IS NOT NULL
            ORDER BY t.transaction_id
            LIMIT %s
        """, (BATCH_SIZE,))
        transactions = cursor.fetchall()

        cursor.execute("""
            SELECT COUNT(*) AS missing_feature_rows
            FROM transactions t
            LEFT JOIN ai_predictions p
                ON p.transaction_id = t.transaction_id
            WHERE p.transaction_id IS NULL
              AND (
                  t.oldbalance_org IS NULL
                  OR t.newbalance_orig IS NULL
                  OR t.oldbalance_dest IS NULL
                  OR t.newbalance_dest IS NULL
              )
        """)
        missing_feature_rows = cursor.fetchone()["missing_feature_rows"]

        if not transactions:
            print("No unprocessed transactions with complete V3 features found.")
            if missing_feature_rows:
                print(
                    f"{missing_feature_rows} unprocessed rows lack balance features. "
                    "Populate them through the importer before predicting."
                )
            return

        X = build_features(transactions, feature_columns)
        probabilities = model.predict_proba(X)[:, 1]
        predictions = model.predict(X)

        for row, probability, predicted_class in zip(
            transactions, probabilities, predictions
        ):
            score = round(float(probability), 4)
            if score >= HIGH_RISK_THRESHOLD:
                risk = "High"
                high += 1
            elif score >= MEDIUM_RISK_THRESHOLD:
                risk = "Medium"
                medium += 1
            else:
                risk = "Low"
                low += 1

            label = "Fraud" if int(predicted_class) == 1 else "Legitimate"

            cursor.execute("""
                INSERT INTO ai_predictions
                    (transaction_id, fraud_score, risk_level, prediction, model_version)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                row["transaction_id"], score, risk, label, "V3_RandomForest"
            ))
            processed += 1

        # The existing AFTER INSERT trigger creates alerts for High-risk predictions.
        conn.commit()

        print("\n========== V3 PROCESSING SUMMARY ==========")
        print(f"Transactions processed: {processed}")
        print(f"High-risk predictions: {high}")
        print(f"Medium-risk predictions: {medium}")
        print(f"Low-risk predictions: {low}")
        print("Database changes committed.")
        if missing_feature_rows:
            print(f"Skipped unprocessed rows missing balance features: {missing_feature_rows}")

    except Exception:
        conn.rollback()
        print("Processing failed; this batch was rolled back.")
        raise
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
