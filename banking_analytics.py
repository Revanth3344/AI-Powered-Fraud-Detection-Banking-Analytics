import joblib
import shap
import mysql.connector
import pandas as pd


# ==================================================
# 1. CONFIGURATION
# ==================================================

MODEL_PATH = "trained_models_v2/random_forest.pkl"
FEATURE_PATH = "trained_models_v2/feature_columns.pkl"

ALLOWED_STATUSES = [
    "Open",
    "Investigating",
    "Resolved",
    "False Positive"
]


# ==================================================
# 2. DATABASE CONNECTION
# ==================================================

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="3344",
        database="banking_fraud_system"
    )


# ==================================================
# 3. CUSTOMER ANALYTICS
# ==================================================

def show_customer_analytics(cursor):

    cursor.execute("""
        SELECT
            c.customer_id,
            c.full_name AS customer_name,
            COUNT(DISTINCT t.transaction_id) AS total_transactions,
            ROUND(SUM(t.amount), 2) AS total_amount,
            ROUND(AVG(t.amount), 2) AS average_amount,
            COUNT(DISTINCT CASE
                WHEN p.risk_level = 'High'
                THEN t.transaction_id
            END) AS high_risk_transactions
        FROM customers c
        JOIN accounts ac
            ON c.customer_id = ac.customer_id
        JOIN transactions t
            ON ac.account_id = t.account_id
        LEFT JOIN ai_predictions p
            ON t.transaction_id = p.transaction_id
        GROUP BY c.customer_id, c.full_name
        ORDER BY high_risk_transactions DESC, total_amount DESC
    """)

    rows = cursor.fetchall()

    print("\n========== CUSTOMER ANALYTICS ==========")

    if not rows:
        print("No customer transaction data found.")
        return

    for row in rows:
        print(
            f"Customer: {row['customer_name']} | "
            f"Transactions: {row['total_transactions']} | "
            f"Total amount: {row['total_amount']} | "
            f"Average amount: {row['average_amount']} | "
            f"High-risk transactions: {row['high_risk_transactions']}"
        )


# ==================================================
# 4. MERCHANT ANALYTICS
# ==================================================

def show_merchant_analytics(cursor):

    cursor.execute("""
        SELECT
            m.merchant_id,
            m.merchant_name,
            COUNT(DISTINCT t.transaction_id) AS total_transactions,
            ROUND(SUM(t.amount), 2) AS total_amount,
            COUNT(DISTINCT CASE
                WHEN p.risk_level = 'High'
                THEN t.transaction_id
            END) AS high_risk_transactions,
            ROUND(AVG(p.fraud_score), 4) AS average_fraud_score
        FROM merchants m
        JOIN transactions t
            ON m.merchant_id = t.merchant_id
        LEFT JOIN ai_predictions p
            ON t.transaction_id = p.transaction_id
        GROUP BY m.merchant_id, m.merchant_name
        ORDER BY high_risk_transactions DESC, total_amount DESC
    """)

    rows = cursor.fetchall()

    print("\n========== MERCHANT ANALYTICS ==========")

    if not rows:
        print("No merchant transaction data found.")
        return

    for row in rows:
        print(
            f"Merchant: {row['merchant_name']} | "
            f"Transactions: {row['total_transactions']} | "
            f"Total amount: {row['total_amount']} | "
            f"High-risk transactions: {row['high_risk_transactions']} | "
            f"Average fraud score: {row['average_fraud_score']}"
        )


# ==================================================
# 5. OPEN FRAUD ALERTS
# ==================================================

def show_open_alerts(cursor):

    cursor.execute("""
        SELECT
            a.alert_id,
            a.transaction_id,
            a.alert_type,
            a.severity,
            a.status,
            a.created_at,
            p.fraud_score,
            p.risk_level
        FROM fraud_alerts a
        JOIN ai_predictions p
            ON a.prediction_id = p.prediction_id
        WHERE a.status = 'Open'
        ORDER BY p.fraud_score DESC, a.created_at DESC
    """)

    rows = cursor.fetchall()

    print("\n========== OPEN FRAUD ALERTS ==========")

    if not rows:
        print("No open fraud alerts.")
        return

    for row in rows:
        print(
            f"Alert ID: {row['alert_id']} | "
            f"Transaction: {row['transaction_id']} | "
            f"Type: {row['alert_type']} | "
            f"Severity: {row['severity']} | "
            f"Score: {row['fraud_score']} | "
            f"Risk: {row['risk_level']} | "
            f"Status: {row['status']}"
        )


# ==================================================
# 6. CUSTOMER BEHAVIOUR CLUSTERS
# ==================================================

def show_customer_behavior(cursor):

    cursor.execute("""
        SELECT
            s.account_id,
            c.full_name AS customer_name,
            s.total_transactions,
            s.average_transaction_amount,
            s.total_transaction_amount,
            s.transaction_type_diversity,
            s.cluster_label,
            s.analysis_date
        FROM customer_behavior_segments s
        JOIN accounts ac
            ON s.account_id = ac.account_id
        JOIN customers c
            ON ac.customer_id = c.customer_id
        ORDER BY s.cluster_label, s.account_id
    """)

    rows = cursor.fetchall()

    print("\n========== CUSTOMER BEHAVIOUR CLUSTERS ==========")

    if not rows:
        print(
            "No clusters found. "
            "Run customer_behavior_analysis.py first."
        )
        return

    for row in rows:
        print(
            f"Customer: {row['customer_name']} | "
            f"Account: {row['account_id']} | "
            f"Cluster: {row['cluster_label']} | "
            f"Transactions: {row['total_transactions']} | "
            f"Average amount: {row['average_transaction_amount']} | "
            f"Transaction types: {row['transaction_type_diversity']}"
        )


# ==================================================
# 7. LOAD MODEL FOR SHAP EXPLANATIONS
# ==================================================

def load_explanation_model():

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

    explainer = shap.TreeExplainer(model)

    return model, feature_columns, explainer


# ==================================================
# 8. EXPLAIN A TRANSACTION USING SHAP
# ==================================================

def explain_transaction(cursor, model, feature_columns, explainer):

    try:
        transaction_id = int(
            input("\nEnter transaction ID to explain: ").strip()
        )
    except ValueError:
        print("Please enter a valid integer transaction ID.")
        return

    cursor.execute("""
        SELECT
            t.transaction_id,
            t.amount,
            t.transaction_type,
            p.fraud_score AS stored_fraud_score,
            p.risk_level AS stored_risk_level,
            p.prediction AS stored_prediction,
            p.model_version
        FROM transactions t
        LEFT JOIN ai_predictions p
            ON t.transaction_id = p.transaction_id
        WHERE t.transaction_id = %s
    """, (transaction_id,))

    transaction = cursor.fetchone()

    if transaction is None:
        print("Transaction not found.")
        return

    transaction_type = str(
        transaction["transaction_type"]
    ).strip().upper()

    input_data = pd.DataFrame([{
        "amount": float(transaction["amount"]),
        "type": transaction_type
    }])

    input_data = pd.get_dummies(
        input_data,
        columns=["type"],
        dtype=int
    )

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    predicted_class = int(
        model.predict(input_data)[0]
    )

    print("\n========== TRANSACTION EXPLANATION ==========")
    print("Transaction ID:", transaction_id)
    print("Amount:", transaction["amount"])
    print("Transaction type:", transaction_type)
    print(f"Current model fraud probability: {probability:.4f}")

    print(
        "Current model prediction:",
        "Fraud" if predicted_class == 1 else "Legitimate"
    )

    print(
        "Stored prediction:",
        transaction["stored_prediction"]
        if transaction["stored_prediction"] is not None
        else "Not available"
    )

    print(
        "Stored risk level:",
        transaction["stored_risk_level"]
        if transaction["stored_risk_level"] is not None
        else "Not available"
    )

    print(
        "Stored model version:",
        transaction["model_version"]
        if transaction["model_version"] is not None
        else "Not available"
    )

    print("\nCalculating SHAP explanation...")

    shap_values = explainer.shap_values(input_data)

    if isinstance(shap_values, list):
        contributions = shap_values[1][0]

    elif getattr(shap_values, "ndim", 0) == 3:
        contributions = shap_values[0, :, 1]

    else:
        contributions = shap_values[0]

    explanation = pd.DataFrame({
        "feature": feature_columns,
        "value": input_data.iloc[0].values,
        "contribution": contributions
    })

    explanation["absolute_impact"] = (
        explanation["contribution"].abs()
    )

    explanation = explanation.sort_values(
        "absolute_impact",
        ascending=False
    )

    print("\n========== SHAP FEATURE CONTRIBUTIONS ==========")

    for _, row in explanation.iterrows():

        contribution = float(row["contribution"])

        if contribution > 0:
            direction = "increased"
        elif contribution < 0:
            direction = "decreased"
        else:
            direction = "had approximately no effect on"

        print(
            f"{row['feature']} = {row['value']} | "
            f"Contribution: {contribution:.6f} | "
            f"This feature {direction} the model output."
        )

    print(
        "\nNote: SHAP explains the model output. "
        "It does not prove that a transaction is genuinely fraudulent."
    )


# ==================================================
# 9. MANAGE FRAUD ALERTS
# ==================================================

def manage_fraud_alerts(cursor, conn):

    print("\n========== FRAUD ALERT MANAGEMENT ==========")

    # Display recent alerts with their current status.
    cursor.execute("""
        SELECT
            a.alert_id,
            a.transaction_id,
            a.alert_type,
            a.severity,
            a.status,
            a.created_at,
            p.fraud_score
        FROM fraud_alerts a
        LEFT JOIN ai_predictions p
            ON a.prediction_id = p.prediction_id
        ORDER BY a.created_at DESC, a.alert_id DESC
        LIMIT 30
    """)

    alerts = cursor.fetchall()

    if not alerts:
        print("No fraud alerts found.")
        return

    print("\nRecent fraud alerts:")

    for alert in alerts:
        print(
            f"Alert ID: {alert['alert_id']} | "
            f"Transaction: {alert['transaction_id']} | "
            f"Score: {alert['fraud_score']} | "
            f"Severity: {alert['severity']} | "
            f"Status: {alert['status']}"
        )

    try:
        alert_id = int(
            input("\nEnter the Alert ID to update: ").strip()
        )
    except ValueError:
        print("Invalid Alert ID.")
        return

    # Confirm that the requested alert exists.
    cursor.execute("""
        SELECT alert_id, status
        FROM fraud_alerts
        WHERE alert_id = %s
    """, (alert_id,))

    alert = cursor.fetchone()

    if alert is None:
        print("Alert not found.")
        return

    print("\nCurrent status:", alert["status"])
    print("\nAvailable statuses:")

    for index, status in enumerate(ALLOWED_STATUSES, start=1):
        print(f"{index}. {status}")

    try:
        choice = int(input("Select the new status: ").strip())
    except ValueError:
        print("Enter a valid option.")
        return

    if choice < 1 or choice > len(ALLOWED_STATUSES):
        print("Invalid status option.")
        return

    new_status = ALLOWED_STATUSES[choice - 1]

    if new_status == alert["status"]:
        print("The alert already has this status.")
        return

    # Use the stored procedure created in MySQL Workbench.
    try:
        cursor.callproc(
            "UpdateFraudAlert",
            [alert_id, new_status]
        )

        conn.commit()

        cursor.execute("""
            SELECT alert_id, status
            FROM fraud_alerts
            WHERE alert_id = %s
        """, (alert_id,))

        updated_alert = cursor.fetchone()

        print("\nAlert updated successfully!")
        print("Alert ID:", updated_alert["alert_id"])
        print("New status:", updated_alert["status"])

    except mysql.connector.Error as error:
        conn.rollback()
        print("Could not update alert:", error)


# ==================================================
# 10. MAIN APPLICATION
# ==================================================

def main():

    conn = None
    cursor = None

    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)

        print("Connected to banking_fraud_system.")

        model = None
        feature_columns = None
        explainer = None

        while True:

            print("\n========== BANKING FRAUD ANALYTICS ==========")
            print("1. Customer analytics")
            print("2. Merchant analytics")
            print("3. Open fraud alerts")
            print("4. Customer behaviour clusters")
            print("5. Explain a transaction using SHAP")
            print("6. Manage fraud alerts")
            print("7. Exit")

            choice = input("Enter your choice: ").strip()

            try:
                if choice == "1":
                    show_customer_analytics(cursor)

                elif choice == "2":
                    show_merchant_analytics(cursor)

                elif choice == "3":
                    show_open_alerts(cursor)

                elif choice == "4":
                    show_customer_behavior(cursor)

                elif choice == "5":

                    if model is None:
                        print(
                            "Loading Random Forest model "
                            "and SHAP explainer..."
                        )

                        model, feature_columns, explainer = (
                            load_explanation_model()
                        )

                    explain_transaction(
                        cursor,
                        model,
                        feature_columns,
                        explainer
                    )

                elif choice == "6":
                    manage_fraud_alerts(cursor, conn)

                elif choice == "7":
                    print("Exiting application.")
                    break

                else:
                    print("Enter a number between 1 and 7.")

            except (
                mysql.connector.Error,
                ValueError,
                KeyError,
                OSError
            ) as error:

                print("Operation failed:", error)

    except mysql.connector.Error as error:
        print("Database connection failed:", error)

    except (OSError, ImportError) as error:
        print("Model or SHAP loading failed:", error)

    finally:

        if cursor is not None:
            cursor.close()

        if conn is not None and conn.is_connected():
            conn.close()

        print("Database connection closed.")


if __name__ == "__main__":
    main()