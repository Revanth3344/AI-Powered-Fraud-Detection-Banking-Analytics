import mysql.connector
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==========================================
# 1. CONNECT TO MYSQL
# ==========================================

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="3344",
    database="banking_fraud_system"
)

cursor = conn.cursor(dictionary=True)

try:
    print("Connected to banking_fraud_system.")

    # ==========================================
    # 2. EXTRACT CUSTOMER TRANSACTION FEATURES
    # ==========================================

    query = """
        SELECT
            ac.account_id,
            COUNT(t.transaction_id) AS total_transactions,
            AVG(t.amount) AS average_transaction_amount,
            SUM(t.amount) AS total_transaction_amount,
            COUNT(DISTINCT t.transaction_type)
                AS transaction_type_diversity
        FROM accounts ac
        JOIN transactions t
            ON ac.account_id = t.account_id
        GROUP BY ac.account_id
        HAVING COUNT(t.transaction_id) > 0
    """

    cursor.execute(query)
    rows = cursor.fetchall()

    if len(rows) < 3:
        raise ValueError(
            "At least 3 accounts with transactions are required."
        )

    df = pd.DataFrame(rows)

    print(f"Accounts analysed: {len(df)}")

    # ==========================================
    # 3. PREPARE FEATURES
    # ==========================================

    feature_columns = [
        "total_transactions",
        "average_transaction_amount",
        "total_transaction_amount",
        "transaction_type_diversity"
    ]

    df[feature_columns] = df[feature_columns].astype(float)

    # Handle any invalid or missing numeric values
    df[feature_columns] = df[feature_columns].fillna(0)

    # ==========================================
    # 4. STANDARDIZE FEATURES
    # ==========================================

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        df[feature_columns]
    )

    # ==========================================
    # 5. APPLY K-MEANS CLUSTERING
    # ==========================================

    number_of_clusters = min(3, len(df))

    kmeans = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )

    df["cluster_label"] = kmeans.fit_predict(
        scaled_features
    )

    print("\nClustering completed.")

    # ==========================================
    # 6. SAVE CUSTOMER BEHAVIOUR SEGMENTS
    # ==========================================

    insert_query = """
        INSERT INTO customer_behavior_segments (
            account_id,
            total_transactions,
            average_transaction_amount,
            total_transaction_amount,
            transaction_type_diversity,
            cluster_label,
            analysis_date
        )
        VALUES (%s, %s, %s, %s, %s, %s, NOW())
        ON DUPLICATE KEY UPDATE
            total_transactions = VALUES(total_transactions),
            average_transaction_amount =
                VALUES(average_transaction_amount),
            total_transaction_amount =
                VALUES(total_transaction_amount),
            transaction_type_diversity =
                VALUES(transaction_type_diversity),
            cluster_label = VALUES(cluster_label),
            analysis_date = NOW()
    """

    for _, row in df.iterrows():
        cursor.execute(
            insert_query,
            (
                int(row["account_id"]),
                int(row["total_transactions"]),
                round(float(row["average_transaction_amount"]), 2),
                round(float(row["total_transaction_amount"]), 2),
                int(row["transaction_type_diversity"]),
                int(row["cluster_label"])
            )
        )

    conn.commit()

    # ==========================================
    # 7. DISPLAY RESULTS
    # ==========================================

    print("\n========== CUSTOMER BEHAVIOUR SEGMENTS ==========")

    print(
        df[
            [
                "account_id",
                "total_transactions",
                "average_transaction_amount",
                "total_transaction_amount",
                "transaction_type_diversity",
                "cluster_label"
            ]
        ].to_string(index=False)
    )

    print(
        "\nCustomer behaviour segments saved successfully!"
    )

except Exception:
    conn.rollback()
    raise

finally:
    cursor.close()
    conn.close()