
import pandas as pd
import mysql.connector

# ==================================================
# 1. CONFIGURATION
# ==================================================

DATASET_PATH = "dataset.csv"
IMPORT_LIMIT = 1000

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "3344",
    "database": "banking_fraud_system"
}


# ==================================================
# 2. CONNECT TO MYSQL
# ==================================================

conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()

try:
    # ==================================================
    # 3. LOAD AND VALIDATE DATASET
    # ==================================================

    df = pd.read_csv(DATASET_PATH).head(IMPORT_LIMIT).copy()

    required_columns = [
        "type",
        "amount",
        "nameDest",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    df = df.dropna(subset=required_columns)

    for column in [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(subset=[
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest"
    ])

    df = df[df["amount"] >= 0].copy()

    if df.empty:
        raise ValueError("No valid transactions found in the dataset.")

    # ==================================================
    # 4. GET VALID FOREIGN KEYS
    # ==================================================

    cursor.execute("SELECT account_id FROM accounts ORDER BY account_id")
    account_ids = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT merchant_id FROM merchants ORDER BY merchant_id")
    merchant_ids = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT location_id FROM locations ORDER BY location_id")
    location_ids = [row[0] for row in cursor.fetchall()]

    if not account_ids:
        raise ValueError("No accounts exist in the accounts table.")

    # ==================================================
    # 5. GENERATE TRANSACTION IDS
    # ==================================================

    cursor.execute("""
        SELECT COALESCE(MAX(transaction_id), 0)
        FROM transactions
    """)

    next_id = cursor.fetchone()[0] + 1

    # ==================================================
    # 6. INSERT TRANSACTIONS WITH V3 FEATURES
    # ==================================================

    insert_query = """
        INSERT INTO transactions (
            transaction_id,
            account_id,
            merchant_id,
            location_id,
            amount,
            transaction_type,
            transaction_time,
            payment_method,
            device_type,
            recipient_account,
            transaction_status,
            oldbalance_org,
            newbalance_orig,
            oldbalance_dest,
            newbalance_dest
        )
        VALUES (
            %s, %s, %s, %s, %s, %s,
            NOW(), %s, %s, %s, %s,
            %s, %s, %s, %s
        )
    """

    inserted = 0

    for index, (_, row) in enumerate(df.iterrows()):

        transaction_type = str(row["type"]).strip().upper()

        account_id = account_ids[index % len(account_ids)]

        merchant_id = (
            merchant_ids[index % len(merchant_ids)]
            if merchant_ids else None
        )

        location_id = (
            location_ids[index % len(location_ids)]
            if location_ids else None
        )

        recipient = str(row["nameDest"])

        cursor.execute(
            insert_query,
            (
                next_id,
                account_id,
                merchant_id,
                location_id,
                float(row["amount"]),
                transaction_type,
                "Online",
                "Mobile",
                recipient,
                "Success",
                float(row["oldbalanceOrg"]),
                float(row["newbalanceOrig"]),
                float(row["oldbalanceDest"]),
                float(row["newbalanceDest"])
            )
        )

        next_id += 1
        inserted += 1

    # ==================================================
    # 7. COMMIT
    # ==================================================

    conn.commit()

    print("\n========== IMPORT SUMMARY ==========")
    print(f"Successfully imported: {inserted}")

    cursor.execute("SELECT COUNT(*) FROM transactions")
    print("Total transactions in database:", cursor.fetchone()[0])

    cursor.execute("""
        SELECT COUNT(*)
        FROM transactions
        WHERE oldbalance_org IS NOT NULL
          AND newbalance_orig IS NOT NULL
          AND oldbalance_dest IS NOT NULL
          AND newbalance_dest IS NOT NULL
    """)

    print(
        "Transactions with complete balance features:",
        cursor.fetchone()[0]
    )

except Exception as error:
    conn.rollback()
    print("Import failed:", error)
    raise

finally:
    cursor.close()
    conn.close()

