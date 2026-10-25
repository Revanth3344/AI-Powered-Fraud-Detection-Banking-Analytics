BANKSPHERE WEBSITE UPDATE

Copy/merge app.py, templates/, static/, and requirements.txt into your existing Fraud Detection Project folder. Do not delete your dataset or model folders.

Install/update packages:
python -m pip install -r requirements.txt

Start:
python app.py

Open:
http://127.0.0.1:5000

New feature:
Open AI explainability from the sidebar or visit /explain. Enter a transaction ID with all balance features populated. The page uses trained_models_v2/random_forest.pkl and feature_columns.pkl, recalculates a prediction, and calculates SHAP contributions. Stored database prediction is displayed separately for comparison.

Database configuration is read from environment variables DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT, with local development defaults in app.py. Check those settings against your MySQL setup.

This version uses the same feature formulas shown in your uploaded fraud_explanation.py and run_fraud_detection.py. The website is for local demo use; do not expose Flask debug mode publicly.

Final UI polish: blue circular star cursor, borderless BOB with a small red/black mask icon, BOB-only label, and compact top-aligned cinematic quotes.
