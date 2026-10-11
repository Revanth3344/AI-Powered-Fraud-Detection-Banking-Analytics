import pandas as pd

# Load dataset
data = pd.read_csv("dataset.csv")

# Create ML dataset
ml_data = data[[
    "step",
    "type",
    "amount",
    "isFraud"
]].copy()

print("Original dataset shape:", data.shape)
print("ML dataset shape:", ml_data.shape)

print("\nML dataset:")
print(ml_data.head())

print("\nTransaction types:")
print(ml_data["type"].value_counts())

# Convert transaction type into numerical columns
ml_data = pd.get_dummies(
    ml_data,
    columns=["type"],
    dtype=int
)

print("\nAfter encoding:")
print(ml_data.head())

print("\nNew columns:")
print(ml_data.columns.tolist())

# Create hour of day from step
ml_data["hour"] = ml_data["step"] % 24

print("\nAfter feature engineering:")
print(ml_data.head())

X = ml_data.drop("isFraud", axis=1)
y = ml_data["isFraud"]

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nX columns:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())

#dividing 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nFraud distribution in training:")
print(y_train.value_counts())

print("\nFraud distribution in testing:")
print(y_test.value_counts())

class_weight="balanced"


#Training

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed!")

y_pred = model.predict(X_test)

print("Predictions completed!")
print(y_pred[:20])

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

tree_model.fit(X_train, y_train)

print("\nDecision Tree training completed!")

tree_pred = tree_model.predict(X_test)

print("\nDecision Tree predictions completed!")
print(tree_pred[:20])

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, tree_pred))

tree_cm = confusion_matrix(y_test, tree_pred)

print("\nDecision Tree Confusion Matrix:")
print(tree_cm)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Create the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# 2. Train the model
rf_model.fit(X_train, y_train)

print("\nRandom Forest training completed!")

# 3. Make predictions on unseen test data
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest predictions completed!")

# 4. Classification report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))

# 5. Confusion matrix
rf_cm = confusion_matrix(y_test, rf_pred)

print("\nRandom Forest Confusion Matrix:")
print(rf_cm)