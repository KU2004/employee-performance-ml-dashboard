import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

from data_generator import generate_data
from preprocess import preprocess_data

# =============================
# 1. Generate & Save Data
# =============================
data = generate_data()

data.to_csv("data/employee_data.csv", index=False)
print("✅ Dataset saved")

# =============================
# 2. Preprocess
# =============================
data = preprocess_data(data)

# =============================
# 3. Split
# =============================
X = data.drop("Performance_Label", axis=1)
y = data["Performance_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================
# 4. Train Model (Advanced)
# =============================
model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# =============================
# 5. Evaluation
# =============================
y_pred = model.predict(X_test)

print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")

# =============================
# 6. Feature Importance
# =============================
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.savefig("outputs/feature_importance.png")

# =============================
# 7. Save Model
# =============================
joblib.dump(model, "models/model.pkl")

print("\n✅ Model saved")