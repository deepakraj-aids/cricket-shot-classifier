import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Combine all CSVs
data_dir = 'pose_data'
all_data = []

for file in os.listdir(data_dir):
    if file.endswith('.csv'):
        path = os.path.join(data_dir, file)
        df = pd.read_csv(path)
        if not df.empty and 'label' in df.columns:
            all_data.append(df)
        else:
            print(f"⚠️ Skipped empty or invalid file: {file}")

# Check if any valid data was loaded
if not all_data:
    print("❌ No valid pose data found. Please collect more data.")
    exit()

# Step 2: Prepare dataset
data = pd.concat(all_data, ignore_index=True)
print(f"✅ Total samples loaded: {len(data)}")

if data.empty:
    print("❌ Combined data is empty. Exiting.")
    exit()

X = data.drop('label', axis=1)
y = data['label']

# Step 3: Scaling
scaler = StandardScaler()
try:
    X_scaled = scaler.fit_transform(X)
except ValueError as e:
    print("❌ Scaling failed:", e)
    exit()

# Step 4: Train-Test Split and Training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 5: Accuracy
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {acc * 100:.2f}%")

# Step 6: Save model and scaler
joblib.dump(clf, 'pose_classifier.joblib')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model and scaler saved successfully.")