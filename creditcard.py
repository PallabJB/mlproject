import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Load the dataset
url = 'C:/ml/creditcard.csv'  # Replace with the correct path or URL
data = pd.read_csv(url)

# Display the first few rows of the dataset
print(data.head())

# Checking for missing values
print(data.isnull().sum())

# Descriptive statistics to understand the dataset
print(data.describe())

# Visualizing class distribution (Fraud vs Non-Fraud)
plt.figure(figsize=(8,6))
sns.countplot(x='Class', data=data,hue='Class',palette='Set1')
plt.title("Class Distribution (0 - Not Fraud, 1 - Fraud)")
plt.show()

# Correlation Matrix - Correlation of features
correlation_matrix = data.corr()

# Visualizing the Correlation Matrix with a heatmap
plt.figure(figsize=(12,8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True, square=True)
plt.title("Correlation Matrix of Features")
plt.show()

# Data Preprocessing
X = data.drop('Class', axis=1)  # Drop 'Class' (target variable) from features
y = data['Class']  # Target variable

# Handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scaling the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training - Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions with Random Forest
y_pred_rf = rf_model.predict(X_test)

# Evaluating Random Forest model performance
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"\nRandom Forest Classification Report:\n{classification_report(y_test, y_pred_rf)}")

# Confusion Matrix for Random Forest
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Plotting the Confusion Matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Feature Importance for Random Forest
feature_importance = pd.Series(rf_model.feature_importances_, index=data.drop('Class', axis=1).columns)

# Plotting Top 10 Features for Fraud Detection
plt.figure(figsize=(10, 6))
feature_importance.nlargest(10).plot(kind='barh', color='royalblue')
plt.title("Top 10 Features for Fraud Detection")
plt.show()

# Pairplot to visualize pairwise relationships between features
# We will limit this to a smaller subset of features to prevent clutter.
subset_data = data[['V1', 'V2', 'V3', 'V4', 'V5', 'Class']]  # A subset of features to visualize pairwise relations
sns.pairplot(subset_data, hue='Class', palette='Set1')
plt.title("Pairplot of Selected Features")
plt.show()

# Real-time Prediction Simulation: Predicting on a new sample (real-time scenario)
sample_data = X_test[0].reshape(1, -1)  # Taking the first row of the test set for prediction
sample_pred_rf = rf_model.predict(sample_data)

print(f"\nReal-time Prediction (Random Forest): Predicted Fraud (1) or Not Fraud (0): {sample_pred_rf[0]}")
