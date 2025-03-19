import joblib
import pandas as pd
from sklearn.metrics import r2_score

# Load the trained model
model = joblib.load('linear_regression_model.pkl')  # Ensure this file is in the same folder

# Load your test dataset
data = pd.read_csv('final_cleaned_dataset.csv')  # Replace with the correct file name

# Display the first few rows to verify
print("Dataset Preview:")
print(data.head())

# Check if the target column 'modal_price' exists
if 'modal_price' not in data.columns:
    raise ValueError("Target column 'modal_price' not found in the dataset. Check column names:", data.columns)

# Split data into features (X) and target (y)
X_test = data.drop(columns=['modal_price'])  # Input features
y_test = data['modal_price']

# Make predictions
y_pred = model.predict(X_test)

# Calculate R² score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.4f}")

# Save predictions to a new CSV file
data['Predicted_Modal_Price'] = y_pred
data.to_csv('predicted_results.csv', index=False)
print("Predictions saved to 'predicted_results.csv'.")
