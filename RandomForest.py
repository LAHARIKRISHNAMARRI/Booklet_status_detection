import pandas as pd
import matplotlib.pyplot as plt
import os
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Check current working directory
print("Current working directory:", os.getcwd())

# Load dataset
try:
    df = pd.read_csv('C:/Users/Lenovo/PycharmProjects/pythonProject/Booklets_100.csv')  # Adjust path as needed
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: The file path is incorrect or the file does not exist.")
    raise

# Verify column names and data types
print("\nDataset Columns:", df.columns)
print("Data Types:\n", df.dtypes)
print("First Few Rows:\n", df.head())

# Step 1: Calculate summary metrics
try:
    total_students = df['Student_ID'].nunique()
    total_collected = df['Collected'].sum()
    total_missing = (df['Collected'] == 0).sum()
    missing_codes = df.loc[df['Collected'] == 0, 'Booklet_Code'].tolist()
    total_corrected = (df['Corrected'] == 1).sum()

    # Display summary information
    print("\nTotal Students:", total_students)
    print("Total Booklets Collected:", total_collected)
    print("Total Booklets Missing:", total_missing)
    print("Missing Booklet Codes:", missing_codes)
    print("Total Corrected Booklets:", total_corrected)
except KeyError as e:
    print(f"Error: {e} column is missing from the dataset.")
    raise

# Step 2: Add 'Status' based on Collected and Corrected columns
try:
    df['Status'] = df.apply(lambda row: 'Missing' if row['Collected'] == 0 else
                            ('Corrected' if row['Corrected'] == 1 else 'Not Corrected'), axis=1)
    print("\nStatus column added.")
except KeyError as e:
    print(f"Error: {e} column is missing when creating the 'Status' column.")
    raise

# Save the processed data
processed_path = 'C:/Users/Lenovo/PycharmProjects/pythonProject/Processed_Booklets.csv'
df.to_csv(processed_path, index=False)
print(f"Processed data saved to {processed_path}")

# Extended Analysis: Correction Summary by Student
try:
    correction_summary = df.groupby('Student_ID')['Corrected'].mean() * 100
    correction_summary = correction_summary.reset_index().rename(columns={'Corrected': 'Percent_Corrected'})
    print("\nCorrection Summary by Student:")
    print(correction_summary)
except KeyError as e:
    print(f"Error: {e} column is missing when calculating correction summary.")
    raise

# Function to send alert to Telegram
def send_telegram_alert(message):
    bot_token = '7927743887:AAH1Xq0Lz-gKZ6fGs1y20LJ_cazZI0E9yOw'  # Your bot token
    chat_ids = [6314789907, 5602736198]  # Your chat IDs
    url = f'https://api.telegram.org/bot{bot_token}/sendMessage'
    # For multiple chat IDs, iterate through them
    for chat_id in chat_ids:
        params = {'chat_id': chat_id, 'text': message}
        response = requests.post(url, params=params)
        if response.status_code == 200:
            print(f"Alert sent successfully to chat_id: {chat_id}.")
        else:
            print(f"Failed to send alert to chat_id {chat_id}:", response.text)

# Alert for Missing Booklets
if total_missing > 0:
    alert_message = f"ALERT: {total_missing} booklets are missing. Missing codes: {missing_codes}"
    send_telegram_alert(alert_message)

# Visualization: Correction Progress by Student
plt.figure(figsize=(12, 6))
plt.bar(correction_summary['Student_ID'], correction_summary['Percent_Corrected'], color='skyblue')
plt.xlabel('Student ID')
plt.ylabel('Percentage of Booklets Corrected')
plt.title('Correction Progress by Student')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Visualization: Distribution of Missing Booklets
if total_missing > 0:
    plt.figure(figsize=(8, 5))
    plt.hist(df.loc[df['Collected'] == 0, 'Booklet_Code'], bins=20, color='salmon', edgecolor='black')
    plt.xlabel('Booklet Code')
    plt.ylabel('Frequency of Missing Booklets')
    plt.title('Distribution of Missing Booklet Codes')
    plt.tight_layout()
    plt.show()
else:
    print("\nNo missing booklets to display in histogram.")

# Visualization: Breakdown by Status Category
status_counts = df['Status'].value_counts()
print("\nStatus Breakdown:\n", status_counts)
if len(status_counts) > 1:
    plt.figure(figsize=(7, 7))
    plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140,
            colors=['lightcoral', 'lightgreen', 'lightgrey'])
    plt.title('Status Distribution of Booklets')
    plt.show()
else:
    print("\nNot enough categories for a pie chart.")

# ---------------------------
# Random Forest Model Section
# ---------------------------

# Convert Status to a numeric variable: 1 for "Corrected", 0 for both "Missing" and "Not Corrected"
df['Status_Num'] = df['Status'].map({'Missing': 0, 'Not Corrected': 0, 'Corrected': 1})

# For this example, we use 'Collected' as our feature.
# You may include additional features as needed.
X = df[['Collected']]
y = df['Status_Num']

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest classifier.
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set and evaluate accuracy.
y_pred = rf_model.predict(X_test)
print("\nRandom Forest Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Display feature importances.
importances = rf_model.feature_importances_
for idx, val in enumerate(importances):
    print(f"Feature: {X.columns[idx]}, Importance: {val}")
