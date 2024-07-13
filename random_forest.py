import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Load the dataset with space delimiter
file_path = 'housing.csv.xls'
df = pd.read_csv(file_path, sep='\s+')

# Rename columns for better readability (if needed)
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Handle missing values (if any)
df = df.dropna()

# Split the data into features and target
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Standardize the feature values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert the target variable into a binary classification problem
y_binary = np.where(y > y.median(), 1, 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42)

# Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Random Forest Classifier Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')



OUTPUT:
Random Forest Classifier Accuracy: 0.8514851485148515
Confusion Matrix:
[[45 10]
 [ 5 41]]
Classification Report:
              precision    recall  f1-score   support

           0       0.90      0.82      0.86        55
           1       0.80      0.89      0.85        46

    accuracy                           0.85       101
   macro avg       0.85      0.85      0.85       101
weighted avg       0.86      0.85      0.85       101
