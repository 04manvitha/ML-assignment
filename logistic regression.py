import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# Logistic Regression Model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Logistic Regression Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')



OUTPUT:
Logistic Regression Accuracy: 0.8217821782178217
Confusion Matrix:
[[46  9]
 [ 9 37]]
Classification Report:
              precision    recall  f1-score   support

           0       0.84      0.84      0.84        55
           1       0.80      0.80      0.80        46

    accuracy                           0.82       101
   macro avg       0.82      0.82      0.82       101
weighted avg       0.82      0.82      0.82       101
  
