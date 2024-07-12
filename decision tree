import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz 
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import graphviz

# Load the dataset with space delimiter
file_path = 'housing.csv.xls'
df = pd.read_csv(file_path, delim_whitespace=True)

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

# Convert the target variable into a binary classification problem for demonstration
y_binary = np.where(y > y.median(), 1, 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binary, test_size=0.2, random_state=42)

# Decision Tree Classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred = tree_clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Decision Tree Classifier Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')


OUTPUT:

Decision Tree Classifier Accuracy: 0.7920792079207921
Confusion Matrix:
[[43 12]
 [ 9 37]]
