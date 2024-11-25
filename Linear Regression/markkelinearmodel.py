import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the datasets
student_mat = pd.read_csv('student-mat.csv', delimiter=';')
student_por = pd.read_csv('student-por.csv', delimiter=';')

# Combine the datasets
student_data = pd.concat([student_mat, student_por], ignore_index=True)

# Select relevant features and encode categorical variables
features = ['failures', 'Medu', 'Fedu', 'higher', 'age', 'goout', 'romantic', 'traveltime']
target = 'G3'

# Encode categorical variables
student_data['higher_yes'] = student_data['higher'].map({'yes': 1, 'no': 0})
student_data['romantic_yes'] = student_data['romantic'].map({'yes': 1, 'no': 0})

# Updated feature list with encoded columns
selected_features = ['failures', 'Medu', 'Fedu', 'higher_yes', 'age', 'goout', 'romantic_yes', 'traveltime']

# Filter dataset for modeling
data = student_data[selected_features + [target]].dropna()

# Split into features (X) and target (y)
X = data[selected_features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)