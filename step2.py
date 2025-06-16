import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Iris dataset again if needed
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Define features and target
X = df[['sepal length (cm)']]
y = df['sepal width (cm)']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict y values for the training set
y_pred_line = model.predict(X_train)

# Plot scatter and regression line
plt.figure(figsize=(6, 4))
sns.scatterplot(x=X_train['sepal length (cm)'], y=y_train, label='Training Data')
plt.plot(X_train['sepal length (cm)'], y_pred_line, color='red', label='Regression Line')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Linear Regression: Sepal Length vs Sepal Width')
plt.legend()
plt.show()

# Predict on the test set
y_pred_test = model.predict(X_test)

# Calculate Mean Squared Error and R-squared score
mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("Mean Squared Error (MSE):", round(mse, 4))
print("R-squared (R²) Score:", round(r2, 4))

