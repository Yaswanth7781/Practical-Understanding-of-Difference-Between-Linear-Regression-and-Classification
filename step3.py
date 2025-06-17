import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Map numeric targets to names for clarity
df['species_name'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# Filter only Setosa (0) and Versicolor (1)
binary_df = df[df['species'].isin([0, 1])]

# Features and labels
X_bin = binary_df[['petal length (cm)', 'petal width (cm)']]  # You can choose any two features
y_bin = binary_df['species']

# Train-test split
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)

# Train logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train_bin, y_train_bin)


# Define mesh grid for plotting decision boundary
x_min, x_max = X_bin.iloc[:, 0].min() - 1, X_bin.iloc[:, 0].max() + 1
y_min, y_max = X_bin.iloc[:, 1].min() - 1, X_bin.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict on mesh grid
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary and points
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
sns.scatterplot(data=binary_df, x='petal length (cm)', y='petal width (cm)', hue='species_name', palette='coolwarm')
plt.title("Decision Boundary: Logistic Regression (Setosa vs Versicolor)")
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Species')
plt.show()



# Predict on test set
y_pred_bin = logreg.predict(X_test_bin)

# Accuracy
acc = accuracy_score(y_test_bin, y_pred_bin)

# Precision, Recall, F1-score
precision = precision_score(y_test_bin, y_pred_bin)
recall = recall_score(y_test_bin, y_pred_bin)
f1 = f1_score(y_test_bin, y_pred_bin)

# Print metrics
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

print()
print(classification_report(y_test_bin, y_pred_bin, target_names=['Setosa', 'Versicolor']))
