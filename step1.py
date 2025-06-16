# Step 1: Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load Iris dataset from sklearn
iris = load_iris()

# Convert to DataFrame for easier handling
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Display first 10 rows
print(df.head(10))

# Plot histograms for each feature
df.iloc[:, :-1].hist(bins=15, figsize=(10, 6), edgecolor='black')
plt.suptitle('Feature Distributions in Iris Dataset', fontsize=16)
plt.tight_layout()
plt.show()

# Scatter plot of Sepal Length vs Sepal Width
plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=df, 
    x='sepal length (cm)', 
    y='sepal width (cm)', 
    hue='species'
)
plt.title('Sepal Length vs Sepal Width by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend(title='Species')
plt.show()

# Split dataset for regression task (sepal length vs sepal width)
X = df[['sepal length (cm)']]
y = df['sepal width (cm)']

# 80:20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Show split sizes
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


