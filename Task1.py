import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset from seaborn
iris = sns.load_dataset("iris")

# Display the first few rows of the dataset
print(iris.head())

# Pair plot to visualize relationships between numerical variables
sns.pairplot(iris, hue="species", height=2.5)
plt.show()

# Box plot to show the distribution of numerical variables for each species
plt.figure(figsize=(12, 8))
sns.boxplot(x="species", y="sepal_length", data=iris)
plt.title('Box Plot of Sepal Length for Each Species')
plt.show()

# Violin plot to visualize the distribution of numerical variables
plt.figure(figsize=(12, 8))
sns.violinplot(x="species", y="petal_length", data=iris)
plt.title('Violin Plot of Petal Length for Each Species')
plt.show()

# Scatter plot matrix for numerical variables
sns.set(style="whitegrid")
plt.figure(figsize=(10, 8))
sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=iris)
plt.title('Scatter Plot of Sepal Length vs. Sepal Width')
plt.show()
