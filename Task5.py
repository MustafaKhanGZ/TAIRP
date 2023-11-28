import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

titanic_data = sns.load_dataset('titanic')

print(titanic_data.head())

columns_to_drop = ['embarked', 'who', 'adult_male', 'deck', 'embark_town', 'alive']
titanic_data = titanic_data.drop(columns=columns_to_drop, axis=1)

titanic_data['age'].fillna(titanic_data['age'].mean(), inplace=True)
titanic_data.dropna(subset=['fare'], inplace=True) 


titanic_data = pd.get_dummies(titanic_data, columns=['sex', 'class', 'alone'], drop_first=True)

X = titanic_data.drop('survived', axis=1)
y = titanic_data['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

conf_matrix = confusion_matrix(y_test, predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
