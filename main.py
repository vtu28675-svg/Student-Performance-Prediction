import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Marks': [20, 30, 40, 50, 60, 70, 80, 90]
}

# Create dataframe
df = pd.DataFrame(data)

print("Student Dataset")
print(df)

# Scatter Plot
plt.scatter(df['Study_Hours'], df['Marks'])

plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.title('Study Hours vs Marks')

plt.show()

# Heatmap
sns.heatmap(df.corr(), annot=True)

plt.title("Correlation Heatmap")

plt.show()

# Machine Learning Model
X = df[['Study_Hours']]
y = df['Marks']

model = LinearRegression()

model.fit(X, y)

# Prediction
predicted_marks = model.predict([[5]])

print("\nPredicted Marks for 5 Study Hours:", predicted_marks[0])