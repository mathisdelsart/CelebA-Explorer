import pandas as pd

data = pd.read_csv('Data/abalone.csv')

print("Number of rows: ", data.shape[0])
print("Number of columns: ", data.shape[1])
print("The 10 first rows:\n", data.head(10))
print("Categorical columns: ", data.select_dtypes(include=['object']).columns)
print("Numerical columns: ", data.select_dtypes(include=['int64', 'float64']).columns)
print("Number of missing values: ", data.isnull().sum().sum())
print("Statistics: ")
print(data.describe())