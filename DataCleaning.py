import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('cancer dataset.csv')

# Separate diagnosis into y (target variable) and the rest into X (feature variables)
y = df['diagnosis']
X = df.drop(columns=['id', 'diagnosis'])

# Count columns with zero or null values
columns_with_zero_or_null = 0
for col in X.columns:
    if X[col].dtype != 'object':  # Check if the column is numeric
        if X[col].isnull().sum() > 0 or (X[col] == 0).sum() > 0:  # Check if column has null or zero values
            columns_with_zero_or_null += 1

print("Number of columns with zero or null values:", columns_with_zero_or_null)

# Check for zero or null values in each column of X and replace them with the mean value of that column
for col in X.columns:
    if X[col].dtype != 'object':  # Check if the column is numeric
        mean_val = X[col].mean()  # Calculate the mean value of the column
        X[col] = X[col].replace(0, mean_val)  # Replace zero values with the mean value
        X[col].fillna(mean_val, inplace=True)  # Replace null values with the mean value