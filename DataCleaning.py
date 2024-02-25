import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Adjust display options to show all rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

# Generate summary statistics using describe() for all columns
summary = X.describe()

# Print the summary statistics
print(summary)

# Create histograms for all columns and save the images
for col in X.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(X[col], bins=20, color='skyblue', edgecolor='black')
    plt.title('Histogram of {}'.format(col))
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('./histogram/histogram_{}.png'.format(col))  # Save the histogram as an image
    plt.close()  # Close the current figure to free memory

print("Histograms saved successfully.")

# Define pairs of columns for scatter plots
column_pairs = [('radius_mean', 'texture_mean'), ('perimeter_mean', 'area_mean'), ('smoothness_mean', 'compactness_mean')]

# Create scatter plots for each pair of columns
for pair in column_pairs:
    x_column, y_column = pair
    plt.figure(figsize=(8, 6))
    plt.scatter(X[x_column], X[y_column], color='skyblue')
    plt.title('Scatter Plot of {} vs {}'.format(x_column, y_column))
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)
    plt.show()

# Calculate the correlation matrix
corr_matrix = X.corr()

# Create a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Features')
plt.show()
