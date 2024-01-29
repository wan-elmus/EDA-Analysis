
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from pyod.models.knn import KNN
from sklearn.impute import SimpleImputer
from scipy.stats import boxcox
from sklearn.preprocessing import OneHotEncoder

# Load the churn dataset
df = pd.read_csv('churn_data.csv', 
index_col = 'customerID')
df
df.head()

# Filtering data based on 'Churn' values
churn_counts = df['Churn'].value_counts()
churn_counts

# Filter the churn dataset for customers with 'Churn' equal to 'Yes'
churn_df = df[df['Churn'] == 'Yes']
unique_churn_values = churn_df['Churn'].unique()
unique_churn_values

# Filter the dataset for customers with tenure less than 10
least_tenure_df = df[df['tenure'] < 20]
least_tenure_df.head()

# Demonstrate equivalence between boolean and numerical comparisons
False == 0

# Check for outliers using z-scores
z_scores = (df - df.mean()) / df.std()

# Set a z-score threshold for identifying outliers
z_score_threshold = 3

# Identify outliers using z-scores
outliers = (np.abs(z_scores) > z_score_threshold).any(axis=1)

# Visualize outliers using a boxplot
ax = sns.boxplot(data=df, orient='h')
ax.set_xscale("log")
plt.title("Boxplot of Numeric Columns")

# Highlight outliers with red color
outlier_indices = np.where(outliers)[0]
ax.scatter(df.iloc[outlier_indices], np.zeros_like(outlier_indices), color='red', label='Outliers')

plt.show()

# Deal with outliers: Clipping outliers to upper and lower bounds
for column in df.columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    upper_boundary = q3 + 1.5 * iqr
    lower_boundary = q1 - 1.5 * iqr
    df[column] = df[column].clip(lower=lower_boundary, upper=upper_boundary)

# Visualize the data after handling outliers
ax = sns.boxplot(data=df, orient='h')
ax.set_xscale("log")
plt.title("Boxplot of Numeric Columns (After Handling Outliers)")

plt.show()

# Check for missing values
print("Missing Values Before Handling:")
print(df.isna().sum())

# Impute missing values using SimpleImputer with the mean strategy
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Drop rows with missing values
df.dropna(inplace=True)

# Convert categorical columns to numeric
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
df['PhoneService'] = df['PhoneService'].map({'No': 0, 'Yes': 1})

# Using pandas get_dummies
df_dummies = pd.get_dummies(df, columns=['Contract', 'PaymentMethod'], drop_first=True)

# Display information about the DataFrame after encoding
df_dummies.info()


# Convert other categorical columns to numeric using One-Hot Encoding
encoder = OneHotEncoder(drop='first', sparse=False)

# Fit and transform the categorical columns
encoded_columns = pd.DataFrame(encoder.fit_transform(df[['Contract', 'PaymentMethod']]), columns=encoder.get_feature_names(['Contract', 'PaymentMethod']))

# Concatenate the encoded columns with the original DataFrame
df_encoded = pd.concat([df.drop(['Contract', 'PaymentMethod'], axis=1), encoded_columns], axis=1)

# Display information about the DataFrame after encoding
df_encoded.info()

# Display value counts after conversion
print(df['Churn'].value_counts())

# Display information about the DataFrame
df.info()


# Feature Engineering
df_copy = df.copy()

# Transform MonthlyCharges using logarithm
df_copy['MonthlyCharges_log'] = np.log(df_copy['MonthlyCharges'])

# Create a new feature: Ratio of TotalCharges to Tenure
df_copy['TotalCharges_Tenure_Ratio'] = df_copy['TotalCharges'] / df_copy['Tenure']

# Create a new feature: Ratio of MonthlyCharges to TotalCharges
df_copy['MonthlyCharges_to_TotalCharges_Ratio'] = df_copy['MonthlyCharges'] / df_copy['TotalCharges']

# Visualization for the new features
plt.figure(figsize=(15, 5))

# Plot 1: Histogram for MonthlyCharges_log
plt.subplot(1, 3, 1)
df_copy['MonthlyCharges_log'].plot.hist()
plt.title("Histogram: MonthlyCharges_log")

# Plot 2: Scatter Plot for TotalCharges_Tenure_Ratio
plt.subplot(1, 3, 2)
sns.scatterplot(x='TotalCharges', y='Tenure', hue='Churn', data=df_copy)
plt.title("Scatter Plot: TotalCharges vs. Tenure")

# Plot 3: Scatter Plot for MonthlyCharges_to_TotalCharges_Ratio
plt.subplot(1, 3, 3)
sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Churn', data=df_copy)
plt.title("Scatter Plot: MonthlyCharges vs. TotalCharges")

plt.tight_layout()
plt.show()

# Transform customerID column
df_copy['customerID'] = range(1, len(df_copy) + 1)

# Save the cleaned and prepared data to a new CSV file
df_copy.to_csv('new_churn_data.csv', index=False)

# Advanced outlier detection
# Select numeric columns
numeric_columns = df.select_dtypes(exclude=['object']).columns

# Robust scaling of numeric columns
scaler = RobustScaler()
scaled_numeric = scaler.fit_transform(df[numeric_columns])

# Create a boxplot for scaled numeric data
plt.figure(figsize=(10, 6))
sns.boxplot(data=scaled_numeric)
plt.title("Boxplot of Scaled Numeric Columns (RobustScaler)")
plt.show()

od = KNN(contamination=0.01)
od.fit(scaled_numeric)

outliers = od.predict(scaled_numeric)
outliers

outliers.sum()

df[outliers.astype('bool')]

# Box-Cox transform for the 'tenure' column
df['boxcox_tenure'], lambda_value = boxcox(df['tenure'] + 1)  # Adding 1 to handle non-positive values

# Plot the density of the original and transformed data
plt.figure(figsize=(10, 6))
sns.kdeplot(df['tenure'], label='Original Tenure', color='blue')
sns.kdeplot(df['boxcox_tenure'], label='Box-Cox Transformed Tenure', color='orange')
plt.title("Density Plot of Original and Box-Cox Transformed Tenure")
plt.legend()
plt.show()