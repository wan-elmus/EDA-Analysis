
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the churn dataset
df = pd.read_csv('churn_data.csv', 
index_col = 'customerID')
df.tail()
df.head()

# Filtering data
df['Churn'].value_counts()

churn_df = df[df['Churn'] == 'Churn']
churn_df['Churn'].unique()

# Filter the churn dataset for customers with tenure greater than 50
long_tenure_df = df[df['tenure'] > 50]
long_tenure_df.head()

# Demonstrate equivalence between boolean and numerical comparisons
False == 0

ax = sns.boxplot(data=df, orient='h')
ax.set_xscale("log")


# Check for outliers in numeric data
numeric_columns = df.select_dtypes(include=np.number).columns
print(numeric_columns)
sns.boxplot(data=df[numeric_columns])
plt.title("Boxplot of Numeric Columns")
plt.show()

# Deal with outliers: Clipping outliers to upper and lower bounds
for column in numeric_columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    upper_boundary = q3 + 1.5 * iqr
    lower_boundary = q1 - 1.5 * iqr
    df[column] = df[column].clip(lower=lower_boundary, upper=upper_boundary)


# Check for missing values
print("Missing Values Before Handling:")
print(df.isna().sum())

# Drop rows with missing values
df.dropna(inplace=True)

# Convert categorical columns to numeric
df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})
df['Churn'].value_counts()

df['PhoneService'] = df['PhoneService'].replace({'No': 0, 'Yes': 1})

df.info()

# Feature Engineering

df_copy = df.copy()
df_copy['MonthlyCharges'] = np.log(df_copy['MonthlyCharges'])

df_copy['MonthlyCharges'].plot.hist()

# Create a new feature: 
# 1. Ratio of TotalCharges to Tenure
df['TotalCharges_Tenure_Ratio'] = df['TotalCharges'] / df['Tenure']

# Create plots for the new feature
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x='TotalCharges', y='Tenure', hue='Churn', data=df)
plt.title("Scatter Plot: TotalCharges vs. Tenure")

plt.subplot(1, 2, 2)
sns.histplot(df['TotalCharges_Tenure_Ratio'], kde=True, bins=30)
plt.title("Histogram: TotalCharges_Tenure_Ratio")

plt.tight_layout()
plt.show()

# Create a new feature: 
# 1. Ratio of MonthlyCharges to TotalCharges
df['MonthlyCharges_to_TotalCharges_Ratio'] = df['MonthlyCharges'] / df['TotalCharges']

# Create plots for the new feature
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Churn', data=df)
plt.title("Scatter Plot: MonthlyCharges vs TotalCharges")

plt.subplot(1, 2, 2)
sns.histplot(df['MonthlyCharges_to_TotalCharges_Ratio'], kde=True, bins=30)
plt.title("Histogram: TMonthlyCharges_to_TotalCharges_Ratio")

plt.tight_layout()
plt.show()

# Save the cleaned and prepared data to a new CSV file
df.to_csv('prepped_churn_data.csv', index=False)

# Advanced section
# Advanced Outlier detection

numeric_df = df.select_dtypes(exclude=['object'])

# Then we scale it so the standard deviation of all columns are 1 and the means of all columns are 0, called standardization:

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(numeric_df)

sns.boxplot(data = scaled_numeric)

# We set contamination very low so we only get the most extreme values. We do need to install the pyod package before running this next code section.

from pyod.models.knn import KNN

od = KNN(contamination=0.01)
od.fit(scaled_numeric)

outliers = od.predict(scaled_numeric)
outliers

outliers.sum()

df[outliers.astype('bool')]

# Filling missing values with ML
df_missing = numeric_df.copy()
df_missing.loc[df['tenure'] == 45, 'tenure'] = np.nan
df_missing.head()
df_missing.info()
df_missing.isna().sum()

from sklearn.impute import KNNImputer
imputer = KNNImputer()
filled_values = imputer.fit_transform(df_missing)
filled_df = pd.DataFrame(data=filled_values, columns=numeric_df.columns, index=numeric_df.index)

obj_df = df.select_dtypes(include=['object'])

# merge the two dfs back into one
full_df = pd.concat([filled_df, obj_df], axis=1)
full_df.head()
full_df.info()
full_df.isna().sum()

df['tenure'].value_counts()

full_df['tenure'].value_counts()

# Other methods to convert categorical to numeric data
# If we have many categorical values, we can use sklearn's label encoder to preprocess them.

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
contract_le = le.fit_transform(df['Contract'])

print(contract_le)
print(le.classes_)

# We can also use one-hot encoding for multi-category variables. This is fairly easy with pandas:

pd.get_dummies(df['PaymentMethod'])

# With more than 2 values, it will create more columns. Since we can always infer one of the columns from all others (e.g. if all others are 0, we know the last column should be a 1), we can drop one:
pd.get_dummies(df['PaymentMethod'], drop_first=True)

# We can then combine this with the original dataframe:

one_hot_df = pd.concat([df.drop('PaymentMethod', axis=1), pd.get_dummies(df['PaymentMethod'], drop_first=True)],axis=1)
one_hot_df

# The Yeo-Johnson transform

from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer()
df['yj_tenure'] = pt.fit_transform(df[['tenure']])
df[['yj_tenure', 'tenure']].plot.density(subplots=True, sharex=False)

# We can see the YJ-transformed data centers around 0 (it has also been standardized with `StandardScaler`) and looks much more symmetric and normal than the unmodified data.
