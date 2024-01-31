
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pyod.models.knn import KNN
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer

df = pd.read_csv('churn_data.csv')
df.tail()

df.head()

# Filtering based on Contract type
oneyr_contract_df = df[df['Contract'] == 'One year']
oneyr_contract_df

# based on Payment Method
check_df = df[df['PaymentMethod'].str.contains('check')]
check_df

# based on Monthly Charges range
filtered_df = df[(df['MonthlyCharges'] >= 50) & (df['MonthlyCharges'] <= 100)]
filtered_df

df['PhoneService'].value_counts()

# Select numeric columns
numeric_columns = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]

sns.boxplot(data=df[numeric_columns])
plt.title("Boxplot of Numeric Columns")
plt.show()

column = 'TotalCharges'
q1 = df[column].quantile(0.25)
q3 = df[column].quantile(0.75)
iqr = q3 - q1
upper_boundary = q3 + 1.5 * iqr
lower_boundary = q1 - 1.5 * iqr
df[(df[column] < lower_boundary) | (df[column] > upper_boundary)][column]

df['TotalCharges']

df_copy = df.copy()
df_copy.loc[df[column] < lower_boundary, column] = np.nan
df_copy.loc[df[column] > upper_boundary, column] = np.nan

df_copy

df_copy = df.copy()
df_copy[column].clip(lower=lower_boundary, upper=upper_boundary, inplace=True)

df_copy

sns.boxplot(data=df_copy, orient='h')

df_copy.isna().sum()

df_copy['TotalCharges'].fillna(df_copy['TotalCharges'].median(), inplace=True)

df_copy.isna().sum()

df_copy['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})
df_copy['Churn'].value_counts()

df_copy['PhoneService'] = df_copy['PhoneService'].replace({'No': 0, 'Yes': 1})
df_copy['PhoneService'].value_counts()

df.info()

# Log transform
df_copy['MonthlyCharges'] = np.log(df_copy['MonthlyCharges'])
df_copy

sns.histplot(df_copy['MonthlyCharges'], kde=True, bins=30)
plt.title("Histogram: MonthlyCharges log transformed")

df_copy['MonthlyCharges_Tenure_Ratio'] = df_copy['MonthlyCharges'] / df_copy['tenure']

plt.figure(figsize=(12, 5))

sns.scatterplot(x='MonthlyCharges', y='tenure', hue='Churn', data=df_copy)
plt.title("Scatter Plot: MonthlyCharges to tenure Ratio")

plt.show()

# plt.figure(figsize = (12,5))
sns.histplot(df_copy['MonthlyCharges_Tenure_Ratio'], kde=True, bins=30)
plt.title("Histogram: MonthlyCharges to tenure Ratio")

plt.show()

df_copy['MonthlyCharges_to_TotalCharges_Ratio'] = df_copy['MonthlyCharges'] / df_copy['TotalCharges']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 2)
sns.histplot(df_copy['MonthlyCharges_to_TotalCharges_Ratio'], kde=True, bins=30)
plt.title("Histogram: MonthlyCharges_to_TotalCharges_Ratio")

plt.subplot(1, 2, 1)
sns.scatterplot(x='MonthlyCharges', y='TotalCharges', hue='Churn', data=df_copy)
plt.title("Scatter Plot: MonthlyCharges vs TotalCharges")

plt.tight_layout()
plt.show()

df_copy['customerID'] = range(1, len(df_copy) + 1)

df_copy.to_csv('prepped_churn_data.csv', index=False)

df_copy

numeric_df = df.select_dtypes(exclude=['object'])

scaler = MinMaxScaler()
scaled_numeric = scaler.fit_transform(numeric_df)

sns.boxplot(data = scaled_numeric)

# We set contamination very low so we only get the most extreme values.
od = IsolationForest(contamination=0.01)
od.fit(scaled_numeric)

outliers = od.predict(scaled_numeric)
outliers

outliers.sum()

df[outliers == -1]

df_missing = numeric_df.copy()
df_missing.loc[df['TotalCharges'] == 75, 'TotalCharges'] = np.nan
df_missing.head()

df_missing.info()

df_missing.isna().sum()

imputer = KNNImputer()
filled_values = imputer.fit_transform(df_missing)
filled_df = pd.DataFrame(data=filled_values, columns=numeric_df.columns, index=numeric_df.index)

obj_df = df.select_dtypes(include=['object'])

full_df = pd.concat([filled_df, obj_df], axis=1)
full_df.head()

full_df.info()

full_df.info()

full_df.isna().sum()

pd.get_dummies(df['Contract'])

pd.get_dummies(df['PaymentMethod'])

# Combine with the original dataframe

one_hot_df = pd.concat([df.drop('PaymentMethod', axis=1), pd.get_dummies(df['PaymentMethod'], drop_first=True)],axis=1)

one_hot_df

one_hot_df_c = pd.concat([one_hot_df.drop('Contract', axis=1), pd.get_dummies(df['Contract'], drop_first=True)],axis=1)

one_hot_df_c

pt = PowerTransformer()
df['yj_tenure'] = pt.fit_transform(df[['tenure']])
df[['yj_tenure', 'tenure']].plot.density(subplots=True, sharex=False)

