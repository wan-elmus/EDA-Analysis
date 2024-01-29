
import pandas as pd
# %matplotlib inline

# we can give an index number or name for our index column, or leave it blank
df = pd.read_excel('diabetes_data.xlsx', index_col='Patient number')
df.tail()

# Filtering data
df['Diabetes'].value_counts()

diabetes_df = df[df['Diabetes'] == 'Diabetes']
diabetes_df['Diabetes'].unique()

# This allows us to plot subsets of the data or to change certain parts of the data. We can use all the other boolean comparison operators, like >, <, >=, !=, and so on.

older_df = df[df['Age'] == 30]
older_df.head()

False == 0

# Checking for outliers
# Outliers are datapoints that are atypical and far outside of the region of normal data, and can affect machine learning models and statistical analyses in negative ways. Sometimes the outliers are mistakes in the data and should be cleaned, other times they are abnormal or atypical datapoints. Depending on what we think is the case, we can throw out datapoints with outliers, treat them as missing values, or clip them to max and min values.

# Classic ways of detecting outliers are the IQR (inter-quartile range) method, z-score (percentiles), or using standard deviations. These rely on the assumption the data is normally distributed (e.g. in a bell curve shape on the histogram). Boxplots use IQR to draw the whiskers, and we can use those to first see if it looks like there are outliers:

import seaborn as sns

ax = sns.boxplot(data=df, orient='h')
#ax.set_xscale("log")

# Yikes, it looks like a lot of outliers for almost all the numeric columns! Using a standard deviation or IQR outlier detection like this relies on our data being near a normal distribution, or bell-shaped. Since much of it is not, and it looks like we have a lot of outliers in all columns, we are not going to take action with these outliers. But if we wanted to take action with the outliers, we could do something like the following: loop through each column, calculate the IQR boundaries, and then do something with these outliers. Here is an example of examining some of the outliers:

column = 'Cholesterol'
q1 = df[column].quantile(0.25)
q3 = df[column].quantile(0.75)
iqr = q3 - q1
upper_boundary = q3 + 1.5 * iqr
lower_boundary = q1 - 1.5 * iqr
df[(df[column] < lower_boundary) | (df[column] > upper_boundary)][column]

# df_copy['Cholesterol'].info()

# One option: set values as missing. Then we can fill them or drop the values as needed in the next section.

# The `.at` indexer for pandas allows us to set values by providing row selections and a column.

import numpy as np

# make a copy so as to to alter the original data
df_copy = df.copy().reset_index()
df_copy.loc[df[column] < lower_boundary, column] = np.nan
df_copy.loc[df[column] > upper_boundary, column] = np.nan

# Another option: clip values to outlier boundaries:
df_copy = df.copy()
df_copy[column].clip(lower=lower_boundary, upper=upper_boundary, inplace=True)

# We can see this clipping removed outliers from our boxplot, but of course we altered the data:

sns.boxplot(data=df_copy, orient='h')

# In our case, we will assume the data is OK and the outliers are simply a function of the small amount of data and the erratic nature of biological measurements, especially with people who have diabetes. For example, the glucose measurements for diabetics are all over the map.

# Missing values
# Similar to outliers, we can deal with missing values in a few ways: drop the data, or fill it. We can fill the data (impute it) with a few methods:
# - mean: good when distrubtion is near normal (like the height of people in a city)
# - median: works well when distribution is skewed (like housing prices)
# - mode: good for categorical data
# - machine learning: good for complex situations or to maximize the effect of your data cleaning

missing = df.copy()
missing.loc[df['Age'] == 30, 'Age'] = np.nan
missing.isna().sum()

# this would drop any rows with at least 1 missing value
missing.dropna(inplace = True)
missing.isna().sum()

# If we had some missing Glucose values, we might fill those with the median:

df['Glucose'].fillna(df['Glucose'].median(), inplace=True)

# However, remember the distributions were very different between the diabetic and non-diabetic populations. So, ideally, we would fill the missing values for the diabetics and non-diabetics separately.

# Converting Categorical variables to numeric

# For using the `sklearn` machine learning library, we need all data as numeric types, but we have two `object` datatypes which are strings. There are many ways to convert a string column to a numeric column. If the values are `'True'`/`'False'`, we can use `.astype('int')`. Otherwise, a few ways are to use pandas functions like `map`, `replace`, and `apply`. `map` is the most computationally efficient usually, but replace is a little more forgiving and flexible. `map` will change values to NaN if they don't match any keys in the dictionary we provide, whereas replace can replace part or all of the data. However, `map` is computationally faster (it runs faster) so would work better for bigger data or in a production setting.

df['Diabetes'] = df['Diabetes'].replace({'No diabetes': 0, 'Diabetes': 1})
df['Diabetes'].value_counts()

df['Gender'] = df['Gender'].replace({'male': 0, 'female': 1})

# check that all columns are numbers now
df.info()

# If we had more than 2 categories, we can simply add more entries to our dictionary (or use other methods in the advanced section):

df['Diabetes'] = df['Diabetes'].replace({'No diabetes': 0, 'Pre diabetes': 1, 'Diabetes': 2})

# Feature engineering - combining features

# Feature engineering can be an important part of data science when using machine learning. The features, or inputs, we provide to our ML algorithm will influence its performance. A few feature engineering techniques are:

# - mathematical transforms (log, Yeo-Johnson, etc)
# - combining columns
# - extracting features from datetimes

# We'll look at scaling data with a log transform and combining columns here.

# To scale data with a log transform, we can simply use numpy. This transform can be useful for highly skewed data, like our HDL cholesterol measurements.

import numpy as np

df_copy = df.copy()
df_copy['HDL Chol'] = np.log(df_copy['HDL Chol'])

df_copy['HDL Chol'].plot.hist()

df['HDL Chol'].plot.hist()

# We can see how it makes the distribution look more like a normal distribution or bell-curve.

# To combine columns, we simply use normal math. For example, we can create a waist/hip ratio and HDL to total cholesterol ratio, which can be useful in diabetes studies:

df['waist_hip_ratio'] = df['waist'] / df['hip']
df['hdl_chol_ratio'] = df['HDL Chol'] / df['Cholesterol']

df['waist_hip_ratio'].plot.hist()

df['hdl_chol_ratio'].plot.hist()

# Checking results and saving the data
# Now that we have our data cleaned up and prepared, we can save it for future use. Let's give it one last check with `info`, then save it to a CSV.

df.info()

# The data looks good - we have all columns as numeric datatypes and no missing values. We will now save it to a csv:

df.to_csv('prepped_diabetes_data.csv')

# Summary
# Here, we loaded, cleaned, and feature engineered the diabetes dataset. We first inspected the dataset for outliers using the IQR method, and found several outliers. This is likely due to the small amount of data. Because few of the outliers were very isolated from other datapoints and because there were so many outliers, we elected to leave them as-is. We did not find any missing values in the data. We converted the categorical columns to numeric columns with binary label encoding. We created two new features, the waist/hip ratio and HDL/cholesterol ratio. Finally, we saved our data as a CSV and it is ready for the next steps.

# Advanced section
# We will cover:
# - Advanced outlier detection
# - filling missing values with ML
# - Converting categorical to numeric
    # - sklearn labelencoder
    # - one-hot encoding
# - Yeo-Johnson transform

# Although we won't cover it, the `missingno` package is a nice one for visualizing missing values in a dataframe. It's also how pandas-profiling draws some of the missing value plots.

# Advanced outlier detection and feature scaling

# Outlier detection is more involved than you might think. There are new outlier algorithms constantly being developed, and it's an active area of research. The pyod package in Python has many of the cutting-edge algorithms. One problem with many of the cutting-edge algorithms is we need to specify a 'contamination' proportion, which specifies the proportion of outliers. This is usually set by an expectation for the number of outliers that we may know or estimate. The KNN method, which uses distances between points to predict outliers, runs fast and we will use it here. Since it's a distance-based algorithm, it's best to scale our data first before trying to detect outliers with it. However, the scaling doesn't seem to make a big difference here. We can also only use numeric data, so we select those columns first. Let's re-load the data so we're starting from scratch.

df = pd.read_excel('diabetes_data.xlsx', index_col='Patient number')

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

# Now this gives us something to think about. These outliers tend to have one or more measurements on the extreme. For example, the last person seems to be an older woman with a low BMI and a high systolic blood pressure. However, this is again medical data, and the reason we probably have some outliers here is the small amount of data collected.

# Filling missing values with ML

# The KNNImputer from sklearn is fairly easy to use. We simply create the imputer and use the fit_transform method. There are some parameters for the method we could tune as well to try and improve performance. All values going in need to be numeric for this to work.

df_missing = numeric_df.copy()
df_missing.loc[df['hip'] == 39, 'hip'] = np.nan
#df_missing.head()
#df_missing.info()
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

df['hip'].value_counts()

full_df['hip'].value_counts()

# Other methods to convert categorical to numeric data
# If we have many categorical values, we can use sklearn's label encoder to preprocess them.

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
diabetes_le = le.fit_transform(df['Diabetes'])

print(diabetes_le)
print(le.classes_)

# We can also use one-hot encoding for multi-category variables. This is fairly easy with pandas:

pd.get_dummies(df['Gender'])

# With more than 2 values, it will create more columns. Since we can always infer one of the columns from all others (e.g. if all others are 0, we know the last column should be a 1), we can drop one:
pd.get_dummies(df['Gender'], drop_first=True)

# We can then combine this with the original dataframe:

one_hot_df = pd.concat([df.drop('Gender', axis=1), pd.get_dummies(df['Gender'], drop_first=True)],axis=1)
one_hot_df

# The Yeo-Johnson transform
# Much like the log transform can convert our data into a normal-looking (Gaussian or bell-curve) distribution, the Yeo-Johnson can do the same. However, the YJ method is a bit more optimized and advanced.

from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer()
df['yj_HDL'] = pt.fit_transform(df[['HDL Chol']])
df[['yj_HDL', 'HDL Chol']].plot.density(subplots=True, sharex=False)

# We can see the YJ-transformed data centers around 0 (it has also been standardized with `StandardScaler`) and looks much more symmetric and normal than the unmodified data.
