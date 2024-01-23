
import pandas as pd
import matplotlib.pyplot as plt

# load data
# we can give an index number or name for our index column, or leave it blank
df = pd.read_excel('data/diabetes_data.xlsx', index_col='Patient number')
df

df.head()
df.tail()

# Using Pandas for EDA and Visualization
# Numeric EDA
# info
df.info()

# Describe
df.describe()

# columns
df.columns

#select a column
df['Age']

# select a colun and get the counts of each unique value
df['Age'].value_counts()

# similar, but only gets unique values
df['Age'].unique()

# Bar plots
# turn value_counts into a bar plot
df['Age'].value_counts()[:10].plot.bar()
plt.xlabel('Age')
plt.ylabel('Counts')

# If you want to hide the printout of text, assign the last line to the _ variable (essentially, throw away the output that gets printed).

df['Age'].value_counts()[:10].plot.bar()
plt.xlabel('Age')
_ = plt.ylabel('Counts')

# Histograms
df['Glucose'].hist()

# this slightly different interface has a different style and generally looks better without gridlines
df['Glucose'].plot.hist()

# Here, we change the number of bars (bins)
df['Glucose'].plot.hist(bins=30)

# Scatter plots
df.plot.scatter(x='Cholesterol', y='Glucose')

# Advanced EDA an Visualization
# Filtering
over_median_age = df[df['Age'] > df['Age'].median()].copy()

# This uses a boolean comparison, which indexes the dataframe and returns rows where the condition is True
df['Age'] > df['Age'].median()

# We can also negate something with the ~ character
~(df['Age'] > df['Age'].median())

# To combine filters, we use the & (and) and | (or) operators, and be careful to wrap each conditional filter within parentheses:

over_median_age_chol = df[(df['Age'] > df['Age'].median()) & (df['Cholesterol'] > df['Cholesterol'].median())].copy()

over_median_age_chol

# We can filter to get the two groups of diabetes and no diabetes people, and then look at the proportions of the genders in the groups (since the numbers in the groups are not the same). The `shape` attribute of a dataframe is a tuple with (rows, columns), so getting the first element with `[0]` gives us the number of rows. It looks like there isn't a large difference in the balance of male/female genders between these two groups.

df.shape
diabetes_df = df[df['Diabetes'] == 'Diabetes']

diabetes_df['Gender'].value_counts() / diabetes_df.shape[0]
no_diabetes_df = df[df['Diabetes'] == 'No diabetes']

no_diabetes_df['Gender'].value_counts() / no_diabetes_df.shape[0]

# This is also what the normalize argument does
no_diabetes_df['Gender'].value_counts(normalize=True)

# Seaborn for plotting
import phik
import seaborn as sns

_ = sns.histplot(data=df, y='Glucose', hue='Diabetes', stat='density', common_norm=False)

# With seaborn, we can create scatter plots abd color them by groups
sns.relplot(data=df, x='Cholesterol', y='Glucose', hue='Diabetes')

# One other nice plot to examine is a correlogram. This shows the linear correlations between columns. We can see the pairs BMI and weight as well as waist and hip measurements are strongly correlated with each other. This is the Pearson correlation, which shows linear relationships between two numeric columns.

sns.heatmap(df.corr())
sns.heatmap(df.phik_matrix())

# Time series plots
# Time series plots are a little different, since we'll often be using the x-axis as sequential time. With pandas, as long as our timestamp is a timestamp datatype and our dataframe index, we can easily plot timeseries data.

time_df = pd.read_csv('data/temperature.csv', index_col='datetime', parse_dates=['datetime'], infer_datetime_format=True)
time_df

# we can see the index is of type "DatatimeIndex"
time_df.info()

# We could also get our data in a proper format using pd.to_datetime()
time_df2 = pd.read_csv('data/temperature.csv')
time_df2['datetime'] = pd.to_datetime(time_df2['datetime'])
time_df2.set_index('datetime', inplace=True)

time_df2
time_df['Denver'].plot()

# One last trick we'll learn with datetime data is we can *resample* it, meaning change the time increments. We can convert our data to monthly data like so. We need to provide a transformation for the data, like 'mean' to take the average.

time_df_months = time_df.resample('1M').mean()
time_df_months['Denver'].plot()

# If you want to remove missing values, dropna works
time_df.shape
time_df.dropna(inplace=True)
time_df.shape


# Saving a plot
# Saving a plot allows you to get higher resolution and control the size of the plot. In general, we want to first create the figure object with our specified size, then create our figure, then use plt.tight_layout, then save the figure. Remember `plt` is matplotlib which we imported earlier. Here is a respectable figure showing the temperature in Denver over the years. `dpi` is dots per inch. A higher value means higher resolution and bigger filesize. 300 can work well.

# Notice we also convert the units from Kelvin to Farenheight and add reasonable x- and y-labels so that the plot is easily understood. Making a good plot requires this effort.

time_df_months['Denver_F'] = 9 / 5 * (time_df_months['Denver'] - 273) + 32

f = plt.figure(figsize=(5.5, 5.5))
time_df_months['Denver_F'].plot()
plt.xlabel('Year')
plt.ylabel('Temperature (F)')
plt.tight_layout()  # auto-adjust margins
plt.savefig('denver_temps.jpg', dpi=300)

