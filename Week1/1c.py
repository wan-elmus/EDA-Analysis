
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('churn_data.csv')

print(df.head())
print(df.tail())

# 1. Business understanding

# Data preprocessing
# Drop 'customerID' as it is not useful for modeling
df = df.drop('customerID', axis=1)

# Convert binary categorical variable 'Churn' to numerical
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['PhoneService', 'Contract', 'PaymentMethod'], drop_first=True)

# Split the data into features (X) and target variable (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional, but can be beneficial for some models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a machine learning model (Random Forest Classifier in this example)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print('Classification Report:\n', classification_report_str)

# Plot confusion Matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# 2. Data understanding - EDA
# Histogram with the target ('Churn') as the 'hue' for a numeric column (e.g., 'MonthlyCharges')
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', multiple='stack', bins=30)
plt.xlabel('Monthly Charges')
plt.ylabel('Count')
plt.title('Churn Distribution Based on Monthly Charges')
plt.show()

# Bar plots for categorical columns with the target ('Churn') as the 'hue'
categorical_columns = ['PhoneService_Yes', 'Contract_One year', 'Contract_Two year','PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

# Convert boolean columns to strings
bool_columns = ['PhoneService_Yes', 'Contract_One year', 'Contract_Two year','PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

for column in bool_columns:
    df[column] = df[column].astype(str)

# Check unique values in categorical columns
for column in categorical_columns:
    unique_values = df[column].unique()
    print(f"Unique values in {column}: {unique_values}")

# Check data types of columns used in countplot
print(df[categorical_columns + ['Churn']].dtypes)

# Plot countplot for each categorical column
for column in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=column, hue='Churn', palette='Set1')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Churn Distribution Based on {column}')
    plt.show()


# Advanced EDA and Visualization
# Filtering
high_tenure = df[df['tenure'] > 50].copy()

# Combining filters using & (and) and | (or) operators
high_tenure_high_monthly_charges = df[(df['tenure'] > 50) & (df['MonthlyCharges'] > 70)].copy()

high_tenure_high_monthly_charges

df.info()

# Proportions of Churn for different groups
df.shape

churn_df = df[df['Churn'] == 1]
no_churn_df = df[df['Churn'] == 0]

churn_group_proportions = churn_df['PhoneService_Yes'].value_counts(normalize=True)
no_churn_group_proportions = no_churn_df['PhoneService_Yes'].value_counts(normalize=True)

# Display proportions
print("Churn Proportions by PhoneService:")
print(churn_group_proportions)

print("\nNo Churn Proportions by PhoneService:")
print(no_churn_group_proportions)

# Seaborn for plotting
# Scatter plot with color-coded churn status
sns.scatterplot(data=df, x='MonthlyCharges', y='TotalCharges', hue='Churn')

# Heatmap for correlation matrix
# Convert boolean columns to integers
bool_columns = ['PhoneService_Yes', 'Contract_One year', 'Contract_Two year','PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

for column in bool_columns:
    df[column] = df[column].map({'True': 1, 'False': 0}).astype(int)

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f")
plt.show()

