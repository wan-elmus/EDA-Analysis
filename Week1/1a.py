
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('churn_data.csv')

# Data preprocessing
df = df.drop('customerID', axis=1)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, columns=['PhoneService', 'Contract', 'PaymentMethod'], drop_first=True)

df.head()
df.tail()


# Split the data into features (X) and target variable (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=75)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
# Impute missing values in X_train
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

model = LogisticRegression(random_state=75)
model.fit(X_train_imputed, y_train)

# Make predictions on the test set
X_test_imputed = imputer.transform(X_test)
y_pred = model.predict(X_test_imputed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

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

# EDA - Histogram for 'MonthlyCharges'
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='MonthlyCharges', hue='Churn', multiple='stack', bins=30)
plt.xlabel('Monthly Charges')
plt.ylabel('Count')
plt.title('Churn Distribution Based on Monthly Charges')
plt.show()

# EDA - Bar plots for categorical columns with the target ('Churn') as the 'hue'
categorical_columns = ['PhoneService_Yes', 'Contract_One year', 'Contract_Two year',
                        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

# Convert boolean columns to strings
bool_columns = ['PhoneService_Yes', 'Contract_One year', 'Contract_Two year',
                'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

for column in bool_columns:
    df[column] = df[column].astype(str)

# Check data types of columns used in countplot
print(df[categorical_columns + ['Churn']].dtypes)

# EDA - Bar plot for each categorical column
for column in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=column, hue='Churn', palette='Set1')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'Churn Distribution Based on {column}', fontsize=16, fontweight='bold', color='navy')
    plt.xticks(rotation=45)
    plt.legend(title='Churn', title_fontsize='12', loc='upper right')
    plt.show()

# EDA - Scatter plot with color-coded churn status
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='MonthlyCharges', y='TotalCharges', hue='Churn')
plt.xlabel('Monthly Charges')
plt.ylabel('Total Charges')
plt.title('Scatter Plot with Churn Status')
plt.show()

# EDA - Line graph for churn proportions over tenure
churn_proportions = df.groupby('tenure')['Churn'].mean()
plt.figure(figsize=(12, 6))
sns.lineplot(x=churn_proportions.index, y=churn_proportions.values)
plt.xlabel('Tenure')
plt.ylabel('Churn Proportion')
plt.title('Churn Proportions Over Tenure')
plt.show()

# Pairplot for numeric columns
numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
sns.pairplot(df[numeric_columns + ['Churn']], hue='Churn', palette='husl')
plt.suptitle('Pairplot of Numeric Columns with Churn Status', y=1.02, fontsize=16, fontweight='bold')
plt.show()
