
import pandas as pd
import h2o
from h2o.estimators import H2ORandomForestEstimator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read data
df = pd.read_csv("new_churn_data.csv")

df

df.drop(columns=['PhoneService'], inplace=True)
df.head()

payment_method_dummies = pd.get_dummies(df['PaymentMethod'])
contract_dummies = pd.get_dummies(df['Contract'])

df = pd.concat([df, payment_method_dummies, contract_dummies], axis=1)
df.head()

df.drop(columns=['PaymentMethod', 'Contract'], inplace=True)
df

dummy_columns = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)', 'Month-to-month', 'One year', 'Two year']

for column in dummy_columns:
    df[column] = pd.factorize(df[column])[0]

df.sample(5)

df.isna().sum()


# Start and connect to the H2O cluster
h2o.init()

# Convert pandas DataFrame to H2O DataFrame
hf = h2o.H2OFrame(df)

# Split data into training and test sets
train, test = hf.split_frame(ratios=[0.8], seed=42)

# Fit H2O Random Forest to the original data
rf_h2o = H2ORandomForestEstimator(seed=42)
rf_h2o.train(x=hf.columns[:-1], y="Churn", training_frame=train)

# Plot H2O Random Forest's feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x=rf_h2o.varimp(use_pandas=True)['scaled_importance'], 
            y=rf_h2o.varimp(use_pandas=True)['variable'])
plt.title("H2O Random Forest Feature Importances")
plt.show()

# Fit sklearn Random Forest to the original data
X = df.drop('Churn', axis=1)
y = df['Churn']
rf_sklearn = RandomForestClassifier(n_estimators=100, random_state=42)
rf_sklearn.fit(X, y)

# Feature selection using sklearn Random Forest
sfm = SelectFromModel(rf_sklearn, threshold=0.1)
sfm.fit(X, y)
selected_features = X.columns[sfm.get_support()]

# Print selected features
print("Selected features using sklearn Random Forest:", selected_features)

# Hyperparameter tuning for sklearn Random Forest
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10, 20]}

grid_search = GridSearchCV(rf_sklearn, param_grid, cv=5)
grid_search.fit(X, y)

# Best hyperparameters
best_params_sklearn = grid_search.best_params_

# Fit sklearn Random Forest with best hyperparameters
rf_sklearn_tuned = RandomForestClassifier(**best_params_sklearn)
rf_sklearn_tuned.fit(X, y)

# Evaluate sklearn Random Forest model
y_pred_sklearn = rf_sklearn_tuned.predict(X)
accuracy_sklearn = accuracy_score(y, y_pred_sklearn)
precision_sklearn = precision_score(y, y_pred_sklearn)
recall_sklearn = recall_score(y, y_pred_sklearn)
f1_sklearn = f1_score(y, y_pred_sklearn)

print("Evaluation of sklearn Random Forest Model:")
print(f"Accuracy: {accuracy_sklearn:.4f}, Precision: {precision_sklearn:.4f}, Recall: {recall_sklearn:.4f}, F1 Score: {f1_sklearn:.4f}")

# Hyperparameter tuning for H2O Random Forest
hyper_params = {'ntrees': [50, 100, 200],
                'max_depth': [None, 5, 10, 20]}

search_criteria = {'strategy': "Cartesian"}

grid = h2o.grid.H2OGridSearch(model=H2ORandomForestEstimator, grid_id='rf_grid', hyper_params=hyper_params, search_criteria=search_criteria)

grid.train(x=hf.columns[:-1], y="Churn", training_frame=train)

# Get best H2O Random Forest model
best_rf_h2o = grid.models[0]

# Evaluate H2O Random Forest model
y_pred_h2o = best_rf_h2o.predict(test)
y_pred_h2o = y_pred_h2o.as_data_frame()['predict'].values
accuracy_h2o = accuracy_score(test['Churn'].as_data_frame(), y_pred_h2o)
precision_h2o = precision_score(test['Churn'].as_data_frame(), y_pred_h2o)
recall_h2o = recall_score(test['Churn'].as_data_frame(), y_pred_h2o)
f1_h2o = f1_score(test['Churn'].as_data_frame(), y_pred_h2o)

print("Evaluation of H2O Random Forest Model:")
print(f"Accuracy: {accuracy_h2o:.4f}, Precision: {precision_h2o:.4f}, Recall: {recall_h2o:.4f}, F1 Score: {f1_h2o:.4f}")
