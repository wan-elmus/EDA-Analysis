
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("prepped_churn_data.csv")

df.index = range(1, len(df) + 1)

df.insert(0, "customerID", df.index)

df.head(5)

payment_method_dummies = pd.get_dummies(df['PaymentMethod'])
contract_dummies = pd.get_dummies(df['Contract'])

# Combine the dummy variables with the original DataFrame
df = pd.concat([df, payment_method_dummies, contract_dummies], axis=1)
df.head()

categorical_columns = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)', 'Month-to-month', 'One year', 'Two year']

for column in categorical_columns:
    df[column] = pd.factorize(df[column])[0]

df.sample(5)

# Drop the original categorical columns and the customerID column
df.drop(['PaymentMethod', 'Contract', 'customerID'], axis=1, inplace=True)
df.head(5)


# Break the data into features and targets
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Fit a decision tree to the training data
dt_model = DecisionTreeClassifier(max_depth=3) 
dt_model.fit(X_train, y_train)

# Plot the decision tree
tree_rules = export_text(dt_model, feature_names=list(X.columns))
print(tree_rules)

# Hyperparameter tuning for decision tree
param_grid = {'max_depth': [3, 5, 7, 10]}  
dt_model = DecisionTreeClassifier()
grid_search = GridSearchCV(dt_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_max_depth = grid_search.best_params_['max_depth']

# Fit a decision tree with the best hyperparameters
dt_model_tuned = DecisionTreeClassifier(max_depth=best_max_depth)
dt_model_tuned.fit(X_train, y_train)

# Plot the decision tree
tree_rules_tuned = export_text(dt_model_tuned, feature_names=list(X.columns))
print(tree_rules_tuned)

# Plot correlations between features and targets
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr()[['Churn']], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlations between features and Churn")
plt.show()

# Fit a random forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Plot feature importances from the random forest
plt.figure(figsize=(12, 8))
sns.barplot(x=rf_model.feature_importances_, y=X.columns)
plt.title("Random Forest Feature Importances")
plt.show()

# Choose less-important features to remove
less_important_features = ['PhoneService', 'Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'One year'] 

X_train_filtered = X_train.drop(less_important_features, axis=1)
X_test_filtered = X_test.drop(less_important_features, axis=1)

X_train_filtered.shape

# Remove less-important features
X_train_filtered = X_train.drop(less_important_features, axis=1)
X_test_filtered = X_test.drop(less_important_features, axis=1)

# Fit random forest model to new data
rf_model_filtered = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_filtered.fit(X_train_filtered, y_train)

# Plot feature importances after removing less important features
plt.figure(figsize=(12, 8))
sns.barplot(x=rf_model_filtered.feature_importances_, y=X_train_filtered.columns)
plt.title("Random Forest Feature Importances after Removing Less Important Features")
plt.show()

# Evaluate model performance
def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Evaluate initial decision tree model
print("Evaluation of Decision Tree Model:")
evaluate_model(dt_model_tuned, X_test, y_test)

# Evaluate initial random forest model
print("\nEvaluation of Random Forest Model:")
evaluate_model(rf_model, X_test, y_test)

# Evaluate random forest model after removing less important features
print("\nEvaluation of Random Forest Model after Removing Less Important Features:")
evaluate_model(rf_model_filtered, X_test_filtered, y_test)


