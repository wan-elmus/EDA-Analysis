
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scikitplot.estimators import plot_feature_importances

# Load data
df = pd.read_csv("new_churn_data.csv")

# Create features and targets
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

print("Decision Tree:")
print("Training accuracy:", dt.score(X_train, y_train))
print("Testing accuracy:", dt.score(X_test, y_test))

# Bayesian Search for Decision Tree hyperparameter tuning
param_bayes = {'max_depth': (2, 10)}

dt_bayes = BayesSearchCV(dt, param_bayes, n_iter=50, cv=5, random_state=42)
dt_bayes.fit(X_train, y_train)

best_max_depth_dt = dt_bayes.best_params_['max_depth']

dt_tuned = DecisionTreeClassifier(max_depth=best_max_depth_dt)
dt_tuned.fit(X_train, y_train)

print("Tuned Decision Tree:")
print("Training accuracy:", dt_tuned.score(X_train, y_train))
print("Testing accuracy:", dt_tuned.score(X_test, y_test))

# Visualize tuned decision tree
plt.figure(figsize=(15, 15))
_ = plot_tree(dt_tuned, fontsize=8, feature_names=X.columns, filled=True)
plt.show()

# Random Forest
rf = RandomForestClassifier(random_state=42)

# Bayesian Search for Random Forest hyperparameter tuning
param_bayes_rf = {'max_depth': (2, 10)}

rf_bayes = BayesSearchCV(rf, param_bayes_rf, n_iter=50, cv=5, random_state=42)
rf_bayes.fit(X_train, y_train)

best_max_depth_rf = rf_bayes.best_params_['max_depth']

rf_tuned = RandomForestClassifier(random_state=42, max_depth=best_max_depth_rf)
rf_tuned.fit(X_train, y_train)

print("Tuned Random Forest:")
print("Training accuracy:", rf_tuned.score(X_train, y_train))
print("Testing accuracy:", rf_tuned.score(X_test, y_test))

# Feature selection
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr()[['Churn']], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlations between features and Churn")
plt.show()

# Plot feature importances for tuned Random Forest
plt.figure(figsize=(10, 6))
plot_feature_importances(rf_tuned, feature_names=X.columns, x_tick_rotation=90)
plt.show()

# Further feature selection and model evaluation
less_important_features = ['Electronic check', 'TotalCharges', 'TotalCharges_Tenure_Ratio', 'Two year', 'One year', 'Mailed check', 'Credit card (automatic)', 'Bank transfer (automatic)']

X_train_filtered = X_train.drop(less_important_features, axis=1)
X_test_filtered = X_test.drop(less_important_features, axis=1)

# Fit the RandomForestClassifier
rf_model_filtered = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_filtered.fit(X_train_filtered, y_train)

# Visualize feature importances after removing less important features
plt.figure(figsize=(12, 8))
sns.barplot(x=rf_model_filtered.feature_importances_, y=X_train_filtered.columns)
plt.title("Random Forest Feature Importances after Removing Less Important Features")
plt.show()

def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Evaluate the model after feature selection
print("\nEvaluation of Random Forest Model after Removing Less Important Features:")
evaluate_model(rf_model_filtered, X_test_filtered, y_test)
