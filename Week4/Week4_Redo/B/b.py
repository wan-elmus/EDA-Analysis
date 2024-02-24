
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split, GridSearchCV
# from scikitplot.estimators import plot_feature_importances

# # Load data
# df = pd.read_csv("prepped_churn_data.csv")

# # Create features and targets
# features = df.drop('Churn', axis=1)
# targets = df['Churn']

# # Split the data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(features, targets, stratify=targets, random_state=42)

# # Decision Tree
# dt = DecisionTreeClassifier()
# dt.fit(x_train, y_train)

# print("Decision Tree:")
# print("Training accuracy:", dt.score(x_train, y_train))
# print("Testing accuracy:", dt.score(x_test, y_test))

# # Visualize decision tree
# plt.figure(figsize=(15, 15))
# plot_tree(dt, fontsize=8, feature_names=features.columns, filled=True)
# plt.show()

# # Random Forest
# rfc = RandomForestClassifier(random_state=42)

# # Define hyperparameters for tuning
# param_grid = {
#     'max_depth': [2, 5, 10],
#     'max_features': ['auto', 'sqrt', 'log2']
# }

# # Perform GridSearchCV for hyperparameter tuning
# grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
# grid_search.fit(x_train, y_train)

# # Get best parameters
# best_params = grid_search.best_params_
# print("Best parameters:", best_params)

# # Evaluate Random Forest with best parameters
# rfc_best = RandomForestClassifier(random_state=42, **best_params)
# rfc_best.fit(x_train, y_train)

# print("Random Forest:")
# print("Training accuracy:", rfc_best.score(x_train, y_train))
# print("Testing accuracy:", rfc_best.score(x_test, y_test))

# # Feature selection
# plt.figure(figsize=(10, 10))
# sns.heatmap(df.corr(), annot=True)
# plt.show()

# # Plot feature importances
# plt.figure(figsize=(10, 6))
# plot_feature_importances(rfc_best, feature_names=features.columns, x_tick_rotation=90)
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scikitplot.estimators import plot_feature_importances

df = pd.read_csv("new_churn_data.csv")
df.sample(6)

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

print("Decision Tree:")
print("Training accuracy:", dt.score(X_train, y_train))
print("Testing accuracy:", dt.score(X_test, y_test))

dt.get_depth()

param_grid = {'max_depth': [2, 3, 5, 7, 10]} 
dt_model = DecisionTreeClassifier()
grid_search = GridSearchCV(dt_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_max_depth = grid_search.best_params_['max_depth']
best_max_depth

dt_model_tuned = DecisionTreeClassifier(max_depth=best_max_depth)
dt_model_tuned.fit(X_train, y_train)

dt_model_tuned.get_depth()

print("Decision Tree:")
print("Training accuracy:", dt_model_tuned.score(X_train, y_train))
print("Testing accuracy:", dt_model_tuned.score(X_test, y_test))

f = plt.figure(figsize = (15,15))
_ = plot_tree(dt_model_tuned,fontsize=8,feature_names = X.columns, filled=True)

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr()[['Churn']], annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlations between features and Churn")
plt.show()

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train,y_train)

print(rf.score(X_train,y_train))
print(rf.score(X_test,y_test))

param_grid = {
    'max_depth': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best parameters:", best_params)

rf_best = RandomForestClassifier(random_state=42, **best_params)
rf_best.fit(X_train, y_train)

print("Random Forest:")
print("Training accuracy:", rf_best.score(X_train, y_train))
print("Testing accuracy:", rf_best.score(X_test, y_test))

plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True)
plt.show()

plt.figure(figsize=(10, 6))
plot_feature_importances(rf_best, feature_names=X.columns, x_tick_rotation=90)
plt.show()

less_important_features =['Electronic check', 'TotalCharges', 'TotalCharges_Tenure_Ratio','Two year','One year','Mailed check','Credit card (automatic)', 'Bank transfer (automatic)']



X_train_filtered = X_train.drop(less_important_features, axis=1)
X_test_filtered = X_test.drop(less_important_features, axis=1)

# Verify the shapes of X_train and y_train
print("Shape of X_train:", X_train_filtered.shape)
print("Shape of y_train:", y_train.shape)

# Fit the RandomForestClassifier
rf_model_filtered = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model_filtered.fit(X_train_filtered, y_train)

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
    
print("\nEvaluation of Random Forest Model after Removing Less Important Features:")
evaluate_model(rf_model_filtered, X_test_filtered, y_test)