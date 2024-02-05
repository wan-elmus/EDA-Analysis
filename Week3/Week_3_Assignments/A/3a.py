
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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

# Convert categorical columns to numeric
for column in categorical_columns:
    df[column] = pd.factorize(df[column])[0]

df.sample(5)

# Drop the original categorical columns
df.drop(['PaymentMethod', 'Contract'], axis=1, inplace=True)
df.head(5)

df.info()

df.isna().sum()

features = df.drop('Churn', axis=1)
targets = df['Churn']

features.sample()

targets.head()

X = features
y = targets

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

lr_model = LogisticRegression(max_iter=5000)
lr_model.fit(X_train, y_train)

lr_model.fit(X_train, y_train)

df['Churn'].value_counts(normalize=True)

print(lr_model.score(X_train, y_train))
print(lr_model.score(X_test, y_test))

# Predict test dataset

predictions = lr_model.predict(X_test)
predictions

# construct the confusion matix
cm = confusion_matrix(y_test, predictions, labels=lr_model.classes_)

# format and display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

lr_model.predict_proba(X_test)

lr_model.predict(X_test)[:15]

(lr_model.predict_proba(X_test)[:10, 1] > 0.5).astype('int')

predictions_lower_thresh = (lr_model.predict_proba(X_test)[:, 1] > 0.2).astype('int')
predictions_lower_thresh

print(accuracy_score(y_test, predictions_lower_thresh))

tn, fp, fn, tp  = confusion_matrix(y_test, predictions_lower_thresh).flatten()
print(tp / (tp + fn))

lr_model.coef_

coef_df = pd.DataFrame(data=lr_model.coef_, columns=features.columns)

coef_df.T.sort_values(by=0).plot.bar(legend=False)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


rf_model= RandomForestClassifier(max_depth=5, n_jobs=-1, random_state=42)
gb_model = GradientBoostingClassifier(max_depth=4, random_state=42)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

print(rf_model.score(X_train, y_train))
print(rf_model.score(X_test, y_test))

print(gb_model.score(X_train, y_train))
print(gb_model.score(X_test, y_test))

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_rf, labels=rf_model.classes_)

# format and display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

cm = confusion_matrix(y_test, y_pred_gb, labels=gb_model.classes_)

# format and display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gb_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Define classifiers and hyperparameters
classifiers = {
    'Random Forest': (RandomForestClassifier(), {'max_depth': [3, 5, 10, None], 'n_estimators': [10, 100, 200], 'max_features': [1, 3, 5, 7], 'min_samples_leaf': [1, 2, 3], 'min_samples_split': [2, 3, 4]}),
    'Logistic Regression': (LogisticRegression(), {'max_iter': [2000, 4000, 6000]}),
    'Gradient Boosting': (GradientBoostingClassifier(), {'max_depth': [3, 5, 10, None], 'n_estimators': [10, 100, 200], 'max_features': [1, 3, 5, 7], 'min_samples_leaf': [1, 2, 3], 'min_samples_split': [2, 3, 4]})
}

# Perform grid search for each classifier
for name, (classifier, param_grid) in classifiers.items():
    grid = GridSearchCV(classifier, param_grid=param_grid, cv=3, scoring='accuracy')
    model_grid = grid.fit(X_train, y_train)
    print(f'Best hyperparameters for {name} are: {model_grid.best_params_}')
    print(f'Best score for {name} is: {model_grid.best_score_}')
    
model_grid.best_estimator_

print(classification_report(y_test, model_grid.predict(X_test)))

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(model_grid, X_test, y_test)
plt.show()

RocCurveDisplay.from_estimator(lr_model, X_test, y_test)
plt.show()

probabilities = lr_model.predict_proba(X_test)[:, 1]

prob_df = pd.DataFrame(data={'predicted_probability': probabilities, 'target': y_test})

sns.histplot(data=prob_df, x='predicted_probability', hue='target', stat='density', common_norm=False)

index = prob_df[(prob_df['target'] == 1) & (prob_df['predicted_probability'] < 0.5)].index
prob_df.loc[index]

X_test.loc[index]
