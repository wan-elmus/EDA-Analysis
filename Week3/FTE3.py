
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import plot_confusion_matrix
# packages necessary for plotting conmatrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix


# we can give an index number or name for our index column, or leave it blank
df = pd.read_csv('../data/prepped_diabetes_data.csv', index_col='Patient number')
df

df.head(10)
df.tail(10)
df.sample(5)

df.info()

# Modeling

# We can now do some modeling and analysis of the data. We will use a simple binary classifier, which predicts a probability of a 1 as the outcome for each datapoint. The model we'll start with is logistic regression.

# With our problem, we are doing classification, where we predict the class - a 0 (no diabetes) or 1 (diabetes). Classification is a type of supervised learning. For this, we have features (inputs) and targets (outputs), and we train a model (fit the model) with data. We call this data our "training data". From the training data, our algorithm learn patterns in the data and we can make predictions about the data.

# First, let's break up our data into features and targets:

features = df.drop('Diabetes', axis=1)
targets = df['Diabetes']

features.head()
targets.head()

# Next, we split our data into train and test sets. We will use the training data to fit our model, and evaluate performance on both the train and test sets. It's important to evaluate the model on unseen data (our test data), because we can overfit to our training data. Overfitting happens when our model is too complex and fits to noise in the data. This results in a high score on the training data but poor performance on the test data. Underfitting happens when our model is not complex enough, and results in poor performance on both the training and testing data. We can also use cross-validation to break up our data into several versions of train and test sets, but simply using a train and test set is the foundation for this. The sklearn library makes it easy to do this:

x_train, x_test, y_train, y_test = train_test_split(features, targets, random_state=42)

# We can also change the size of train and test sets with either the `train_size` or `test_size`. The default test size is 0.25 or 25%.

x_train.shape

x_test.shape

y_train.shape

y_test.shape

x_train, x_test, y_train, y_test = train_test_split(features, targets, stratify=targets, random_state=42, test_size=.25)

len(x_train)

len(x_test)

# We can now fit our model to the training data. All sklearn models share a similar pattern: we create the model, then fit it to data. Once it's been fit, we can use methods like predict and predict_proba to predict values and probabilities of values.

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(x_train, y_train)

lr_model.fit(x_train, y_train)

# Evaluation

# Usually we would try several models and choose the best one based on metrics. The `score` method of classifiers in sklearn has accuracy as it's default method. We simply give the `score` method our features and targets, and it computes accuracy. We can see our accuracy (0.908 or 90.8% on the test set) is better than the "no information rate", or simply predicting that all datapoints are the "majority class" (0). That  would give us 0.846, or around 85%, accuracy.

# our "no information" rate is 84.6%, or the majority class fraction
df['Diabetes'].value_counts(normalize=True)

print(lr_model.score(x_train, y_train))
print(lr_model.score(x_test, y_test))

# We can also see our test accuracy is slightly lower than our training score. If the test score is very much lower than our training score, it's a sign of overfitting. In this case, our difference isn't too bad although it could be slightly better.

# The score we want to use to evaluate the performance of the model is the test score.

# Another useful evaluation tool, especially for binary classification, is the confusion matrix, which we can plot with sklearn easily:

# This was deprecated in sklearn 
#confusion_matrix( x_test, y_test)#, colorbar=False) # this argument only works with sklearn 0.24 and up


# Plotting confusion matrix
#gather the predictions for our test dataset
predictions = lr_model.predict(x_test)

# construct the confusion matix - this retrns an array
cm = confusion_matrix(y_test, predictions, labels=lr_model.classes_)

# format and display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lr_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# The confusion matrix shows true negatives (TN, or a prediction of 0 when the true label is 0), false negatives (FN, prediction=0 true=1), true positives (TP, prediction=true=1) and false positives (FP, prediction=1 true=0). From this, we can get an idea of how the algorithm is performing and compare multiple models. For example, here, we might care a lot about false negatives, since we would be missing people who may get diabetes and they may not be properly  treated. A false positive may cause someone to take precautionary measures, but may not be so bad. We can also tune our algorithm to reduce false negatives, which is covered in the advanced section.

# Another aspect of evaluation in CRISP-DM is checking our model against our original business or organizational objectives. For example, in step 1 of CRISP-DM, we may have set an objective of correctly predicting 90% of positive diabetes cases. We could then use the true positive rate (TPR = TP / (TP + FN) , also known as recall) to evaluate performance. In this case, the TPR is 9/15 or 60%, and not high enough. However, with some tuning of the model, we can achieve this. Most sklearn models have a `predict_proba()` method which predicts probabilities for each class:

lr_model.predict_proba(x_test)[:15]

# This gives us the probability for 0 (first column) and 1 (second column). By default, the `predict()` method of models (used in the confusion matrix function) uses a threshold of 0.5:

lr_model.predict(x_test)[:15]

(lr_model.predict_proba(x_test)[:5, 1] > 0.5).astype('int')

# However, if we lower our threshold, we can get fewer false negatives but more false positives. We can use a threshold of 0.2, so any prediction probability of 0.2 or above is rounded up to 1:

predictions_lower_thresh = (lr_model.predict_proba(x_test)[:, 1] > 0.13).astype('int')
predictions_lower_thresh

# We can check the accuracy and true positive rate (recall) with the new predictions:

print(accuracy_score(y_test, predictions_lower_thresh))
tn, fp, fn, tp  = confusion_matrix(y_test, predictions_lower_thresh).flatten()
print(tp / (tp + fn))

# We can see that with a lower threshold of 0.13, we can achive over 90% TPR, although accuracy has dropped to 82% (below the no information rate). However, if the TPR is more important than the accuracy, then we could use this model.

# Last, we can look at the coefficients from the model. In general, larger coefficients mean a feature is more strongly related to the target, but we should scale the features to get an accurate read on this.

lr_model.coef_

features.columns

coef_df = pd.DataFrame(data=lr_model.coef_, columns=features.columns)

coef_df.T.sort_values(by=0).plot.bar(legend=False)

10**0.3

10**-.15

# We can see the unscaled data has the height and gender as the most important features. With gender, we could say between female (0) and male (1) the log odds of having diabetes increases by 0.3. Or, taking `10**0.3`, we can see that the odds of diabetes increases by a factor of 2 between men and women in this dataset. We can also see that age, glucose, weight, and other factors are positively correlated with the occurance of diabetes, which makes sense.

# Advanced section
# Here, we will cover:

# - other ML models in sklearn
# - tuning hyperparameters
# - other evaluation metrics (ROC/AUC, classification report, precision/recall/F1 score)
# - using probabilities of predictions

# Other ML Models

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

rf_model = RandomForestClassifier(max_depth=5, n_jobs=-1, random_state=42)
gb_model = GradientBoostingClassifier(max_depth=4, random_state=42)

rf_model.fit(x_train, y_train)
gb_model.fit(x_train, y_train)

print(rf_model.score(x_train, y_train))
print(rf_model.score(x_test, y_test))

print(gb_model.score(x_train, y_train))
print(gb_model.score(x_test, y_test))

# plot_confusion_matrix(rf_model, x_test, y_test, cmap='Blues')


#gather the predictions for our test dataset
predictions = rf_model.predict(x_test)

# construct the confusion matix - this retrns an array
cm = confusion_matrix(y_test, predictions, labels=rf_model.classes_)

# format and display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#plot_confusion_matrix(gb_model, x_test, y_test, cmap='Blues')

#gather the predictions for our test dataset
predictions = gb_model.predict(x_test)

# construct the confusion matix - this retrns an array
cm = confusion_matrix(y_test, predictions, labels=gb_model.classes_)

# format and display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gb_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# We can see the models are overfitting here, since the training accuracy is much higher than the test accuracy. This means we are fitting to the noise in the data. It makes sense with the outlier detection from last week, since from the IQR method it looked like a lot of samples were outliers. These ensemble tree-based methods are prone to overfitting like this. With any model, we can optimize the hyperparameters to minimize or remove overfitting and optimize performance.

# OPtimizing hyperparameters

# Hyperparameters are the settings for the ML algorithms, like max_depth above, which sets the max depth of the trees used in the classifiers. There are many ways to tune them - we could change values by hand and compare scores. Or, sklearn has some methods that can be used. A few of those methods are random and grid search. We provide a range or list of hyperparmeters to try, and it either randomly tries different combinations and picks the best result (highest accuracy or other metric) or it methodically searches through every combination with grid search.

# However, one of the best methods is to use is Bayesian optimization, from scikit-optimize (skopt). First, we need to install the package with `conda install -c conda-forge scikit-optimize`. Currently (4-2021) sckit-opt has an issue with the latest sklearn, and we need to downgrade sklearn to version 0.23 or follow another solution from [here](https://github.com/scikit-optimize/scikit-optimize/issues/978). We can install sklearn 0.23 with `conda install -c conda-forge scikit-learn=0.23`. Then we can use the hyperparameter search:

# from skopt import BayesSearchCV

# opt = BayesSearchCV(
#     RandomForestClassifier(),
#     {
#         'max_depth': (3, 20),
#         'n_estimators': (50, 500),
#         'max_features': (3, 14),
#         'min_samples_split': (2, 5)
#     },
#     n_iter=32,
#     cv=3
# )

# opt.fit(x_train, y_train)

# print("val. score: %s" % opt.best_score_)
# print("test score: %s" % opt.score(x_test, y_test))

# opt.best_estimator_

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

grid_space={'max_depth':[3,5,10,None],
              'n_estimators':[10,100,200],
              'max_features':[1,3,5,7],
              'min_samples_leaf':[1,2,3],
              'min_samples_split':[1,2,3]
           }


grid = GridSearchCV(rf,param_grid=grid_space,cv=3,scoring='accuracy')
model_grid = grid.fit(x_train, y_train)

print('Best hyperparameters are: '+str(model_grid.best_params_))
print('Best score is: '+str(model_grid.best_score_))

#saving the best performing model
model_grid.best_estimator_

# This takes a while to run, since it is trying several combinations of hyperparameters. Essentially, it is checking the cross-validation accuracy scores and then trying the next combination of hyperparameters that seems like it should improve performance the most.

# We can search the Bayesian optimization increased our accuracy of the model but a good amount, and is slightly better than the 90.8% accuracy from our linear regression.

# Different models have different hyperparameters, and learning which ones are important and what values they often take is part of learning ML and building up the expertise.  For random forests, we used some of the most important hyperparamaters, but not all hyperparameters. The max_depth argument is how deep the decision trees can be, while n_estimators is the number of trees. max_features is the number of features it randomly selects from for each tree, and min_samples_split is how many samples need to be in a leaf of the tree to split it. We'll learn more about decision trees next week.

# For logistic regression, we can optimize the regularization with the C, penalty, and l1_ratio hyperparameters.

# Other Evaluation metrics

# Now that we have an optimized model, let's look at some other evaluation metrics to score it with We already saw accuracy, which is the percent of correct predictions out of the total number of samples. We can also look at some more specific counts of correct values with precision and recall. Precision is the number of TP divided by the number of all predicted positives (TP + FP). Recall is the number of TP divided by the number of all real positives (TP + FN). In our case, we might care most about improving recall. We can easily access these metrics with sklearn:

from sklearn.metrics import classification_report

print(classification_report(y_test, model_grid.predict(x_test)))

# It shows the precision and recall for each class, as well as a macro and weighted average (micro). Macro is simply the average between the values for the classes, whiche the weighted average (micro) adds up the values for each individual class in the calculations. So macro is (precision_0 + precision_1) / 2 while micro is (TP_0 + TP_1) / (TP_0 + TP_1 + FP_0 + FP_1) for precision.

# We also see the F1 score, which is the harmonic mean between precision and recall 2 * (P * R) / (P + R). We also see support, which is the number of samples.

# The package `yellowbrick` also provides a way to plot this classification report, as well as some other evaluation plotting functions.

# Another nice metric for binary classification is the reciever operating characteristic (ROC) curve and AUC score:

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(model_grid, x_test, y_test)
plt.show()

# This shows the FPR vs TPR (FPR = FP / (FP + TN) and TPR = TP / (TP + FN)). Each point is calculated by taking a value for the rounding threshold (the value where we round up a predicted probability to 1). A perfect model would touch the upper left of the plot, meaning a 100% true positive rate and 0% false positive rate. We can also get the AUC, or area under the curve, which is the integral of the line. It is the area under the ROC curve. For a perfect model, this would be 1. A model that randomly guesses values would approximately be a diagonal line from 0, 0 to 1, 1.

# We can compare this AUC score to our other model, and we actually see the logistic regression model performs better. We are using the `drop_intermediate=False` argument to keep all the TPR and FPR values for all thresholds (all unique predicted probability values in the predictions) so we can use it in the next section.

RocCurveDisplay.from_estimator(lr_model, x_test, y_test)
plt.show()

# Using prediction probabilities

# It looks like our logistic regression model is best, so we'll use that. With most models in sklearn, there is a predict_proba method we can use, as we saw. We can get the probability for class 1 (diabetes) and create some plots:

probabilities = lr_model.predict_proba(x_test)[:, 1]

prob_df = pd.DataFrame(data={'predicted_probability': probabilities, 'target': y_test})

import seaborn as sns

sns.histplot(data=prob_df, x='predicted_probability', hue='target', stat='density', common_norm=False)

# It looks like our predictions are mostly good, but we do have several low-confidence predictions for the occurance of diabetes. We can use pandas filtering to examine these and see if there are commonalities among them.

index = prob_df[(prob_df['target'] == 1) & (prob_df['predicted_probability'] < 0.5)].index
prob_df.loc[index]

x_test.loc[index]

# We can see these patients have glucose values similar to non-diabetics, and similar heights. Since the model had height negatively correlated with diabetes, and these patients have taller than average height in the dataset, it could be partly why they are being misclassified (or at least not confidently correctly classified). We might try removing the height column from the data, since intuitively it shouldn't be correlated to diabetes (one would think). Then we could re-evaluate the model.


# One last thing we can do with the probabilities of our classifier is try and get an optimal threshold for rounding. Essentially, we get the unique sorted thresholds from our probabability predictions (sorted from greatest to least), add on a value of 1 to the beginning of the thresholds list or array, then get the threshold which has the maximum TPR-FPR. This is called Youden's J and is one method for optimizing the threshold. To get all TPR and FPR rates from the ROC curve, we need to use the function like so `roc = plot_roc_curve(lr_model, x_test, y_test, drop_intermediate=False)`. Then we can get roc.tpr and roc.fpr.

