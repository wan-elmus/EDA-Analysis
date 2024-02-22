

import pandas as pd
from pycaret.classification import predict_model, load_model

def predict_churn():
    df = pd.read_csv('new_churn_data.csv')
    model = load_model('ridge')
    predictions = predict_model(model, df)
    predictions.rename({'prediction_label': 'Churn_prediction'}, axis=1, inplace=True)
    predictions['Churn_prediction'].replace({1: 'Churn', 0: 'No Churn'}, inplace=True)
    return predictions['Churn_prediction']

# Call the function and print the predictions
print(predict_churn())
