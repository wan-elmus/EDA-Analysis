
import pandas as pd
from pycaret.classification import predict_model, load_model

df = pd.read_csv('new_churn_data.csv')

model = load_model('LR')

# Make predictions
predictions = predict_model(model, df)

# Rename the prediction label and replace values
predictions.rename({'prediction_label': 'Churn_prediction'}, axis=1, inplace=True)
predictions['Churn_prediction'].replace({1: 'Churn', 0: 'No Churn'}, inplace=True)

print(predictions['Churn_prediction'])
