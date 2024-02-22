import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(filepath):
    """
    Load churn data into a DataFrame from a given filepath.
    """
    return pd.read_csv(filepath)

def make_predictions(df, model_name='LDA'):
    """
    Use the specified PyCaret model to make predictions on the provided DataFrame.
    """
    # Load the pre-trained PyCaret model
    model = load_model(model_name)
    
    # Make predictions on the DataFrame
    predictions = predict_model(model, data=df)
    
    # Rename and map the prediction labels
    predictions['Churn_prediction'] = predictions['prediction_label'].map({1: 'Churn', 0: 'No Churn'})
    
    return predictions['Churn_prediction']

if __name__ == "__main__":
    df = load_data('new_churn_data.csv')
    predictions = make_predictions(df, model_name='LDA')
    print('Predictions:')
    print(predictions)
