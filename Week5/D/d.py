
# import pandas as pd
# from pycaret.classification import setup, compare_models, predict_model, save_model, load_model
# import pickle
# from IPython.display import Code

# def read_data(file_path):
#     """Reads data from a CSV file into a pandas DataFrame."""
#     return pd.read_csv(file_path)

# def initialize_auto_ml_environment(data, target_variable):
#     """Initializes the auto ML environment using PyCaret's setup function."""
#     return setup(data, target=target_variable)

# def select_best_model():
#     """Compares different classification models and selects the best-performing one."""
#     return compare_models()

# def predict_churn(model, data):
#     """Predicts the target variable for the given data using the provided model."""
#     return predict_model(model, data)

# def save_best_model(model, file_name):
#     """Saves the best model to a file."""
#     save_model(model, file_name)

# def save_best_model_pickle(model, file_name):
#     """Saves the best model using pickle serialization."""
#     with open(file_name, 'wb') as f:
#         pickle.dump(model, f)

# def load_model_pickle(file_name):
#     """Loads the saved model using pickle deserialization."""
#     with open(file_name, 'rb') as f:
#         return pickle.load(f)

# def remove_target_variable(data):
#     """Creates new_data by copying the DataFrame and dropping the target variable column."""
#     return data.drop('Churn', axis=1).copy()

# def load_saved_model(file_name):
#     """Loads the saved model using PyCaret's load_model function."""
#     return load_model(file_name)

# def display_code(file_name):
#     """Creates a Python module named 'predict_churn.py' using IPython's Code display."""
#     return Code(file_name)

# import subprocess

# def run_script(script_name):
#     """Runs the specified Python script."""
#     subprocess.run(['python', script_name], check=True)

# # Main code execution
# if __name__ == "__main__":
#     # Read data
#     df = read_data("preped_churn_data.csv")
    
#     # Initialize auto ML environment
#     automl_setup = initialize_auto_ml_environment(df, target_variable='Churn')
    
#     # Compare models and select best
#     best_model = select_best_model()
#     print(best_model)
    
#     # Predict churn for specific row
#     selected_rows = df.iloc[-2:-12]
#     predicted_rows = predict_churn(best_model, selected_rows)
#     print(predicted_rows)
    
    
#     # Save best model
#     save_best_model(best_model, 'LR')
#     save_best_model_pickle(best_model, 'LR_model.pk')
    
#     # Load saved model
#     loaded_model = load_model_pickle('LR_model.pk')
    
#     # Create new data
#     new_data = remove_target_variable(selected_rows)
#     print(new_data)
    
#     # Predict using loaded model
#     loaded_model_prediction = loaded_model.predict(new_data)
#     print(loaded_model_prediction)
    
#     # Load saved model using PyCaret
#     loaded_lr = load_saved_model('LR')
#     loaded_lr_prediction = predict_churn(loaded_lr, new_data)
#     print(loaded_lr_prediction)
    
#     # Display code
#     code_display = display_code('predict_churn.py')
    
#     # Run script
#     run_script('predict_churn.py')


import pandas as pd
from pycaret.classification import setup, compare_models, predict_model, save_model, load_model
import pickle
from IPython.display import Code
import subprocess

# Read data from a CSV file into a pandas DataFrame
df = pd.read_csv("preped_churn_data.csv")

# Initialize auto ML environment with the target variable as 'Churn'
automl_setup = setup(df, target='Churn')

# Compare various classification models and select the best-performing one
best_model = compare_models()

# Extract the second-to-last row from the DataFrame
selected_rows = df.iloc[-2:-12]

# Use the best_model to predict the target variable for the selected rows
predicted_rows = predict_model(best_model, selected_rows)

# Save the best_model with the name 'LR'
save_model(best_model, 'LR')

# Serialize and save the best_model using pickle
with open('LR_model.pk', 'wb') as f:
    pickle.dump(best_model, f)

# Load the saved model using pickle deserialization
with open('LR_model.pk', 'rb') as f:
    loaded_model = pickle.load(f)

# Create new_data by copying the second-to-last rows and dropping the 'Churn' column
new_data = selected_rows.drop('Churn', axis=1).copy()

# Use the loaded_model to predict the target variable for new_data
loaded_model_prediction = loaded_model.predict(new_data)
print(loaded_model_prediction)

# Load the saved 'LR' model using PyCaret's load_model function
loaded_lr = load_model('LR')

# Use the loaded_lr model to predict the target variable for new_data
loaded_lr_prediction = predict_model(loaded_lr, new_data)
print(loaded_lr_prediction)

# Display code for creating 'predict_churn.py' module
code_display = Code('predict_churn.py')

# Execute the 'predict_churn.py' script
%run predict_churn.py








# # Drop rows with missing values from the original DataFrame before making predictions
# df_no_missing = df.dropna()

# # Use the loaded_model to predict the target variable for the new_data
# loaded_model_prediction = loaded_model.predict(df_no_missing.drop('Churn', axis=1))

# # Return the probability of churn for each new prediction
# probability_of_churn = loaded_model.predict_proba(df_no_missing.drop('Churn', axis=1))[:, 1]

# # Get the percentile where the prediction is in the distribution of probability predictions from the training dataset
# percentile_rank = (
#     (loaded_model.predict_proba(df_no_missing.drop('Churn', axis=1))[:, 1] <= probability_of_churn).mean() * 100
# )
# percentile_rank