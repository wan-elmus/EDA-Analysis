
# # Import libraries and modules
# import pandas as pd
# import pickle
# from IPython.display import Code
# from pycaret.classification import setup, compare_models, predict_model, save_model, load_model

# # Load prepped churn data from CSV into a pandas DataFrame
# df = pd.read_csv("prepped_churn_data.csv")

# # Set up PyCaret's auto ML environment with the target variable as 'Churn'
# automl_setup = setup(df, target='Churn')

# # Determine the type of the automl_setup object
# automl_type = type(automl_setup)

# # Access the 8th element of the automl_setup object (may vary depending on PyCaret version)
# automl_element = automl_setup[8]

# # Compare various classification models and select the best-performing one
# best_model = compare_models()

# # Retrieve information about the best-performing model
# best_model_info = best_model

# # Extract the second-to-last row from the DataFrame
# selected_row = df.iloc[-2:-1]

# # Use the best_model to predict the target variable for the selected row
# predicted_row = predict_model(best_model, selected_row)

# # Save the best_model with the name 'LR'
# save_model(best_model, 'LR')

# # Serialize and save the best_model using pickle
# with open('LR_model.pk', 'wb') as f:
#     pickle.dump(best_model, f)

# # Load the saved model using pickle deserialization
# with open('LR_model.pk', 'rb') as f:
#     loaded_model = pickle.load(f)

# # Create new_data by copying the second-to-last row and dropping the 'Churn' column
# new_data = selected_row.copy()
# new_data.drop('Churn', axis=1, inplace=True)

# # Use the loaded_model to predict the target variable for new_data
# loaded_model_prediction = loaded_model.predict(new_data)

# # Load the saved 'LR' model using PyCaret's load_model function
# loaded_lr = load_model('LR')

# # Use the loaded_lr model to predict the target variable for new_data
# loaded_lr_prediction = predict_model(loaded_lr, new_data)

# # Display code for creating 'predict_churn.py' module
# code_display = Code('predict_churn.py')

# # Execute the 'predict_churn.py' script
# run_script = %run predict_churn.py

# # Additional comments or documentation can be added as needed throughout the code for clarity.

# Import libraries and modules
import pandas as pd
import pickle
from IPython.display import Code
from pycaret.classification import setup, compare_models, predict_model, save_model, load_model

# Load prepped churn data from CSV into a pandas DataFrame
df = pd.read_csv("prepped_churn_data.csv")

# Set up PyCaret's auto ML environment with the target variable as 'Churn'
automl_setup = setup(df, target='Churn')

# Determine the type of the automl_setup object
automl_type = type(automl_setup)

# Access the 8th element of the automl_setup object (may vary depending on PyCaret version)
automl_element = automl_setup.get_config("X_train")

# Compare various classification models and select the best-performing one
best_model = compare_models()

# Retrieve information about the best-performing model
best_model_info = best_model

# Extract the second-to-last row from the DataFrame
selected_row = df.iloc[-2:-1]

# Use the best_model to predict the target variable for the selected row
predicted_row = predict_model(best_model, selected_row)

# Save the best_model with the name 'LR'
save_model(best_model, 'LR')

# Serialize and save the best_model using pickle
with open('LR_model.pk', 'wb') as f:
    pickle.dump(best_model, f)

# Load the saved model using pickle deserialization
with open('LR_model.pk', 'rb') as f:
    loaded_model = pickle.load(f)

# Create new_data by copying the second-to-last row and dropping the 'Churn' column
new_data = selected_row.copy()
new_data.drop('Churn', axis=1, inplace=True)

# Use the loaded_model to predict the target variable for new_data
loaded_model_prediction = loaded_model.predict(new_data)

# Return the probability of churn for each new prediction
probability_of_churn = loaded_model.predict_proba(new_data)[:, 1]

# Get the percentile where the prediction is in the distribution of probability predictions from the training dataset
percentile_rank = (
    (loaded_model.predict_proba(df.drop('Churn', axis=1))[:, 1] <= probability_of_churn).mean() * 100
)

# Load the saved 'LR' model using PyCaret's load_model function
loaded_lr = load_model('LR')

# Use the loaded_lr model to predict the target variable for new_data
loaded_lr_prediction = predict_model(loaded_lr, new_data)
