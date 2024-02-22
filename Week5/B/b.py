
# Import pandas library as pd
import pandas as pd

# Import necessary PyCaret functions and classes
from pycaret.classification import (
    setup, compare_models, predict_model,
    save_model, load_model
)

# Import the 'pickle' module to serialize and deserialize Python objects
import pickle

# Import Code class from IPython.display
from IPython.display import Code

# Read the prepped churn data from the CSV file into a pandas DataFrame
df = pd.read_csv("prepped_churn_data.csv")

# Use PyCaret's setup function to initialize the auto ML environment, specifying the target variable as 'Churn'
automl_setup = setup(df, target='Churn')

# Check the type of the automl_setup object
automl_type = type(automl_setup)

# Access the 8th element of the automl_setup object (This can vary depending on PyCaret version)
automl_element = automl_setup[8]

# Compare different classification models and select the best-performing one
best_model = compare_models()

# Display information about the best-performing model
best_model_info = best_model

# Select a specific row (second-to-last row in this case) from the DataFrame
selected_row = df.iloc[-2:-1]

# Use the best_model to predict the target variable for the selected row
predicted_row = predict_model(best_model, selected_row)

# Save the best_model to a file named 'LR'
save_model(best_model, 'LR')

# Save the best_model using pickle serialization
with open('LR_model.pk', 'wb') as f:
    pickle.dump(best_model, f)

# Load the saved model using pickle deserialization
with open('LR_model.pk', 'rb') as f:
    loaded_model = pickle.load(f)

# Create new_data by copying the second-to-last row of the DataFrame and dropping the 'Churn' column
new_data = selected_row.copy()
new_data.drop('Churn', axis=1, inplace=True)

# Use the loaded_model to predict the target variable for the new_data
loaded_model_prediction = loaded_model.predict(new_data)

# Load the saved 'LR' model using PyCaret's load_model function
loaded_lr = load_model('LR')

# Use the loaded_lr model to predict the target variable for the new_data
loaded_lr_prediction = predict_model(loaded_lr, new_data)

# Create a Python module named 'predict_churn.py' using IPython's Code display
code_display = Code('predict_churn.py')

# Run the 'predict_churn.py' script
run_script = %run predict_churn.py
