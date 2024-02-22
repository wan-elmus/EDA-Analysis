
# Import pandas library as pd
import pandas as pd

# Read the prepped churn data from the CSV file into a pandas DataFrame
df = pd.read_csv("prepped_churn_data.csv")

# Import necessary PyCaret functions and classes
from pycaret.classification import setup, compare_models, predict_model, save_model, load_model

# Use PyCaret's setup function to initialize the auto ML environment, specifying the target variable as 'Churn'
automl = setup(df, target='Churn')

# Check the type of the automl object
type(automl)

# Access the 8th element of the automl object (This can vary depending on PyCaret version)
automl[8]

# Compare different classification models and select the best-performing one
best_model = compare_models()

# Display information about the best-performing model
best_model

# Select a specific row (second-to-last row in this case) from the DataFrame
df.iloc[-2:-1]

# Use the best_model to predict the target variable for the selected row
predict_model(best_model, df.iloc[-2:-1])

# Save the best_model to a file named 'LR'
save_model(best_model, 'LR')

# Import the 'pickle' module to serialize and deserialize Python objects
import pickle

# Save the best_model using pickle serialization
with open('LR_model.pk', 'wb') as f:
    pickle.dump(best_model, f)

# Load the saved model using pickle deserialization
with open('LR_model.pk', 'rb') as f:
    loaded_model = pickle.load(f)

# Create new_data by copying the second-to-last row of the DataFrame and dropping the 'Churn' column
new_data = df.iloc[-2:-1].copy()
new_data.drop('Churn', axis=1, inplace=True)

# Use the loaded_model to predict the target variable for the new_data
loaded_model.predict(new_data)

# Load the saved 'LR' model using PyCaret's load_model function
loaded_lr = load_model('LR')

# Use the loaded_lr model to predict the target variable for the new_data
predict_model(loaded_lr, new_data)

# Create a Python module named 'predict_churn.py' using IPython's Code display
from IPython.display import Code
Code('predict_churn.py')

from IPython.display import Code
Code('predict_churn.py')

# Run the 'predict_churn.py' script
%run predict_churn.py


