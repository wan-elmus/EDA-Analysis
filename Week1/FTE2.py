
import pandas as pd
import matplotlib.pyplot as plt

# load data
# we can give an index number or name for our index column, or leave it blank
df = pd.read_excel('data/diabetes_data.xlsx', index_col='Patient number')
df

