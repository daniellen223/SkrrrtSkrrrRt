import pandas as pd
import torch

# Load the data
data = pd.read_csv('train.csv')
print(data.head())
# One-hot encode categorical columns
categorical_columns = ['brand', 'model', 'fuel_type', 'transmission', 'engine', 'ext_col', 'int_col']
data_encoded = pd.get_dummies(data, columns=categorical_columns)

# Convert binary columns manually (e.g., 'Yes'/'No' or similar binary data)
binary_columns = ['accident', 'clean_title']
data_encoded['accident'] = data_encoded['accident'].map({'None reported': 0, 'At least 1 accident or damage reported': 1})
data_encoded['clean_title'] = data_encoded['clean_title'].map({'Yes': 1, 'No': 0})

# Convert boolean columns to integers
bool_columns = data_encoded.select_dtypes(include=['bool']).columns
data_encoded[bool_columns] = data_encoded[bool_columns].astype(int)

# Convert any remaining object columns to strings or handle them as needed
object_columns = data_encoded.select_dtypes(include=['object']).columns
if len(object_columns) > 0:
    print("Non-numeric columns found, attempting to convert:", object_columns)
    data_encoded[object_columns] = data_encoded[object_columns].apply(lambda x: pd.to_numeric(x, errors='coerce'))

# Ensure all columns are numeric before proceeding
print(data_encoded.dtypes)

# Convert columns to appropriate types (if necessary)
data_encoded['price'] = pd.to_numeric(data_encoded['price'], errors='coerce')

# Separate features and target
arraydata = data_encoded.iloc[:, :-1]  # Assuming the last column is the target
arraytarget = data_encoded['price']  # Explicitly use the 'price' column

# Ensure target is numeric
arraytarget = pd.to_numeric(arraytarget, errors='coerce')

# Convert to tensor
datatensor = torch.tensor(arraydata.values, dtype=torch.float32)
targettensor = torch.tensor(arraytarget.values, dtype=torch.float32)

print(datatensor)
print(targettensor)
