import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv('train.csv')

# Define categorical columns for label encoding
categorical_columns = ['brand', 'model', 'fuel_type', 'transmission', 'engine', 'ext_col', 'int_col']

# Initialize the label encoder
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Convert binary columns manually (e.g., 'Yes'/'No' or similar binary data)
binary_columns = ['accident', 'clean_title']
data['accident'] = data['accident'].map({'None reported': 0, 'At least 1 accident or damage reported': 1})
data['clean_title'] = data['clean_title'].map({'Yes': 1, 'No': 0})

# Convert any remaining object columns to numeric values
object_columns = data.select_dtypes(include=['object']).columns
if len(object_columns) > 0:
    print("Non-numeric columns found, attempting to convert:", object_columns)
    data[object_columns] = data[object_columns].apply(lambda x: pd.to_numeric(x, errors='coerce'))

# Ensure all columns are numeric
print(data.dtypes)

# Debug: Print the price column before converting it to a tensor
print("Price column (before conversion):")
print(data['price'].head())

# Check for missing values in the price column
print("Missing values in price column:")
print(data['price'].isna().sum())

# Ensure 'price' is numeric
data['price'] = pd.to_numeric(data['price'], errors='coerce')

# Separate features and target
arraydata = data.iloc[:, :-1]  # Assuming the last column is the target
arraytarget = data['price']  # Explicitly use the 'price' column

# Debug: Print the target column after extraction to ensure it's valid
print("Target column (price):")
print(arraytarget.head())

# Convert to tensor
datatensor = torch.tensor(arraydata.values, dtype=torch.float32)
targettensor = torch.tensor(arraytarget.values, dtype=torch.float32)

print("Feature Tensor:")
print(datatensor)
print("Feature Tensor Size:", datatensor.size())  # Or datatensor.shape


print("Target Tensor:")
print(targettensor)
print("Target Tensor Size:", targettensor.size())  # Or targettensor.shape