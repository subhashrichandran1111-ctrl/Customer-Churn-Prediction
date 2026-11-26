
from pathlib import Path
import pandas as pd

# Get the main project folder
BASE_DIR = Path(__file__).resolve().parents[1]

# Locate the dataset
file_path = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Display first five records
print("\nFirst Five Records:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns.tolist())

# Display dataset information
print("\nDataset Information:")
df.info()

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display churn distribution
print("\nChurn Distribution:")
print(df["Churn"].value_counts())
