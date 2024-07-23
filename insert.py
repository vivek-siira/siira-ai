import pandas as pd
import pyodbc
import logging
import math

# Set up logging
logging.basicConfig(level=logging.INFO)

file_path = 'C:/Users/Mohamed El-Mamon/Downloads/test_file.csv'
df = pd.read_csv(file_path)

# Define the table name
table_name = 'Galanovela'

# Function to generate SQL insert statements
def generate_insert_statements(df, table_name, batch_size=100):
    columns = ', '.join(df.columns)
    sql_statements = []
    
    for start_row in range(0, len(df), batch_size):
        batch = df.iloc[start_row:start_row + batch_size]
        values = []
        
        for _, row in batch.iterrows():
            row_values = ', '.join(f"'{str(val).replace('\'', '\'\'')}'" if pd.notna(val) else 'NULL' for val in row)
            values.append(f"({row_values})")
        
        values_string = ',\n'.join(values)
        sql = f"INSERT INTO {table_name} ({columns}) VALUES\n{values_string};"
        sql_statements.append(sql)
    
    return sql_statements

# Generate the insert statements
sql_statements = generate_insert_statements(df, table_name)

# Display the first few statements as a sample
sample_statements = sql_statements[:2]
sample_statements
