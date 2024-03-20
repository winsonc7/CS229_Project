import pandas as pd
import pyarrow.parquet as pq

# Load the Parquet dataset
parquet_file_path = 'hf_data/test-00000-of-00001 (12).parquet'
parquet_table = pq.read_table(parquet_file_path)

# Convert Parquet table to pandas DataFrame
df = parquet_table.to_pandas()

# Convert DataFrame to JSON format
json_data = df.to_json(orient='records')

# Write JSON data to a file
json_file_path = 'hf_data/elem_math.json'
with open(json_file_path, 'w') as json_file:
    json_file.write(json_data)