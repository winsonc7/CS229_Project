import json

# List to store combined data
combined_data = []

# List of JSON files to combine
json_files = ['hf_data/college_bio.json', 'hf_data/college_chem.json', 'hf_data/college_cs.json', 'hf_data/college_math.json', 'hf_data/college_physics.json', 'hf_data/elem_math.json', 'hf_data/high_school_bio.json', 'hf_data/high_school_chem.json', 'hf_data/high_school_cs.json', 'hf_data/high_school_math.json', 'hf_data/high_school_physics.json', 'hf_data/high_school_stats.json']

# Loop through each JSON file
for file_name in json_files:
    with open(file_name, 'r') as file:
        data = json.load(file)
        combined_data.extend(data)

# Write combined data to a new JSON file
output_file = 'hf_data/all_data.json'
with open(output_file, 'w') as outfile:
    json.dump(combined_data, outfile, indent=4)