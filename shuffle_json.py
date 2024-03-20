import json
import random

# Read the JSON file
input_file = 'hf_data/all_data.json' 
with open(input_file, 'r') as file:
    data = json.load(file)

# Shuffle the entries
random.shuffle(data)

# Write shuffled data back to the JSON file
output_file = 'hf_data/all_data.json'
with open(output_file, 'w') as outfile:
    json.dump(data, outfile, indent=4)