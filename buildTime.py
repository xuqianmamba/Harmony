import os
import csv

# Define the function to process each log.csv and extract the necessary columns
def process_file(input_path, combined_data):
    with open(input_path, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        # Skip the header row
        next(reader)
        
        # Process each row
        for row in reader:
            # Extract the relevant columns: dataset, mode, train_time, add_time, preSearch_time
            dataset = row[0]
            mode = row[1]
            train_time = row[-3]  # 3rd column from the end
            add_time = row[-2]    # 2nd column from the end
            preSearch_time = row[-1]  # last column
            
            # Combine into the desired format and append to the combined data list
            combined_data.append([dataset, mode, train_time, add_time, preSearch_time])

# Define the root directory to search for log.csv files
root_dir = "/es01/home/lvxg/vdb/VectorDB/benchmarks"

# Initialize an empty list to store the combined data
combined_data = []

# Traverse the directory and process each log.csv file
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file == "log.csv":
            input_path = os.path.join(subdir, file)
            process_file(input_path, combined_data)

# Write combined data to a single output file
output_file = "build_time.csv"
with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    # Write the header row
    header = ['dataset', 'mode', 'train_time', 'add_time', 'preSearch_time']
    writer.writerow(header)
    # Write the rows of combined data
    writer.writerows(combined_data)
