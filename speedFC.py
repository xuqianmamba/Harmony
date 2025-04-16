import csv
import os
from itertools import groupby

# Function to process a single CSV file
def process_csv(input_file, data_dict):
    with open(input_file, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            key = (row['dataset'], row['mode'], row['worker'])  # Grouping by dataset and mode
            if key not in data_dict:
                data_dict[key] = []
            data_dict[key].append(row)

# Function to write the grouped and sorted data to an output file
def write_output(output_file, data_dict, fieldnames):
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()  # Write the header

        # Iterate over each group, sort by 'nprobe' and write the rows
        for (dataset, mode, worker), rows in data_dict.items():
            # Sort rows within the group by 'nprobe'
            rows_sorted = sorted(rows, key=lambda x: float(x['average_variance']))
            for row in rows_sorted:
                writer.writerow(row)
            outfile.write("\n")  # Separate groups by a newline

# Main function to process all CSV files in the directory and write the sorted output
def process_directory(root_dir, output_file):
    data_dict = {}  # Dictionary to store grouped data
    fieldnames = None  # Placeholder for fieldnames

    # Traverse through directory and process each 'log.csv'
    # for subdir, _, files in os.walk(root_dir):
    #     for file in files:
    #         if file == 'log.csv':
    input_path = "combined_processed_log.csv"
    if fieldnames is None:
        # Read the fieldnames from the first file
        with open(input_path, 'r') as infile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames
    process_csv(input_path, data_dict)

    # Write the final output to a file
    write_output(output_file, data_dict, fieldnames)

# Define the directory path and output file
root_dir = "/es01/home/lvxg/vdb/VectorDB/benchmarks"  # Modify if necessary
output_file = "sorted_combined_fc.csv"

# Process the directory and write the output
process_directory(root_dir, output_file)

print(f"Data has been grouped, sorted, and saved to {output_file}")
