import os
import csv
from itertools import groupby

# Function to process each input file
def process_file(input_file, combined_data):
    # Read the input file and group data
    data = {}
    with open(input_file, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            key = tuple(row[col] for col in ['dataset', 'mode', 'nb', 'nq', 'd', 'nlist', 'nprobe', 'k', 'orderOptimize', 'UnblockSend', 'cut', 'block', 'worker', 'group', 'team', 'ratio', 'teamRatio'])
            time_speedup = float(row['time_speedup'])
            recall = float(row['1-recall'])  # Read the recall value
            query_time = float(row['query_time'])
            original_time = float(row['original_time'])
            var = float(row['variance'])
            
            if key not in data:
                data[key] = {'time_speedups': [], 'recalls': [], 'query_time' : [], 'original_time' : [], 'var' : []}
            
            data[key]['time_speedups'].append(time_speedup)
            data[key]['recalls'].append(recall)
            data[key]['query_time'].append(query_time)
            data[key]['original_time'].append(original_time)
            data[key]['var'].append(var)
            

    # Calculate averages
    averages = []
    for key, values in data.items():
        avg_time_speedup = sum(values['time_speedups']) / len(values['time_speedups'])
        avg_recall = sum(values['recalls']) / len(values['recalls'])  # Calculate average recall
        avg_query_time = sum(values['query_time']) / len(values['query_time'])  # Calculate average recall
        avg_orignal_time = sum(values['original_time']) / len(values['original_time'])  # Calculate average recall
        avg_var = sum(values['var']) / len(values['var'])  # Calculate average recall
        
        # Append the calculated averages for each group
        row_data = list(key) + [avg_time_speedup, avg_recall, avg_query_time, avg_orignal_time, avg_var]
        averages.append(row_data)

    # Sort by average time_speedup in descending order
    averages.sort(key=lambda x: x[-5], reverse=True)  # Sort by average_time_speedup (second to last column)

 # Append a newline (empty row) between different dataset groups
    if averages:
        # Adding a blank row if the dataset has changed (based on 'dataset' column)
        if combined_data and combined_data[-1][0] != averages[0][0]:
            combined_data.append([])  # Empty row for newline
        
        # Append the sorted data to the combined_data list
        combined_data.extend(averages)
    # Append the sorted data to the combined_data list
    # combined_data.extend(averages)

# Initialize combined data storage
combined_data = []

# Traverse the directory and process each log.csv
root_dir = "/es01/home/lvxg/vdb/VectorDB/benchmarks"
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file == "log.csv":
            input_path = os.path.join(subdir, file)
            process_file(input_path, combined_data)
            # combined_data.extend('\n')

# Write combined data to a single output file
output_file = "combined_processed_log.csv"
with open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    header = ['dataset', 'mode', 'nb', 'nq', 'd', 'nlist', 'nprobe', 'k', 'orderOptimize', 'UnblockSend', 'cut', 'block', 'worker', 'group', 'team', 'ratio', 'teamRatio', 'average_time_speedup', 'average_recall', 'average_query_time', 'average_original_time', 'average_variance']
    writer.writerow(header)
    writer.writerows(combined_data)

print(f"Combined processed data saved to {output_file}")
