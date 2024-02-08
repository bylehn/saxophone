import matplotlib.pyplot as plt
import os
import re

# Initialize lists to hold cores and times
cores = [1, 2, 4, 8, 16, 32]
times = []

# Regular expression to find execution time in output files
time_regex = re.compile(r'Execution Time: (\d+\.\d+) seconds')

print(f"Current working directory: {os.getcwd()}")  # Confirm working directory
print("Files in the current directory:")  # List files for verification
for file in os.listdir('.'):
    print(file)

# Loop through each core count and read the corresponding output file
for core in cores:
    found_file = False
    # Adjust this pattern to match your output file naming scheme
    filename_pattern = f'python_test_{core}_cores_.*\.out'
    print(f"Looking for files with pattern: {filename_pattern}")  # Debug pattern being used
    for filename in os.listdir('.'):
        if re.match(filename_pattern, filename):
            found_file = True
            print(f"Reading file: {filename}")  # Debugging print
            with open(filename, 'r') as file:
                for line in file:
                    match = time_regex.search(line)
                    if match:
                        times.append(float(match.group(1)))
                        print(f"Found time: {match.group(1)}")  # Debugging print
                        break
    if not found_file:
        print(f"No output file found for {core} cores")

# Check if times list is still empty
if not times:
    print("No execution times were found. Check output files and regex.")
else:
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(cores, times, marker='o', linestyle='-', color='b')
    plt.title('Execution Time vs. Number of Cores')
    plt.xlabel('Number of Cores')
    plt.ylabel('Execution Time (seconds)')
    plt.grid(True)
    plt.savefig('execution_time_vs_cores.png')
    plt.show()

