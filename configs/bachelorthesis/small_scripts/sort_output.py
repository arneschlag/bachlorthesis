import pstats
import sys

# Load the profiling data from the binary file
stats = pstats.Stats('/home/scl9hi/bachelorthesis/bolt_mfsd/profile_eval.txt')  # Replace 'output.prof' with the actual cProfile output file

search_string = "/home/scl9hi/bachelorthesis/bolt_mfsd/"

# Sort the statistics by cumulative time
stats.sort_stats('time')

# Open the output file for writing
with open('output.txt', 'w') as f:
    # Redirect standard output to the file
    stats.stream = f

    # Print the statistics while filtering
    stats.print_stats()


with open('filtered_output.txt', "a") as w:
    with open('output.txt', "r") as f:
        for i, line in enumerate(f.readlines()):
            if i in [0,3] or search_string in line:
                w.write(line)
