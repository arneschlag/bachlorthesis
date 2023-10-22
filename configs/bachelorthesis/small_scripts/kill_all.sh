#!/bin/bash

# Define the user for which you want to kill jobs
user="scl9hi"

# Get a list of running job IDs for the user
job_ids=$(bjobs -u $user | awk 'NR > 1 {print $1}')

# Loop through each job ID and kill it
for job_id in $job_ids; do
    bkill $job_id
done
