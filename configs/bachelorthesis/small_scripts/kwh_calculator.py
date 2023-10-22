import re
from datetime import datetime

timestamp_format = "%a %b %d %H:%M:%S %Y"
def main(file_path):
    total_gpu_time = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_job_gpu_time = 0
    current_job_gpu_model = None
    job_env = {}

    total = 0
    total_hours = 0
    for i,  line in enumerate(lines):
        if line.startswith("Job <"):
            # New job
            if len(job_env.keys()) > 0:
                # compute kWh
                if "gpu" in job_env:
                    watt = {"hi-026l": 250,
                            "hi-027l": 250,
                            "hi-028l": 250,
                            "hi-029l": 250,
                            "hi-031l": 250,
                            "hi-032l": 250,
                            "hi-033l": 250,
                            "hi-034l": 350,
                            "hi-030l": 300,
                            "hi-035l": 300,
                            "hi-036l": 300,}
                    w = watt[job_env['gpu']]
                    seconds = (job_env['end'] - job_env['start']).total_seconds()
                    hours = seconds/3600
                    kWh = w/1000
                    total += hours*kWh
                    total_hours +=hours

            job_env = {}
        if "Dispatched" in line:
            # Start Time
            start_time = datetime.strptime(line.split(': ')[0]+" 2023", timestamp_format)
            job_env["start"] = start_time

        if "Completed " in line:
            # End Time
            end_time = datetime.strptime(line.split(': ')[0]+" 2023", timestamp_format)
            job_env["end"] = end_time

        if "GPU_ALLOCATION" in line:
            lines[i+2] # GPU info
            gpuclass = lines[i+2].split(' ')[1]
            job_env["gpu"] = gpuclass

    # Update total GPU time with the GPU time of the last job
    #total_gpu_time += current_job_gpu_time

    print(total, "kWh")
    print(total_hours, "hours")
    print(total*(0.2), "Euro")


if __name__ == "__main__":
    file_path = "/home/scl9hi/archives2.txt"  # Replace with the actual file path
    total = 0
    main(file_path)
