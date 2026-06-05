import subprocess
import sys
import time

start = time.time()
jobs = [
    ("robot_ratio.py", "robot_log.txt"),
    ("task_ratio.py", "task_log.txt")
]
for file, log_name in jobs:
    print(f"Running {file}")
    with open(log_name, "w") as log:
        result = subprocess.run(
            [sys.executable, file],
            stdout=log,
            stderr=log
        )
    if result.returncode != 0:
        print(f"{file} failed.")
        break
print("Finished all experiments.")
print("Total runtime:", round((time.time() - start)/3600, 2), "hours")
