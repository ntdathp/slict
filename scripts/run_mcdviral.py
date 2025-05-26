#!/usr/bin/env python3

import os
import subprocess
import time
import signal
import sys

# Configuration
# Root directory containing MCDVIRAL datasets
DATASET_ROOT      = "/home/tmn/DATASETS/MCDVIRAL"
# Path to the ROS launch file to run for each bag
LAUNCH_FILE       = "/home/tmn/catkin_ws/src/slict/launch/run_mcdviral_uloc.launch"
# Delay (in seconds) between processing each folder
DELAY_BETWEEN     = 5
# Timeout (in seconds) waiting for the first /lastcloud message
LASTCLOUD_TIMEOUT = 20

# Global list to keep track of child processes for cleanup
child_procs = []

def cleanup_and_exit(signum, frame):
    """
    Signal handler: terminate all child processes and exit cleanly.
    Called when SIGINT (Ctrl-C) or SIGTERM is received.
    """
    print("\n[INFO] Signal received, terminating all child processes...")
    for proc in child_procs:
        try:
            # Kill the entire process group for each child
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass
    sys.exit(0)

# Register the cleanup handler for Ctrl-C and kill signals
signal.signal(signal.SIGINT,  cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)

def main():
    """
    Main processing loop:
    1. Verify ROS_MASTER_URI is set.
    2. Find all subfolders with 'day' or 'night' in their name.
    3. For each folder:
       a. Start rostopic echo for /liteloam_pose -> CSV.
       b. Launch the ROS launch file with the bag wildcard, tee logs.
       c. Wait up to LASTCLOUD_TIMEOUT for a /lastcloud message.
          - If none arrives, clean up and skip to next folder.
       d. Otherwise, wait for the launch to finish normally.
       e. Clean up the echo process and move to the next folder.
    """
    # Ensure that a ROS master is running
    if not os.environ.get("ROS_MASTER_URI"):
        print("ERROR: ROS_MASTER_URI is not set. Please start roscore first.")
        return

    # List and sort subfolders containing 'day' or 'night'
    subfolders = sorted([
        d for d in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, d))
           and ("day" in d or "night" in d)
    ])
    if not subfolders:
        print("No 'day' or 'night' subfolders found. Exiting.")
        return

    # Process each sequence folder
    for folder in subfolders:
        folder_path = os.path.join(DATASET_ROOT, folder)
        bag_pattern = os.path.join(folder_path, "*.bag")
        log_path    = os.path.join(folder_path, f"{folder}.log")
        csv_path    = os.path.join(folder_path, f"{folder}_liteloam_pose.csv")

        print(f"\n=== Processing folder: {folder} ===")
        print(f"Logs: {log_path} (also shown on terminal)")
        print(f"CSV output: {csv_path}")

        # Open CSV file for writing /liteloam_pose outputs
        with open(csv_path, 'w') as csv_f:
            # 1) Start rostopic echo for /liteloam_pose into CSV file
            echo_proc = subprocess.Popen(
                ["rostopic", "echo", "-p", "/liteloam_pose"],
                stdout=csv_f,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
            child_procs.append(echo_proc)

            # 2) Start the ROS launch, piping all output through 'tee' to both terminal and log file
            launch_cmd = (
                f"roslaunch {LAUNCH_FILE} "
                f"data_path:={DATASET_ROOT} bag_file:={bag_pattern} autorun:=1 "
                f"2>&1 | tee -a {log_path}"
            )
            launch_proc = subprocess.Popen(
                launch_cmd,
                shell=True,
                preexec_fn=os.setsid
            )
            child_procs.append(launch_proc)

            # 3) Monitor for the first /lastcloud message, with timeout
            monitor_proc = subprocess.Popen(
                ["rostopic", "echo", "-n1", "/lastcloud"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            child_procs.append(monitor_proc)

            try:
                # Wait up to LASTCLOUD_TIMEOUT seconds
                monitor_proc.wait(timeout=LASTCLOUD_TIMEOUT)
                print(f"[OK] /lastcloud message received within {LASTCLOUD_TIMEOUT}s.")
            except subprocess.TimeoutExpired:
                print(f"[WARN] No /lastcloud message in {LASTCLOUD_TIMEOUT}s. Aborting this run.")
                # Clean up all running child processes and skip to next folder
                cleanup_and_exit(None, None)

            # 4) If /lastcloud arrived, wait for roslaunch to complete normally
            launch_proc.wait()

            # After launch completes, terminate the echo process
            os.killpg(os.getpgid(echo_proc.pid), signal.SIGTERM)
            echo_proc.wait()

        print(f"Completed folder: {folder}. Check log and CSV for details.")

        # Clear the child process list and pause before next iteration
        child_procs.clear()
        time.sleep(DELAY_BETWEEN)

    print("\nAll folders processed. Exiting.")

if __name__ == '__main__':
    main()
