#!/usr/bin/env python3

import os
import glob
import subprocess
import time
import rospy
import signal

# ======================== Configuration ======================== #
# Root directory containing MCDVIRAL sequences (ntu_day_XX, ntu_night_XX, etc.)
DATASET_ROOT = "/home/dat/Downloads/data/MCDVIRAL"
# Full path to your ROS launch file
LAUNCH_FILE = os.path.expanduser("/home/dat/slict_ws/src/slict/launch/run_mcdviral_uloc.launch")
# Delay (in seconds) between consecutive bag runs
DELAY_BETWEEN = 5
# =============================================================== #

def main():
    # Ensure ROS master is running
    if not os.environ.get("ROS_MASTER_URI"):
        print("ROS_MASTER_URI is not set. Please start roscore first.")
        return

    # Find all subfolders whose names contain "day" or "night"
    subfolders = sorted([
        name for name in os.listdir(DATASET_ROOT)
        if os.path.isdir(os.path.join(DATASET_ROOT, name))
        and ("day" in name or "night" in name)
    ])

    if not subfolders:
        print("No 'day' or 'night' subfolders found. Exiting.")
        return

    for folder in subfolders:
        folder_path = os.path.join(DATASET_ROOT, folder)
        # Collect all .bag files in this folder
        bag_files = sorted(glob.glob(os.path.join(folder_path, "*.bag")))

        if not bag_files:
            print(f"No .bag files in {folder}, skipping.")
            continue

        print(f"\n=== Processing folder: {folder} ===")
        for bag in bag_files:
            print(f"--> Launching bag: {bag}")
            # Call roslaunch, passing data_path and bag_file parameters
            proc = subprocess.Popen(
                [
                    "roslaunch", LAUNCH_FILE,
                    f"data_path:={DATASET_ROOT}",
                    f"bag_file:={bag}",
                    "autorun:=1"
                ],
                preexec_fn=os.setsid
            )
            # Wait until the launch process exits
            proc.wait()
            print("    Launch completed.")
            # Pause before starting the next bag
            time.sleep(DELAY_BETWEEN)

if __name__ == "__main__":
    try:
        # Initialize ROS node (anonymous name so multiple instances won't clash)
        rospy.init_node("batch_mcdviral_runner", anonymous=True)
        main()
    except KeyboardInterrupt:
        print("Execution interrupted by user.")
