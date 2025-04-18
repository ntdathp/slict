#!/usr/bin/env python3

import os
import glob
import subprocess
import signal
import time
import threading
import rospy
from nav_msgs.msg import Odometry  # Using Odometry instead of PoseStamped

# ======================== Configuration ======================== #
DATASET_ROOT = "/home/dat/Downloads/data/ntuviral/"
ROSLAUNCH_FILE = "slict run_ntuviral.launch"
TIMEOUT_SEC = 20
TARGET_FOLDERS = ["spms_01", "sbs_01"]  # Only run these specific folders
# =============================================================== #

last_msg_time = None  # ROS time tracking

def opt_odom_callback(msg):
    global last_msg_time
    last_msg_time = rospy.Time.now()

def monitor_opt_odom(proc):
    global last_msg_time
    rospy.Subscriber("/opt_odom", Odometry, opt_odom_callback)  # Correct message type
    rate = rospy.Rate(1)
    timeout_duration = rospy.Duration(TIMEOUT_SEC)

    while not rospy.is_shutdown():
        now = rospy.Time.now()
        if last_msg_time is None:
            rospy.loginfo("Waiting for first /opt_odom message...")
        else:
            if now - last_msg_time > timeout_duration:
                rospy.logwarn("No /opt_odom received in {}s. Terminating launch.".format(TIMEOUT_SEC))
                if proc.poll() is None:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except ProcessLookupError:
                        rospy.logwarn("Process already exited before termination.")
                break
        rate.sleep()

def run_launch_for_bag(bag_path):
    global last_msg_time
    last_msg_time = None  # Reset time tracking before each run

    print(f"\n>>> Running slict with bag: {bag_path}")

    proc = subprocess.Popen(
        [
            "roslaunch", *ROSLAUNCH_FILE.split(),
            f"bag_file:={bag_path}",
            "odom_bag_topic:=/opt_odom",
            "autorun:=1"
        ],
        preexec_fn=os.setsid
    )

    # Wait a few seconds to ensure roslaunch and rosbag playback start
    time.sleep(5)

    monitor_thread = threading.Thread(target=monitor_opt_odom, args=(proc,))
    monitor_thread.start()

    proc.wait()
    monitor_thread.join()
    print("Launch finished.\n")

def main():
    if not os.environ.get("ROS_MASTER_URI"):
        print("ROS Master URI not set. Please make sure roscore is running.")
        return

    dataset_folders = [os.path.join(DATASET_ROOT, folder) for folder in TARGET_FOLDERS]

    for folder in dataset_folders:
        bag_files = sorted(glob.glob(os.path.join(folder, "*.bag")))
        for bag_path in bag_files:
            run_launch_for_bag(bag_path)
            time.sleep(5)

if __name__ == "__main__":
    try:
        rospy.init_node("opt_odom_monitor", anonymous=True)
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
