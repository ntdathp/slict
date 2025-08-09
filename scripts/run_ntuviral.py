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
DATASET_ROOT    = "/home/dat/Downloads/data/ntuviral/"
ROSLAUNCH_FILE  = "slict run_ntuviral.launch"
TIMEOUT_SEC     = 10
TARGET_FOLDERS  = ["tnp_01", "tnp_02", "tnp_03"]  # Only run these folders
# =============================================================== #

last_msg_time = None  # will hold ROS time of last /opt_odom

def opt_odom_callback(msg):
    global last_msg_time
    last_msg_time = rospy.Time.now()

def monitor_opt_odom(proc):
    """Subscribe to /opt_odom and kill 'proc' if no msgs arrive for TIMEOUT_SEC."""
    global last_msg_time
    rospy.Subscriber("/opt_odom", Odometry, opt_odom_callback)
    rate = rospy.Rate(1)
    timeout = rospy.Duration(TIMEOUT_SEC)

    while not rospy.is_shutdown():
        now = rospy.Time.now()
        if last_msg_time is None:
            rospy.loginfo(" Waiting for first /opt_odom message...")
        elif now - last_msg_time > timeout:
            rospy.logwarn(f"No /opt_odom for {TIMEOUT_SEC}s. Terminating launch.")
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except ProcessLookupError:
                    rospy.logwarn(" Process already exited.")
            break
        rate.sleep()

def run_launch_for_bag(bag_path):
    """Launch slict on the given bag, monitor /opt_odom, then wait for termination."""
    global last_msg_time
    last_msg_time = None

    print(f"\n>>> Running slict with bag: {bag_path}")
    proc = subprocess.Popen(
        ["roslaunch", *ROSLAUNCH_FILE.split(),
         f"bag_file:={bag_path}",
         "odom_bag_topic:=/opt_odom",
         "autorun:=1"],
        preexec_fn=os.setsid
    )

    # give roslaunch & rosbag play time to start
    time.sleep(5)

    monitor_thread = threading.Thread(target=monitor_opt_odom, args=(proc,))
    monitor_thread.start()

    proc.wait()
    monitor_thread.join()
    print(" Launch finished.\n")

def main():
    if not os.environ.get("ROS_MASTER_URI"):
        print("ROS_MASTER_URI not set. Please start roscore first.")
        return

    for folder in TARGET_FOLDERS:
        full_folder = os.path.join(DATASET_ROOT, folder)
        # find all .bag but skip any *_opt_odom.bag already generated
        bag_files = sorted(glob.glob(os.path.join(full_folder, "*.bag")))
        bag_files = [b for b in bag_files if not b.endswith("_opt_odom.bag")]

        for bag_path in bag_files:
            run_launch_for_bag(bag_path)
            time.sleep(5)

if __name__ == "__main__":
    try:
        rospy.init_node("opt_odom_monitor", anonymous=True)
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
