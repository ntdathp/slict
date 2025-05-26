#!/usr/bin/env python3
import rospy
import csv
from geometry_msgs.msg import PoseStamped
import atexit

# global writer and file handle
csvfile = None
writer = None

def pose_callback(msg):
    global writer
    t  = msg.header.stamp.to_sec()
    x  = msg.pose.position.x
    y  = msg.pose.position.y
    z  = msg.pose.position.z
    qx = msg.pose.orientation.x
    qy = msg.pose.orientation.y
    qz = msg.pose.orientation.z
    qw = msg.pose.orientation.w

    # echo to console
    rospy.loginfo(f"[{t:.3f}] x={x:.3f}, y={y:.3f}, z={z:.3f}")

    # write to CSV
    writer.writerow([t, x, y, z, qx, qy, qz, qw])
    csvfile.flush()

def cleanup():
    if csvfile:
        csvfile.close()

if __name__ == "__main__":
    rospy.init_node("liteloam_pose_logger", anonymous=True)

    # get output path from parameter or use default
    filename = rospy.get_param("~output_file", "liteloam_poses.csv")
    csvfile = open(filename, "w", newline="")
    writer = csv.writer(csvfile)
    # write CSV header
    writer.writerow(["time", "x", "y", "z", "qx", "qy", "qz", "qw"])
    csvfile.flush()

    # ensure file is closed on shutdown
    atexit.register(cleanup)
    rospy.on_shutdown(cleanup)

    # subscribe and spin
    rospy.Subscriber("/liteloam_pose", PoseStamped, pose_callback)
    rospy.loginfo(f"Logging /liteloam_pose → {filename}")
    rospy.spin()
