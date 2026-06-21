#!/usr/bin/env python3
import rospy
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped

def odom_cb(msg):
    t = TransformStamped()
    t.header.stamp    = msg.header.stamp
    t.header.frame_id = msg.header.frame_id   # "odom"
    t.child_frame_id  = msg.child_frame_id    # "body"
    t.transform.translation.x = msg.pose.pose.position.x
    t.transform.translation.y = msg.pose.pose.position.y
    t.transform.translation.z = msg.pose.pose.position.z
    t.transform.rotation      = msg.pose.pose.orientation
    br.sendTransform(t)

if __name__ == "__main__":
    rospy.init_node("odom_to_tf")
    br = tf2_ros.TransformBroadcaster()
    rospy.Subscriber("/opt_odom", Odometry, odom_cb)
    rospy.spin()
