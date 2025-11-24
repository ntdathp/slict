#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import PoseStamped
import csv
import os


class RelocPoseCSVLogger(object):
    def __init__(self):
        # ===== Params =====
        # Thư mục output
        output_dir = rospy.get_param("~output_dir", ".")  # mặc định là thư mục hiện tại
        # Tên file csv
        csv_name = rospy.get_param("~csv_name", "reloc_pose_log.csv")

        # Tạo thư mục nếu chưa có
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Ghép thành đường dẫn đầy đủ
        self.output_path = os.path.join(output_dir, csv_name)

        # ===== Open CSV file =====
        self.csv_file = open(self.output_path, mode='w')
        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            "index", "timestamp",
            "x", "y", "z",
            "qx", "qy", "qz", "qw"
        ])
        self.csv_file.flush()

        self.index = 0

        self.sub = rospy.Subscriber(
            "/reloc_pose",
            PoseStamped,
            self.callback,
            queue_size=100
        )

        rospy.loginfo("[RelocPoseCSVLogger] Logging /reloc_pose to %s", self.output_path)

        rospy.on_shutdown(self.on_shutdown)


    def callback(self, msg: PoseStamped):
        # Convert ROS time -> float with nanosecond precision
        stamp = msg.header.stamp
        timestamp = stamp.secs + stamp.nsecs * 1e-9  # e.g. 1644823132.492110014

        p = msg.pose.position
        q = msg.pose.orientation

        row = [
            self.index,
            "%.9f" % timestamp,  # string with 9 decimal digits
            p.x, p.y, p.z,
            q.x, q.y, q.z, q.w
        ]

        self.csv_writer.writerow(row)
        self.csv_file.flush()
        self.index += 1

    def on_shutdown(self):
        rospy.loginfo("[RelocPoseCSVLogger] Shutting down, closing CSV file.")
        try:
            self.csv_file.close()
        except Exception:
            pass


if __name__ == "__main__":
    rospy.init_node("reloc_pose_to_csv")
    logger = RelocPoseCSVLogger()
    rospy.spin()
