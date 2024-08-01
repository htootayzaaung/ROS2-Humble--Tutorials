from __future__ import division
import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal

class Robot(Node):
    def __init__(self):
        super().__init__('robot')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sensitivity = 10
        self.green_detected = False
        self.red_detected = False
        self.forward = Twist()
        self.forward.linear.x = 0.2
        self.stop = Twist()
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        green_lower = np.array([60 - self.sensitivity, 100, 100])
        green_upper = np.array([60 + self.sensitivity, 255, 255])
        red_lower = np.array([0, 100, 100])
        red_upper = np.array([10, 255, 255])

        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        red_mask = cv2.inRange(hsv, red_lower, red_upper)

        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if green_contours:
            largest_green = max(green_contours, key=cv2.contourArea)
            if cv2.contourArea(largest_green) > 1000:
                self.green_detected = True
                self.red_detected = False
            else:
                self.green_detected = False
        else:
            self.green_detected = False

        if red_contours:
            largest_red = max(red_contours, key=cv2.contourArea)
            if cv2.contourArea(largest_red) > 1000:
                self.red_detected = True
                self.green_detected = False
            else:
                self.red_detected = False
        else:
            self.red_detected = False
        cv2.imshow("Robot POV", cv_image)

    def control_robot(self):
        if self.green_detected and not self.red_detected:
            self.publisher.publish(self.forward)
        elif self.red_detected:
            self.publisher.publish(self.stop)

def main(args=None):
    rclpy.init(args=args)
    robot = Robot()

    def shutdown_hook():
        robot.publisher.publish(Twist())
        rclpy.shutdown()

    signal.signal(signal.SIGINT, lambda sig, frame: shutdown_hook())

    while rclpy.ok():
        rclpy.spin_once(robot, timeout_sec=0.1)
        robot.control_robot()

    robot.destroy_node()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
