# Part B - Stitching images.

import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from std_msgs.msg import UInt32
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal

class imageStitcher(Node):
    def __init__(self):

        super().__init__('imageStitcher')

        # Initialise a publisher to publish messages to the robot base
        # We covered which topic receives messages that move the robot in the 3rd Lab Session

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.rate = self.create_rate(10)  # 10 Hz

        # Remember to initialise a CvBridge() and set up a subscriber to the image topic you wish to use
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.subscription  # prevent unused variable warning


    def callback(self, data):
        # Convert the received image into a opencv image
        # But remember that you should always wrap a call to this conversion method in an exception handler
        image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        cv2.namedWindow('camera_Feed',cv2.WINDOW_NORMAL)
        cv2.imshow('camera_Feed', image)
        cv2.resizeWindow('camera_Feed',320,240)
        

        # You can use openCV to detect if a key is pressed
        # cv2.waitkey() will return a value > -1 if a key is pressed
        # You can test if a particular key is pressed using the ord method.
        # For example if key = ord("i"): would test key i was pressed

        key = cv2.waitKey(3)

    def rotate_robot(self):

        # Initialize unused components of desired velocity to zero
        desired_velocity = Twist()
        desired_velocity.linear.x = 0.
        desired_velocity.linear.y = 0.
        desired_velocity.linear.z = 0.
        desired_velocity.angular.x = 0.
        desired_velocity.angular.y = 0.


        # Set desired angle in radians
        # desired_velocity..... =

        # store current time: t0
        t0, _ = self.get_clock().now().seconds_nanoseconds()

        current_angle = 0

        # loop to publish the velocity estimate until desired angle achieved
        # current angle = current angular velocity * (t1 - t0)
        while (current_angle < self.angle_r):
            # Publish the velocity
            self.publisher.publish(desired_velocity)

            # t1 is the current time
            t1, _ = self.get_clock().now().seconds_nanoseconds()  # to_msg()

            # Calculate current angle
            # current_angle =

            self.rate.sleep()

        # set velocity to zero to stop the robot
        self.stop()


    def performStitch(self):
        # Initialize the SIFT feature detector and extractor


        # Detect keypoints and compute descriptors for both images


        # Draw keypoints on the images


        # Display the images with keypoints


        # Initialize the feature matcher using brute-force matching


        # Match the descriptors using brute-force matching


        # Sort the matches by distance (lower is better)


        # Draw the top N matches


        # Display the images with matches


        # Estimate the homography matrix using RANSAC


        # Print the estimated homography matrix


        # Warp the first image using the homography


        # Blending the warped image with the second image using alpha blending



        # Display the blended image
        
        return


    def stop(self):
        desired_velocity = Twist()
        desired_velocity.linear.x = 0.0  # Send zero velocity to stop the robot
        self.publisher.publish(desired_velocity)

        

# Create a node of your class in the main and ensure it stays up and running
# handling exceptions and such
def main():

    
    def signal_handler(sig, frame):
        stitcher.stop()
        rclpy.shutdown()

    # Instantiate your class
    # and rclpy.init the entire node
    
    rclpy.init(args=None)
    stitcher = imageStitcher()
    
    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(stitcher,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():

            # check if a key has been pressed and store the current image

            # When you have an image rotate the robot
            # stitcher.robot_rotate()

                
            # Once you have both images we can try to stitch them together
            # but make sure you don't capture any more images
            # stitcher.performStitch()
            pass


    except ROSInterruptException:
            pass

    # Remember to destroy all image windows before closing node
    cv2.destroyAllWindows()

# Check if the node is executing in the main path
if __name__ == '__main__':
    main()
