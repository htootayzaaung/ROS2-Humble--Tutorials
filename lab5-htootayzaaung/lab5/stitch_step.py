import threading
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.exceptions import ROSInterruptException
import math
import signal

class imageStitcher(Node):
    def __init__(self):
        super().__init__('image_stitcher')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.rate = self.create_rate(10)  # 10 Hz
        self.image1_acquired = False
        self.image2_acquired = False
        self.initial_rotation_started = False
        self.angle_r = math.pi / 4
        self.rotating_complete = False
        
    def callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        cv2.imshow('Camera Feed', image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('i'):  # Key to capture image
            if not self.image1_acquired:
                self.image1 = image
                self.image1_acquired = True
                print("First image acquired.")
                # After acquiring the first image, ensure the robot continues to rotate if not already doing so

            """
            elif self.image1_acquired and not self.image2_acquired:
                self.image2 = image
                self.image2_acquired = True
                print("Second image acquired.")
            """
            
        if self.rotating_complete == True:
            self.image2 = image
            self.image2_acquired = True
                
    def rotate_robot(self):
        desired_velocity = Twist()
        desired_velocity.linear.x = 0.
        desired_velocity.linear.y = 0.
        desired_velocity.linear.z = 0.
        desired_velocity.angular.x = 0.
        desired_velocity.angular.y = 0.

        desired_velocity = Twist()
        current_angle = 0.0
        desired_velocity.angular.z = -math.pi / 12
        t0, _ = self.get_clock().now().seconds_nanoseconds()
        current_angle = 0

        while current_angle < self.angle_r:
            self.publisher.publish(desired_velocity)
            t1, _ = self.get_clock().now().seconds_nanoseconds()  # to_msg()
            current_angle = -desired_velocity.angular.z * (t1 - t0)
            self.rate.sleep()

        self.stop()  # Stop the robot after rotating the specified angle
        self.rotating_complete = True

    def stop(self):
        desired_velocity = Twist()
        desired_velocity.angular.z = 0.0
        self.publisher.publish(desired_velocity)
        self.initial_rotation_started = False

    def performStitch(self):
        sift = cv2.SIFT_create()

        keypoints1, descriptors1 = sift.detectAndCompute(self.image1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(self.image2, None)

        image1_keypoints = cv2.drawKeypoints(self.image1, keypoints1, None)
        image2_keypoints = cv2.drawKeypoints(self.image2, keypoints2, None)

        cv2.namedWindow("Image 1 with Keypoints", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Image 2 with Keypoints", cv2.WINDOW_NORMAL)
        cv2.imshow("Image 1 with Keypoints", image1_keypoints)
        cv2.imshow("Image 2 with Keypoints", image2_keypoints)
        cv2.resizeWindow("Image 1 with Keypoints", 320, 240)
        cv2.resizeWindow("Image 2 with Keypoints", 320, 240)
        cv2.waitKey(0)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches_bf = bf.match(descriptors1, descriptors2)
        matches_bf = sorted(matches_bf, key=lambda x: x.distance)

        num_matches = 50
        image_matches_bf = cv2.drawMatches(self.image1, keypoints1, self.image2, keypoints2, matches_bf[:num_matches], None)
        cv2.namedWindow("Brute-Force Matching", cv2.WINDOW_NORMAL)
        cv2.imshow('Brute-Force Matching', image_matches_bf)
        cv2.resizeWindow('Brute-Force Matching', 640, 480)
        cv2.waitKey(0)

        tp = []  # target points
        qp = []  # query points
        for m in matches_bf:
            tp.append(keypoints2[m.trainIdx].pt)
            qp.append(keypoints1[m.queryIdx].pt)
        tp, qp = np.float32((tp, qp))

        homography, _ = cv2.findHomography(qp, tp, cv2.RANSAC, 5.0)

        result = cv2.warpPerspective(self.image2, homography, (self.image1.shape[1]+300, self.image1.shape[0]))
        padded_left_img = cv2.copyMakeBorder(self.image1, 0, 0, 0, result.shape[1] - self.image1.shape[1],cv2.BORDER_CONSTANT )
        alpha = 0.5  
        blended_image = cv2.addWeighted(padded_left_img, alpha, result, 1 - alpha, 0)

        cv2.namedWindow("Blended Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Blended Image", blended_image)
        cv2.resizeWindow('Blended Image', 1280, 960)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
            if (stitcher.image1_acquired == True):
                 # When you have an image rotate the robot
                stitcher.rotate_robot()
            if (stitcher.rotating_complete == True):
                # Once you have both images we can try to stitch them together
                # but make sure you don't capture any more images
                stitcher.performStitch()


    except ROSInterruptException:
            pass

    # Remember to destroy all image windows before closing node
    cv2.destroyAllWindows()

# Check if the node is executing in the main path
if __name__ == '__main__':
    main()
