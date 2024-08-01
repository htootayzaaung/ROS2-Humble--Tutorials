# Exercise 3 - If green object is detected, and above a certain size, then send a message (print or use lab2)

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


    
class colourIdentifier(Node):
    def __init__(self):
        super().__init__('cI')
        # Initialise any flags that signal a colour has been detected (default to false)

        # Initialise the value you wish to use for sensitivity in the colour detection (10 should be enough)

        # Remember to initialise a CvBridge() and set up a subscriber to the image topic you wish to use
        # We covered which topic to subscribe to should you wish to receive image data
        self.sensitivity = 15
        self.green_detected = False  # Flag for green color detection
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        
    def callback(self, data):
        # But remember that you should always wrap a call to this conversion method in an exception handler
        try:
            # Convert the received image into a opencv image
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)

        # Set the upper and lower bounds for the colour you wish to identify - green
        hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
        # Convert the rgb image into a hsv image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Filter out everything but a particular colour using the cv2.inRange() method
        green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
        # Apply the mask to the original image using the cv2.bitwise_and() method
        result_image = cv2.bitwise_and(image, image, mask=green_mask)

        # Find the contours that appear within the certain colour mask using the cv2.findContours() method
        # For <mode> use cv2.RETR_LIST for <method> use cv2.CHAIN_APPROX_SIMPLE

        contours, _ = cv2.findContours(green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Loop over the contours
            # There are a few different methods for identifying which contour is the biggest:
            # Loop through the list and keep track of which contour is biggest or
            # Use the max() method to find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            #Moments can calculate the center of the contour
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            area = cv2.contourArea(largest_contour)
            
            #Check if the area of the shape you want is big enough to be considered
            # If it is then change the flag for that colour to be True(1)
            #<What do you think is a suitable area?>
            if area > 1000:  # Threshold for area
                self.green_detected = True
                # draw a circle on the contour you're identifying
                #minEnclosingCircle can find the centre and radius of the largest contour(result from max())
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                center = (int(x), int(y))
                # Then alter the values of any flags
                cv2.circle(result_image, center, int(radius), (0, 255, 0), 2)

        if self.green_detected:
            print("Green color detected!")

            #if the flag is true (colour has been detected)
            #print the flag or colour to test that it has been detected
            #alternatively you could publish to the lab1 talker/listener

        #Show the resultant images you have created. You can show all of them or just the end result if you wish to.
        cv2.imshow('Detected Colors', result_image)
        cv2.waitKey(3)


# Create a node of your class in the main and ensure it stays up and running
# handling exceptions and such
def main():
    def signal_handler(sig, frame):
        rclpy.shutdown()

    
    # Instantiate your class
    # And rclpy.init the entire node
    rclpy.init(args=None)
   
    cI = colourIdentifier()

    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(cI,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            continue
    except ROSInterruptException:
        pass

    # Remember to destroy all image windows before closing node
    cv2.destroyAllWindows()


# Check if the node is executing in the main path
if __name__ == '__main__':
    main()
