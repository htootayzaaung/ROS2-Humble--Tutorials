import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math

class CircularWalker(Node):
    def __init__(self):
        super().__init__('circular_walker')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.rate = self.create_rate(10)  # 10 Hz

    def walk_in_circle(self):
        desired_velocity = Twist()
        desired_velocity.linear.x = 0.2  # Adjust linear velocity as needed
        desired_velocity.angular.z = math.pi / 6  # 30 degrees in radians, adjust as needed

        while rclpy.ok():
            self.publisher.publish(desired_velocity)
            self.rate.sleep()

def main():
    rclpy.init()
    circular_walker = CircularWalker()

    try:
        circular_walker.walk_in_circle()
    except KeyboardInterrupt:
        pass  # Handle Ctrl-C gracefully
    finally:
        # Stop the robot before shutting down the node
        circular_walker.publisher.publish(Twist())
        circular_walker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
