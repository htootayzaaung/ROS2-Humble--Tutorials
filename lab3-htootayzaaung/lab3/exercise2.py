import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.exceptions import ROSInterruptException
import signal
import math

class SquareWalker(Node):
    def __init__(self):
        super().__init__('square_walker')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.rate = self.create_rate(10)  # 10 Hz

    def move_straight(self):
        move_cmd = Twist()
        move_cmd.linear.x = 0.4
        self.publisher.publish(move_cmd)
        for _ in range(30):
            self.publisher.publish(move_cmd)
            self.rate.sleep()

    def turn(self):
        turn_cmd = Twist()
        turn_cmd.angular.z = math.pi/20
        for _ in range(100):
            self.publisher.publish(turn_cmd)
            self.rate.sleep()

def main():
    def signal_handler(sig, frame):
        first_walker.stop()
        rclpy.shutdown()

    rclpy.init(args=None)
    square_walker = SquareWalker()

    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(square_walker,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            square_walker.move_straight()
            square_walker.turn()
    except ROSInterruptException:
        pass

if __name__ == '__main__':
    main()
