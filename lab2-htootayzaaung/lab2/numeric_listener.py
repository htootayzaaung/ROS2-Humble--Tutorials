import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int8

class NumericListener(Node):
    def __init__(self):
        super().__init__('numeric_listener')
        self.string_subscription = self.create_subscription(String, 'chatter', self.string_listener_callback, 10)
        self.numeric_subscription = self.create_subscription(Int8, 'numeric_chatter', self.numeric_listener_callback, 10)

    def string_listener_callback(self, msg):
        self.get_logger().info(f'I heard: {msg.data!r}')

    def numeric_listener_callback(self, msg):
        self.get_logger().info(f'I heard: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    numeric_listener = NumericListener()
    rclpy.spin(numeric_listener)

if __name__ == '__main__':
    main()
