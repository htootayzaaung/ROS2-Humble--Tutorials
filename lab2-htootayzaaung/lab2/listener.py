import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Listener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(String, 'chatter', self.listener_callback, 10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: {msg.data!r}')


def main(args=None):
    rclpy.init(args=args)
    listener = Listener()
    rclpy.spin(listener)


if __name__ == '__main__':
    main()