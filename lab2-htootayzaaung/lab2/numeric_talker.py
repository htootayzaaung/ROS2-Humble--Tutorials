import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int8

class NumericTalker(Node):
    def __init__(self):
        super().__init__('numeric_talker')
        self.string_publisher = self.create_publisher(String, 'chatter', 10)
        self.numeric_publisher = self.create_publisher(Int8, 'numeric_chatter', 10)

        timer_in_seconds = 0.5
        self.timer = self.create_timer(timer_in_seconds, self.talker_callback)
        self.counter = 0

    def talker_callback(self):
        string_msg = String()
        string_msg.data = f'Hello World, {self.counter}'
        self.string_publisher.publish(string_msg)
        self.get_logger().info(f'Publishing: {string_msg.data}')

        numeric_msg = Int8()
        numeric_msg.data = self.counter
        self.numeric_publisher.publish(numeric_msg)
        self.get_logger().info(f'Publishing: {numeric_msg.data}')

        self.counter += 1
        if self.counter > 127:
            self.counter = 0

def main(args=None):
    rclpy.init(args=args)
    numeric_talker = NumericTalker()
    rclpy.spin(numeric_talker)

if __name__ == '__main__':
    main()
