import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from std_msgs.msg import Float32
from std_msgs.msg import String
import math

class Car(Node):
    def __init__(self):
        super().__init__('car_node')

        self.x = 0.0
        self.y = 0.0
        self.yaw = math.pi / 2
        self.velocity = 0.0  # m/s
        self.steering_angle = 0.0  # rad
        self.acceleration = 0.0  # m/s²
        self.state = "running"

        self.create_subscription(String, '/car/state', self.update_state, 10)
        self.create_subscription(Float32, '/car/steering', self.update_steering, 10)
        self.create_subscription(Float32, '/car/acceleration', self.update_acceleration, 10)

        self.location_publisher = self.create_publisher(Point, '/env/car_position', 10)
        self.yaw_publisher = self.create_publisher(Float32, '/env/car_yaw', 10)

        self.timer_period = .1
        self.timer = self.create_timer(self.timer_period, self.update)

        self.wheel_base = 0.5  # edjes between edjes

        self.get_logger().info("Car node running successfully")

    
    def update_state(self, msg):
        if self.state == "running" and msg.data == "finish":
            self.get_logger().info("State changed to finish. Stopping car.")
            self.state = "finish"
            self.velocity = 0.0
            self.acceleration = 0.0
        elif self.state == "finish" and msg.data == "running":
            self.get_logger().warn("Cannot return to running from finish. Ignoring.")

    def update_steering(self, msg):
        self.steering_angle = msg.data

    def update_acceleration(self, msg):
        self.acceleration = msg.data

    def update(self):
        if self.state != "running":
            return

        dt = self.timer_period
        self.velocity += self.acceleration * dt

        if abs(self.steering_angle) > 1e-4:
            turning_radius = self.wheel_base / math.tan(self.steering_angle)
            angular_velocity = self.velocity / turning_radius
        else:
            angular_velocity = 0.0

        self.yaw += angular_velocity * dt
        self.x += self.velocity * math.cos(self.yaw) * dt
        self.y += self.velocity * math.sin(self.yaw) * dt

        msg = Point()
        msg.x = self.x
        msg.y = self.y
        msg.z = 0.0

        yaw_msg = Float32()
        yaw_msg.data = self.yaw

        self.location_publisher.publish(msg)
        self.yaw_publisher.publish(yaw_msg)

def main(args=None):
    rclpy.init(args=args)
    node = Car()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
