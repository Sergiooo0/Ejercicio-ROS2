import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from auria_msgs.msg import PointArray
import numpy as np

class ControllerCar(Node):
    HZ = 10  # Frecuencia en hercios

    def __init__(self):
        super().__init__('nodo')
        self.logger = self.get_logger()
        self.logger.info('Nodo creado')

        self.cones_blue = []
        self.cones_yellow = []

        self.publisher = self.create_publisher(Float32, "/car/acceleration", self.HZ)
        self.subscription = self.create_subscription(PointArray, '/vision/nearby_cones/blue', self.save_cones_blue, 10)
        self.subscription2 = self.create_subscription(PointArray, '/vision/nearby_cones/yellow', self.save_cones_yellow, 10)
        
        self.timer = self.create_timer(0.1, self.publish_acc)

    def publish_acc(self):
        msg = Float32()
        msg.data = 0.1
        self.publisher.publish(msg)

    def save_cones_blue(self, msg):
        self.cones_blue = []
        self.cones_blue.extend(msg.points)
        self.logger.info(f"Conos blue: {self.cones_blue}")
    
    def save_cones_yellow(self, msg):
        self.cones_yellow = []
        self.cones_yellow.extend(msg.points)

    def update(self):
        if self.cones_blue.points == [] or self.cones_yellow.points == []:
            self.logger.warn("No hay conos en el sensor")
            return

def main(args=None):
    rclpy.init(args=args)
    node = ControllerCar()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print("Finalizando nodo")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
