import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Point
from auria_msgs.msg import PointArray
from std_msgs.msg import String
import numpy as np
import math

class ControllerCar(Node):
    HZ = 10  # Frecuencia en hercios

    def __init__(self, n_cones=3, log = False):
        super().__init__('nodo')
        self.log = log
        self.logger = self.get_logger()
        self.logger.info('Nodo controlador de coche creado')

        # Dos columnas: x, y
        self.cones_blue = np.empty((0, 2))
        self.cones_yellow = np.empty((0, 2))

        self.car_position = None
        self.car_yaw = None
        self.car_steering = 0.0

        # Número de conos más cercanos a considerar
        # Dos o tres conos, son los mejores valores
        self.n_cones = n_cones

        self.counter = 0

        self.accelerate_pub = self.create_publisher(Float32, "/car/acceleration", self.HZ)
        self.steering_pub = self.create_publisher(Float32, "/car/steering", self.HZ)
        self.state_pub = self.create_publisher(String, "/car/state", self.HZ)

        self.create_subscription(Point, '/env/car_position', self.update_car_position, self.HZ)
        self.create_subscription(PointArray, '/vision/nearby_cones/blue', self.save_cones_blue, self.HZ)
        self.create_subscription(PointArray, '/vision/nearby_cones/yellow', self.save_cones_yellow, self.HZ)
        self.create_subscription(Float32, '/env/car_yaw', self.update_car_yaw, self.HZ)
        
        self.timer = self.create_timer(0.1, self.publish_acc)

    def publish_acc(self):
        self.counter += 1
        if self.counter == 1:
            # Publicamos una aceleración inicial para que el coche empiece a moverse
            # Esto una única vez
            msg = Float32()
            msg.data = 0.1
            self.accelerate_pub.publish(msg)
            return
        if self.counter == 500:
            # Publicamos una aceleración cero para que el coche se mantenga en marcha
            msg = Float32()
            msg.data = 0.0
            self.accelerate_pub.publish(msg)
            return
        # Comprobamos si tenemos información suficiente
        if not self.check_information():
            return

        # Tomamos los conos más cercanos (el primero por simplicidad)
        blues, blue_dist = self.find_closest_cone(self.cones_blue, n_cones=self.n_cones)
        yellows, yellow_dist = self.find_closest_cone(self.cones_yellow, n_cones=self.n_cones)

        if blues.shape[0] == 0 or yellows.shape[0] == 0:
            # Si no hay conos delante del coche, no hacemos nada
            return

        # Punto medio del pasillo
        centers_x = (blues[:, 0] + yellows[:, 0]) / 2.0
        centers_y = (blues[:, 1] + yellows[:, 1]) / 2.0
        # Calculamos el centro del pasillo
        # Hacemos una media ponderada para evitar que coja curvas muy cerradas
        # A menor distancia del coche, más peso
        try:
            center_x = np.average(centers_x, weights=1.0 / blue_dist)
            center_y = np.average(centers_y, weights=1.0 / yellow_dist)
        except ZeroDivisionError:
            self.logger.info("División por cero")
            center_x = np.mean(centers_x)
            center_y = np.mean(centers_y)

        # Vector desde el coche al centro del pasillo
        dx = center_x - self.car_position.x
        dy = center_y - self.car_position.y

        # Ángulo al centro desde la posición del coche
        angle_to_center = math.atan2(dy, dx)

        # Diferencia entre el ángulo deseado y el actual (yaw)
        steering_angle = angle_to_center - self.car_yaw

        # Normalizamos a [-pi, pi] por si se pasa de rango
        steering_angle = (steering_angle + math.pi) % (2 * math.pi) - math.pi

        # Publicamos el ángulo de dirección
        steer_msg = Float32()
        steer_msg.data = steering_angle
        self.steering_pub.publish(steer_msg)

        if self.log:
            self.logger.info(f"Steering angle: {steering_angle:.2f} rad (target angle: {angle_to_center:.2f}, yaw: {self.car_yaw:.2f})")
        
    def update_car_position(self, msg):
        self.car_position = msg
    
    def update_car_yaw(self, msg):
        self.car_yaw = msg.data

    def save_cones_blue(self, msg):
        self.cones_blue = np.array([[p.x, p.y] for p in msg.points])

    def save_cones_yellow(self, msg):
        self.cones_yellow = np.array([[p.x, p.y] for p in msg.points])


    def find_closest_cone(self, cones: np.ndarray, n_cones: int = 4) -> np.ndarray:
        """
        Encuentra los conos más cercanos al coche.
        :param cones: Array de conos (Nx2) donde cada fila es [x, y]
        :return: El cono más cercano al coche.
        """
        car_x = self.car_position.x
        car_y = self.car_position.y
        dir_x = math.cos(self.car_yaw)
        dir_y = math.sin(self.car_yaw)
        dx = cones[:, 0] - car_x
        dy = cones[:, 1] - car_y
        dot = dx * dir_x + dy * dir_y
        # Filtrar solo conos que están delante del coche
        valid_cones = cones[dot > 0]

        if valid_cones.shape[0] == 0:
            return np.empty((0, 2)), np.empty((0))
        #Calculamos la distancia de los conos filtrados al coche
        dist = np.linalg.norm(valid_cones - np.array([car_x, car_y]), axis=1)
        # Ordenamos por distancia
        order_idx = np.argsort(dist)
        sorted_cones = valid_cones[np.argsort(dist)]
        sorted_dist = dist[order_idx]
        # Buscamos los 4 más cercanos
        closest_cones = sorted_cones[0:n_cones]
        closest_dist = sorted_dist[0:n_cones]

        return closest_cones, closest_dist
    
    def check_information(self) -> bool:
        """
        Comprueba si hay información suficiente para calcular la dirección.
        """
        if self.cones_blue.shape[0] == 0 or self.cones_yellow.shape[0] == 0:
            if self.log:
                self.logger.info("Aún no hay información sobre los conos")
            return False
        
        if self.car_position is None:
            if self.log:
                self.logger.info("Aún no hay información sobre la posición del coche")
            return False
        
        if self.car_yaw is None:
            if self.log:
                self.logger.info("Aún no hay información sobre el yaw del coche")
            return False
        return True



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
