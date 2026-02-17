import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Imu, LaserScan
from nav_msgs.msg import Odometry

class SensorRemapper(Node):
    def __init__(self):
        super().__init__('sensor_remapper_node')

        # --- GİRİŞ TOPICLERİ (URDF ile uyumlu) ---
        self.sim_lidar_topic = '/points'  # Senin URDF'te /points olarak ayarlı
        self.sim_imu_topic = '/imu'
        self.sim_odom_topic = '/odom'
        
        # --- ÇIKIŞ TOPICLERİ (Araç Standardı) ---
        self.vehicle_lidar_topic = '/vehicle/lidar/points' 
        self.vehicle_imu_topic = '/vehicle/imu/data'
        self.vehicle_odom_topic = '/vehicle/odom'

        # --- ABONELİKLER ---
        # 3D Lidar (PointCloud2)
        self.create_subscription(PointCloud2, self.sim_lidar_topic, self.lidar_callback, 10)
        # IMU
        self.create_subscription(Imu, self.sim_imu_topic, self.imu_callback, 10)
        # Odometry
        self.create_subscription(Odometry, self.sim_odom_topic, self.odom_callback, 10)

        # --- YAYINCILAR ---
        self.lidar_pub = self.create_publisher(PointCloud2, self.vehicle_lidar_topic, 10)
        self.imu_pub = self.create_publisher(Imu, self.vehicle_imu_topic, 10)
        self.odom_pub = self.create_publisher(Odometry, self.vehicle_odom_topic, 10)

        self.get_logger().info('Sensor Remapper V2 (Teknofest Car) Baslatildi.')

    def lidar_callback(self, msg):
        # Frame ID'yi oldugu gibi koruyalim veya standarda cekelim
        # msg.header.frame_id = "sensor_link" # Senin URDF'teki isim
        self.lidar_pub.publish(msg)

    def imu_callback(self, msg):
        # EKF FIX: Gazebo IMU covariance'i 0 basar. Biz elle dolduruyoruz.
        # [0.01 ...] demek "Sensore cok guveniyorum" demektir.
        if msg.orientation_covariance[0] == 0.0:
            cov = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
            msg.orientation_covariance = cov
            msg.angular_velocity_covariance = cov
            msg.linear_acceleration_covariance = cov
        
        self.imu_pub.publish(msg)

    def odom_callback(self, msg):
        # EKF FIX: Odometri covariance düzeltmesi
        if msg.pose.covariance[0] == 0.0:
            # Pose (Konum) için guven matrisi
            msg.pose.covariance = [
                0.01, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.01, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.01, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.01, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.01, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.01
            ]
            # Twist (Hız) için guven matrisi
            msg.twist.covariance = msg.pose.covariance

        # Frame ID'leri kontrol et
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_footprint"
        
        self.odom_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SensorRemapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
