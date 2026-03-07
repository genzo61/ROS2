from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'vehicle_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'tools'), glob('tools/*.sh')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ali',
    maintainer_email='ali@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'igvc_waypoint_navigator = vehicle_bringup.igvc_waypoint_navigator:main',
            'yaris_pilotu = vehicle_bringup.yaris_pilotu:main',
            'lane_tracker = vehicle_bringup.lane_tracker:main',
            'lane_camera_subscriber = vehicle_bringup.lane_camera_subscriber:main',
            'lane_yolo_inference = vehicle_bringup.lane_yolo_inference:main',
            'lane_detection_parser = vehicle_bringup.lane_detection_parser:main',
        ],
    },
)
