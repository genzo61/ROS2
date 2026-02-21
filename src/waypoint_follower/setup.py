from setuptools import setup

package_name = 'waypoint_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ali',
    maintainer_email='ali@example.com',
    description='TEKNOFEST Waypoint Follower - GPS waypoint navigation via Nav2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'waypoint_follower = waypoint_follower.waypoint_follower:main',
            'teknofest_waypoint_follower = waypoint_follower.teknofest_waypoint_follower:main',
            'fake_odom_pub = waypoint_follower.fake_odom_pub:main',
            'fake_gps_pub = waypoint_follower.fake_gps_pub:main',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/waypoint_follower/launch', ['launch/test_waypoint.launch.py']),
    ],
)