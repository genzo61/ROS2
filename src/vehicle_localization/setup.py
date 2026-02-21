from setuptools import setup, find_packages
import os
from glob import glob

setup(
    name='vehicle_localization',
    version='0.0.0',
    packages=find_packages(include=['vehicle_localization', 'vehicle_localization.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/vehicle_localization']),
        ('share/vehicle_localization', ['package.xml']),
        (os.path.join('share', 'vehicle_localization', 'config'), glob('config/*.yaml')),
        (os.path.join('share', 'vehicle_localization', 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ali',
    maintainer_email='ali@example.com',
    description='Vehicle localization package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ekf_filter_node = vehicle_localization.ekf_filter_node:main',
        ],
    },
)