from setuptools import find_packages, setup

package_name = 'vehicle_sensor_adapters'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'sensor_remapper = vehicle_sensor_adapters.sensor_remapper:main',
            'fake_rtk_node = vehicle_sensor_adapters.fake_rtk_node:main',
        ],
    },
)
