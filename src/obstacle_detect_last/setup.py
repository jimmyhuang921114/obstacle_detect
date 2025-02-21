from setuptools import setup

package_name = 'obstacle_detect_last'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],  # ✅ 直接手动指定 package_name
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='darkdemon',
    maintainer_email='jimmyjimmyhuang1114@gmail.com',
    description='Obstacle detection package for ROS2',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'open_rs = obstacle_detect_last.open_rs:main',
            'obs_detect = obstacle_detect_last.obs_detect:main',
            'obs_draw = obstacle_detect_last.obs_draw:main',
        ],
    },
)

