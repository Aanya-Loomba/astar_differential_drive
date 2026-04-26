from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'astar_falcon_planner'
submodules = 'astar_falcon_planner/submodules'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='duality',
    maintainer_email='david@duality.ai',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'falcon_amr_controller = astar_falcon_planner.falcon_amr_controller:main'
        ],
    },
)
