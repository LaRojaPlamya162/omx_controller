from setuptools import find_packages, setup
import os
package_name = 'omx_controller'

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
    maintainer='root',
    maintainer_email='tuananhls16022004@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sac_controller = omx_controller.sac_controller:main',
            'bc_controller = omx_controller.bc_controller:main',
            'iql_controller = omx_controller.iql_controller:main',
            'bc_to_sac_controller = omx_controller.bc_to_sac_controller:main',
            'iql_to_sac_controller = omx_controller.iql_to_sac_controller:main',
            'sac_test_controller = omx_controller.test.sac_controller:main',
            'bc_test_controller = omx_controller.test.bc_controller:main',
            'iql_test_controller = omx_controller.test.iql_controller:main',
            'bc_to_sac_test_controller = omx_controller.test.bc_to_sac_controller:main',
            'iql_to_sac_test_controller = omx_controller.test.iql_to_sac_controller:main',
        ],
    },
)
