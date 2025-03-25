from setuptools import setup

package_name = 'environment'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Javier Pérez Robles',
    maintainer_email='javivi004.ou@gmail.com',
    description="Launches an environment showing all cones and car's position",
    license='MIT',
    entry_points={
        'console_scripts': [
            'env = environment.environment:main',
        ],
    },
)
