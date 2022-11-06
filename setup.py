from setuptools import setup

package_name = 'hri_face_detect'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='juan',
    maintainer_email='juan97pena@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
             'detect = hri_face_detect.detect:main',
             'record = hri_face_detect.record:main',
        ],
    },
)
