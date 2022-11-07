from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hri_face_detect',
            executable='detect',
            name='detect',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'face_mesh': 'earth'},
                {'max_num_faces': '10'}        
                 
            ],
            remappings=[
                ('image', '/camera/rgb/image_raw'),
                ('camera_info', '/camera/rgb/camera_info'),
            ],
        )
    ])


    # <arg name="rgb_camera" default="/camera/color/"/>
    # <arg name="rgb_camera_topic" default="$(arg rgb_camera)/image_raw"/>
    # <arg name="rgb_camera_info" default="$(arg rgb_camera)/camera_info"/>

    # <arg name="face_mesh" default="True"/>
    # <arg name="max_num_faces" default="10"/>

    # <node pkg="hri_face_detect" name="hri_face_detect" type="detect" output="screen">
    #     <param name="face_mesh" value="$(arg face_mesh)"/>
    #     <param name="max_num_faces" value="$(arg max_num_faces)"/>
    #     <remap from="image" to="$(arg rgb_camera_topic)"/>
    #     <remap from="camera_info" to="$(arg rgb_camera_info)"/>

