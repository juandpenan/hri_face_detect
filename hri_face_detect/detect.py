from tf2_ros import TransformBroadcaster
from .face import Face
from .face_detector import FaceDetector
from hri_msgs.msg import IdsList
from sensor_msgs.msg import Image, CameraInfo, RegionOfInterest
from std_msgs.msg import Empty, Header
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node


import cv2

import numpy as np


# nb of pixels between the centers of
# two successive regions of interest to
# consider they belong to the same person
MAX_ROIS_DISTANCE = 20

# max scale factor between two successive
# regions of interest to consider they
# belong to the same person
MAX_SCALING_ROIS = 1.2

# face key points
P3D_RIGHT_EYE = (-20.0, -65.5, -5.0)
P3D_LEFT_EYE = (-20.0, 65.5, -5.0)
P3D_RIGHT_EAR = (-100.0, -77.5, -6.0)
P3D_LEFT_EAR = (-100.0, 77.5, -6.0)
P3D_NOSE = (21.0, 0.0, -48.0)
P3D_STOMION = (10.0, 0.0, -75.0)


points_3D = np.array([P3D_NOSE,
                      P3D_RIGHT_EYE,
                      P3D_LEFT_EYE,
                      P3D_STOMION,
                      P3D_RIGHT_EAR,
                      P3D_LEFT_EAR])


class RosFaceDetector(Node):
    def __init__(self,):
        super().__init__('hri_face_detect')

        self.declare_parameter('/humans/faces/width', 128)
        self.declare_parameter('/humans/faces/height', 128)
        self.declare_parameter('/humans/faces/debug', False)
        self.declare_parameter('/humans/faces/face_mesh', True)
        self.declare_parameter('/humans/faces/max_num_faces', 4)

        self.max_num_faces = self.get_parameter(
            '/humans/faces/max_num_faces').get_parameter_value().integer_value
        self.face_width = self.get_parameter(
            '/humans/faces/width').get_parameter_value().integer_value
        self.face_height = self.get_parameter(
            '/humans/faces/height').get_parameter_value().integer_value
        self.is_shutting_down = False
        self.debug = self.get_parameter(
            '/humans/faces/debug').get_parameter_value().bool_value
        self.face_mesh = self.get_parameter(
            '/humans/faces/face_mesh').get_parameter_value().bool_value

        semaphore_pub = self.create_publisher(Empty,
                                              '/hri_face_detect/ready',
                                              1)
        self.faces_pub = self.create_publisher(
            IdsList,
            "/humans/faces/tracked",
            1)

        self.facedetector = FaceDetector(self.face_mesh, self.max_num_faces)
        self.image_sub = self.create_subscription(
            Image,
            "/image",
            self.callback,
            10)
        self.image_info_sub = self.create_subscription(
            CameraInfo,
            "/camera_info",
            self.info_callback,
            10
        )

        self.get_logger().info(
            "Ready. Waiting for images to be published on %s." %
            self.image_sub.topic)
        semaphore_pub.publish(Empty())

        # last-used face id
        self.last_id = 1

        # list (map ID -> Face) of Face instances, corresponding to the
        # currently tracked faces
        self.knownFaces = {}

        self.tb = TransformBroadcaster(self)

    def distance_rois(self, bb1, bb2):
        x1, y1 = bb1.x_offset + bb1.width / 2, bb1.y_offset + bb1.height / 2
        x2, y2 = bb2.x_offset + bb2.width / 2, bb2.y_offset + bb2.height / 2

        return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

    def find_previous_match(self, bb):
        for _, face in self.knownFaces.items():
            prev_bb = face.bb
            if (self.distance_rois(prev_bb, bb) < MAX_ROIS_DISTANCE *
                MAX_ROIS_DISTANCE and 1 /
                MAX_SCALING_ROIS < prev_bb.width /
                bb.width < MAX_SCALING_ROIS and 1 /
                MAX_SCALING_ROIS < prev_bb.height /
                    bb.height < MAX_SCALING_ROIS):
                return face
        return None

    def info_callback(self, cameraInfo):

        if not hasattr(self, "cameraInfo"):
            self.cameraInfo = cameraInfo

            self.K = np.zeros((3, 3), np.float32)
            self.K[0][0:3] = self.cameraInfo.k[0:3]
            self.K[1][0:3] = self.cameraInfo.k[3:6]
            self.K[2][0:3] = self.cameraInfo.k[6:9]

    def callback(self, rgb_msg):

        if self.is_shutting_down:
            return

        # copy the list of face ID before iterating over detection, so that we
        # can delete non-existant faces at the end.
        knownIds = list(self.knownFaces.keys())

        image = CvBridge().imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        img_height, img_width, _ = image.shape
        camera_optical_frame = rgb_msg.header.frame_id

        if not self.face_mesh:
            detections = self.facedetector.detect(image, img_width, img_height)
        else:
            detections = self.facedetector.detect_face_mesh(
                image, img_height, img_width
            )

        currentIds = []

        for detection in detections:
            x, y, w, h = detection["bb"]
            bb = RegionOfInterest()
            bb.x_offset = max(0, x)
            bb.y_offset = max(0, y)
            bb.width = min(img_height - y, h)
            bb.height = min(img_width - x, w)
            bb.do_rectify = True

            # have we seen this face before? -> check based on whether or not
            # bounding boxes overlaps
            face = self.find_previous_match(bb)

            if face:
                # we re-detect a face!

                currentIds.append(face.id)
                face.nb_frames_visible += 1

                # if it is a 2nd frame, we create all the required publishers.
                if face.nb_frames_visible == 2:
                    face.initialise_publishers()

                # update the face with its current position and landmarks
                face.bb = bb
                face.nose_tip = detection['nose_tip']
                face.mouth_center = detection['mouth_center']
                face.right_eye = detection['right_eye']
                face.left_eye = detection['left_eye']
                face.right_ear_tragion = detection['right_ear_tragion']
                face.left_ear_tragion = detection['left_ear_tragion']
                face.facial_landmarks_msg = detection['facial_landmarks_msg']

            else:
                face = Face(
                    node=self,
                    width=self.face_width,
                    height=self.face_height)
                face.bb = bb
                currentIds.append(face.id)
                self.knownFaces[face.id] = face

        # iterate over faces not seen anymore,
        # and unregister corresponding publishers
        for id in knownIds:
            if id not in currentIds:
                self.knownFaces[id].delete()
                del self.knownFaces[id]

        for _, face in self.knownFaces.items():
            if face.ready:
                if not self.is_shutting_down:
                    face.publish()
                if not self.is_shutting_down:
                    face.publish_images(image)

                if not self.is_shutting_down:
                    if hasattr(self, "K"):
                        face.compute_6d_pose(
                            self.K, image, camera_optical_frame)

                        self.tb.sendTransform(face.head_transform)
                        self.tb.sendTransform(face.gaze_transform)
        faces_msg = IdsList()
        faces_msg.header = rgb_msg.header
        faces_msg.ids = [
            face.id for face in self.knownFaces.values() if face.ready]
        self.faces_pub.publish(faces_msg)
        faces_msg = None

        if self.debug:
            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            self.get_logger().info('%s faces detected' % len(self.knownFaces))
            for _, face in self.knownFaces.items():
                if not face.ready:
                    continue
                cv2.rectangle(
                    image,
                    (face.bb.x_offset, face.bb.y_offset),
                    (
                        face.bb.x_offset + face.bb.width,
                        face.bb.y_offset + face.bb.height,
                    ),
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    image,
                    face.id,
                    (face.bb.x_offset, face.bb.y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                )

            cv2.imshow('MediaPipe Face Detection', image)
            cv2.waitKey(5)

    def close(self):

        self.get_logger().info("Stopping face publishing...")

        self.is_shutting_down = True

        for _, face in self.knownFaces.items():
            face.delete()

        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        ids_list = IdsList()
        ids_list.header = h
        ids_list.ids = []
        self.faces_pub.publish(ids_list)
        self.get_logger().info('Stopped publishing faces.')
        rate = self.create_rate(10)
        rate.sleep()
        rate.destroy()


def main(args=None):
    rclpy.init(args=args)
    detector = RosFaceDetector()
    rclpy.spin(detector)


if __name__ == "__main__":
    main()
