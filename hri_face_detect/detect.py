from tf2_ros import TransformBroadcaster
from .utils import quaternion_from_euler, face_pose_estimation
from hri_msgs.msg import IdsList, FacialLandmarks, NormalizedPointOfInterest2D
from sensor_msgs.msg import Image, CameraInfo, RegionOfInterest
from geometry_msgs.msg import Transform, TransformStamped, Vector3, Quaternion
from std_msgs.msg import Empty, Header
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
import math
import random
import cv2
from PIL import Image as PILImage
import numpy as np
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# nb of pixels between the centers of
# two successive regions of interest to
# consider they belong to the same person
MAX_ROIS_DISTANCE = 20

# max scale factor between two successive
# regions of interest to consider they
# belong to the same person
MAX_SCALING_ROIS = 1.2

# default size in pixels for the re-published faces
# can be changed via the ROS parameters
# /humans/faces/width and /humans/faces/height
cropped_face_width = 128
cropped_face_height = 128

# face key points
P3D_RIGHT_EYE = (-20.0, -65.5, -5.0)
P3D_LEFT_EYE = (-20.0, 65.5, -5.0)
P3D_RIGHT_EAR = (-100.0, -77.5, -6.0)
P3D_LEFT_EAR = (-100.0, 77.5, -6.0)
P3D_NOSE = (21.0, 0.0, -48.0)
P3D_STOMION = (10.0, 0.0, -75.0)


points_3D = np.array(
    [P3D_NOSE, P3D_RIGHT_EYE, P3D_LEFT_EYE, P3D_STOMION, P3D_RIGHT_EAR, P3D_LEFT_EAR]
)

# ros4hri to mediapipe mapping
"""
ros4hri facial landmarks ref: https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/.github/media/keypoints_face.png
mediapipe face mesh landmarks ref: https://i.stack.imgur.com/5Mohl.jpg 
"""
ros4hri_to_mediapipe = [None] * 68
# The ROS4HRI FacialLandmarks message defines 70 landmarks,
# however Mediapipe Face Mesh estimator does not provide
# an estimation for the pupils position ==> 70 - 2 = 68

ros4hri_to_mediapipe[FacialLandmarks.RIGHT_EAR] = 34
ros4hri_to_mediapipe[FacialLandmarks.RIGHT_PROFILE_1] = 227
ros4hri_to_mediapipe[FacialLandmarks.RIGHT_PROFILE_2] = 137
ros4hri_to_mediapipe[FacialLandmarks.RIGHT_PROFILE_3] = 177
ros4hri_to_mediapipe[FacialLandmarks.RIGHT_PROFILE_4] = 215
ros4hri_to_mediapipe[FacialLandmarks.RIGHT_PROFILE_5] = 135
ros4hri_to_mediapipe[FacialLandmarks.RIGHT_PROFILE_6] = 170
ros4hri_to_mediapipe[FacialLandmarks.RIGHT_PROFILE_7] = 171
ros4hri_to_mediapipe[FacialLandmarks.MENTON] = 175
ros4hri_to_mediapipe[FacialLandmarks.LEFT_EAR] = 264
ros4hri_to_mediapipe[FacialLandmarks.LEFT_PROFILE_1] = 447
ros4hri_to_mediapipe[FacialLandmarks.LEFT_PROFILE_2] = 366
ros4hri_to_mediapipe[FacialLandmarks.LEFT_PROFILE_3] = 401
ros4hri_to_mediapipe[FacialLandmarks.LEFT_PROFILE_4] = 435
ros4hri_to_mediapipe[FacialLandmarks.LEFT_PROFILE_5] = 364
ros4hri_to_mediapipe[FacialLandmarks.LEFT_PROFILE_6] = 395
ros4hri_to_mediapipe[FacialLandmarks.LEFT_PROFILE_7] = 396
ros4hri_to_mediapipe[FacialLandmarks.RIGHT_EYEBROW_OUTSIDE] = 70
ros4hri_to_mediapipe[FacialLandmarks.RIGHT_EYEBROW_1] = 63
ros4hri_to_mediapipe[FacialLandmarks.RIGHT_EYEBROW_2] = 105
ros4hri_to_mediapipe[FacialLandmarks.RIGHT_EYEBROW_3] = 66
ros4hri_to_mediapipe[FacialLandmarks.RIGHT_EYEBROW_INSIDE] = 107
ros4hri_to_mediapipe[FacialLandmarks.LEFT_EYEBROW_OUTSIDE] = 300
ros4hri_to_mediapipe[FacialLandmarks.LEFT_EYEBROW_1] = 293
ros4hri_to_mediapipe[FacialLandmarks.LEFT_EYEBROW_2] = 334
ros4hri_to_mediapipe[FacialLandmarks.LEFT_EYEBROW_3] = 296
ros4hri_to_mediapipe[FacialLandmarks.LEFT_EYEBROW_INSIDE] = 336
ros4hri_to_mediapipe[FacialLandmarks.RIGHT_EYE_OUTSIDE] = 130
ros4hri_to_mediapipe[FacialLandmarks.RIGHT_EYE_TOP_1] = 29
ros4hri_to_mediapipe[FacialLandmarks.RIGHT_EYE_TOP_2] = 28
ros4hri_to_mediapipe[FacialLandmarks.RIGHT_EYE_INSIDE] = 243
ros4hri_to_mediapipe[FacialLandmarks.RIGHT_EYE_BOTTOM_1] = 24
ros4hri_to_mediapipe[FacialLandmarks.RIGHT_EYE_BOTTOM_2] = 22
ros4hri_to_mediapipe[FacialLandmarks.LEFT_EYE_OUTSIDE] = 359
ros4hri_to_mediapipe[FacialLandmarks.LEFT_EYE_TOP_1] = 259
ros4hri_to_mediapipe[FacialLandmarks.LEFT_EYE_TOP_2] = 258
ros4hri_to_mediapipe[FacialLandmarks.LEFT_EYE_INSIDE] = 463
ros4hri_to_mediapipe[FacialLandmarks.LEFT_EYE_BOTTOM_1] = 254
ros4hri_to_mediapipe[FacialLandmarks.LEFT_EYE_BOTTOM_2] = 252
ros4hri_to_mediapipe[FacialLandmarks.SELLION] = 6
ros4hri_to_mediapipe[FacialLandmarks.NOSE_1] = 197
ros4hri_to_mediapipe[FacialLandmarks.NOSE_2] = 4
ros4hri_to_mediapipe[FacialLandmarks.NOSE] = 1
ros4hri_to_mediapipe[FacialLandmarks.NOSTRIL_1] = 242
ros4hri_to_mediapipe[FacialLandmarks.NOSTRIL_2] = 141
ros4hri_to_mediapipe[FacialLandmarks.NOSTRIL_3] = 94
ros4hri_to_mediapipe[FacialLandmarks.NOSTRIL_4] = 370
ros4hri_to_mediapipe[FacialLandmarks.NOSTRIL_5] = 462
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_OUTER_RIGHT] = 61
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_OUTER_TOP_1] = 40
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_OUTER_TOP_2] = 37
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_OUTER_TOP_3] = 0
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_OUTER_TOP_4] = 267
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_OUTER_TOP_5] = 270
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_OUTER_LEFT] = 291
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_OUTER_BOTTOM_1] = 321
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_OUTER_BOTTOM_2] = 314
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_OUTER_BOTTOM_3] = 17
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_OUTER_BOTTOM_4] = 84
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_OUTER_BOTTOM_5] = 91
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_INNER_RIGHT] = 62
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_INNER_TOP_1] = 41
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_INNER_TOP_2] = 12
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_INNER_TOP_3] = 271
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_INNER_LEFT] = 292
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_INNER_BOTTOM_1] = 403
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_INNER_BOTTOM_2] = 15
ros4hri_to_mediapipe[FacialLandmarks.MOUTH_INNER_BOTTOM_3] = 179


def normalized_to_pixel_coordinates(
    normalized_x: float,
    normalized_y: float,
    image_width: int,
    image_height: int,
):

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


class Face(Node):

    last_id = 0

    def __init__(self, deterministic_id=False):
        super().__init__('hri_face_detect')

        self.declare_parameter('/humans/faces/width', 128)
        self.declare_parameter('/humans/faces/height', 128)

        cropped_face_width = self.get_parameter('/humans/faces/width').get_parameter_value()
        cropped_face_height = self.get_parameter('/humans/faces/height').get_parameter_value()
        
        if cropped_face_height != cropped_face_width:
            self.get_logger.error(
                "The /humans/faces/width and /humans/faces/height must be equal. Continuing with width = height = %spx"
                % cropped_face_width
            )
        cropped_face_height = cropped_face_width

        # generate unique ID
        if deterministic_id:
            self.id = "f%05d" % Face.last_id
            Face.last_id = (Face.last_id + 1) % 10000
        else:
            self.id = "".join(
                random.choices("abcdefghijklmnopqrstuvwxyz", k=5)
            )  # for a 5 char long ID

        self.nb_frames_visible = 1

        self.bb = None  # a RegionOfInterest instance
        self.nose_tip = ()
        self.mouth_center = ()
        self.right_eye = ()
        self.left_eye = ()
        self.right_ear_tragion = ()
        self.left_ear_tragion = ()

        self.facial_landmarks_msg = None

        self.head_transform = None
        self.gaze_transform = None

        self.roi_pub = None
        self.cropped_pub = None
        self.aligned_pub = None
        self.landmarks_pub = None

        # True once the publishers are initialised
        self.ready = False

    def initialise_publishers(self):
        """Initialises all the publishers for this face.

        Not done in the constructor as we typically wait for a few frames
        before publishing anything (to avoid creating too many spurious faces
        due to detection noise).
        """

        # already initialised?
        if self.ready:
            return

        self.roi_pub = self.create_publisher(
            RegionOfInterest,
            "/humans/faces/%s/roi" % self.id,            
            1,
        )

        self.cropped_pub = self.create_publisher(
            Image,
            "/humans/faces/%s/cropped" % self.id,            
            1,
        )

        self.aligned_pub = self.create_publisher(
            Image,
            "/humans/faces/%s/aligned" % self.id,            
            1,
        )

        self.landmarks_pub = self.create_publisher(
            FacialLandmarks,
            "/humans/faces/%s/landmarks" % self.id,            
            1,
        )
        self.get_logger().info('New face: %s' % self)


        self.ready = True

    def publish(self):

        if not self.ready:
            self.get_logger().error(
                "Trying to publish face information but publishers have not been created yet!"
            )
            return

        self.roi_pub.publish(self.bb)
        self.landmarks_pub.publish(self.facial_landmarks_msg)

    def publish_images(self, src_image):

        if not self.ready:
            self.get_logger().error(
                "Trying to publish face images but publishers have not been created yet!"
            )
            return

        self.publish_cropped_face(src_image)
        self.publish_aligned_face(src_image)

    def publish_cropped_face(self, src_image):

        # no-one interested in the face image? skip it!
        
        if self.cropped_pub.get_subscription_count() == 0:
            return
   
        roi = src_image[
            self.bb.y_offset : self.bb.y_offset + self.bb.height,
            self.bb.x_offset : self.bb.x_offset + self.bb.width,
        ]

        sx = cropped_face_width * 1.0 / self.bb.width
        sy = cropped_face_height * 1.0 / self.bb.height

        scale = min(sx, sy)

        scaled = cv2.resize(roi, None, fx=scale, fy=scale)
        scaled_h, scaled_w = scaled.shape[:2]

        output = np.zeros((cropped_face_width, cropped_face_height, 3), np.uint8)

        x_offset = int((cropped_face_width - scaled_w) / 2)
        y_offset = int((cropped_face_height - scaled_h) / 2)

        output[y_offset : y_offset + scaled_h, x_offset : x_offset + scaled_w] = scaled

        self.cropped_pub.publish(CvBridge().cv2_to_imgmsg(output, encoding="bgr8"))

    def publish_aligned_face(self, src_image):
        """Aligns given face in img based on left and right eye coordinates.

        This function is adapted from MIT-licensed DeepFace.
        Author: serengil
        Original source: https://github.com/serengil/deepface/blob/f07f278/deepface/detectors/FaceDetector.py#L68
        """

        # no-one interested in the face image? skip it!
        if self.aligned_pub.get_subscription_count() == 0:
            return

        img_height, img_width, _ = src_image.shape
        x, y, w, h = self.bb.x_offset, self.bb.y_offset, self.bb.width, self.bb.height

        # expand the ROI a little to ensure the rotation does not introduce black zones
        xm1 = max(0, x - w // 2)
        xm2 = min(x + w + w // 2, img_width)
        ym1 = max(0, y - h // 2)
        ym2 = min(y + h + h // 2, img_height)
        preroi = src_image[ym1:ym2, xm1:xm2]

        left_eye_x, left_eye_y = self.left_eye
        right_eye_x, right_eye_y = self.right_eye

        # -----------------------
        # find rotation direction

        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1  # rotate same direction to clock
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1  # rotate inverse direction of clock

        # find length of triangle edges
        a = math.dist(self.left_eye, point_3rd)
        b = math.dist(self.right_eye, point_3rd)
        c = math.dist(self.right_eye, self.left_eye)

        # apply cosine rule
        if b != 0 and c != 0:

            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            angle = np.arccos(cos_a)  # angle in radian
            angle = (angle * 180) / math.pi  # radian to degree

            # rotate base image

            if direction == -1:
                angle = 90 - angle

            img = PILImage.fromarray(preroi)  # convert to a PIL image to rotate it
            preroi = np.array(img.rotate(-direction * angle, PILImage.BILINEAR))

        roi = preroi[y - ym1 : y - ym1 + h, x - xm1 : x - xm1 + w]

        sx = cropped_face_width * 1.0 / self.bb.width
        sy = cropped_face_height * 1.0 / self.bb.height

        scale = min(sx, sy)

        scaled = cv2.resize(roi, None, fx=scale, fy=scale)
        scaled_h, scaled_w = scaled.shape[:2]

        output = np.zeros((cropped_face_width, cropped_face_height, 3), np.uint8)

        x_offset = int((cropped_face_width - scaled_w) / 2)
        y_offset = int((cropped_face_height - scaled_h) / 2)

        output[y_offset : y_offset + scaled_h, x_offset : x_offset + scaled_w] = scaled

        self.aligned_pub.publish(CvBridge().cv2_to_imgmsg(output, encoding="bgr8"))

    def compute_6d_pose(self, K, image, camera_optical_frame):

        img_height, img_width, _ = image.shape

        points_2D = np.array(
            [
                normalized_to_pixel_coordinates(
                    self.nose_tip[0],
                    self.nose_tip[1],
                    img_width,
                    img_height,
                ),
                normalized_to_pixel_coordinates(
                    self.right_eye[0],
                    self.right_eye[1],
                    img_width,
                    img_height,
                ),
                normalized_to_pixel_coordinates(
                    self.left_eye[0],
                    self.left_eye[1],
                    img_width,
                    img_height,
                ),
                normalized_to_pixel_coordinates(
                    self.mouth_center[0],
                    self.mouth_center[1],
                    img_width,
                    img_height,
                ),
                normalized_to_pixel_coordinates(
                    self.right_ear_tragion[0],
                    self.right_ear_tragion[1],
                    img_width,
                    img_height,
                ),
                normalized_to_pixel_coordinates(
                    self.left_ear_tragion[0],
                    self.left_ear_tragion[1],
                    img_width,
                    img_height,
                ),
            ],
            dtype="double",
        )

        trans_vec, angles = face_pose_estimation(points_2D, K)

        # calculating angle
        self.head_transform  = TransformStamped()
        self.head_transform.header.stamp = self.get_clock().now().to_msg()
        self.head_transform.header.frame_id = camera_optical_frame
        self.head_transform.child_frame_id = "face_" + self.id
        self.head_transform.transform.translation.x = trans_vec[0] / 1000
        self.head_transform.transform.translation.y = trans_vec[1] / 1000
        self.head_transform.transform.translation.z = trans_vec[2] / 1000 
        q = quaternion_from_euler(
                        angles[0] / 180 * np.pi,
                        angles[1] / 180 * np.pi,
                        angles[2] / 180 * np.pi,
                    )
        self.head_transform.transform.rotation.x = q[0]
        self.head_transform.transform.rotation.y = q[1]
        self.head_transform.transform.rotation.z = q[2]
        self.head_transform.transform.rotation.w = q[3]
        

        self.gaze_transform = TransformStamped()
        self.gaze_transform.header.stamp = self.get_clock().now().to_msg()
        self.gaze_transform.header.frame_id = "face_" + self.id
        self.gaze_transform.child_frame_id = "gaze_" + self.id
        self.gaze_transform.transform.translation.x = 0.0
        self.gaze_transform.transform.translation.y = 0.0
        self.gaze_transform.transform.translation.z = 0.0
        q = quaternion_from_euler(
                        -np.pi/2,
                        0,
                        -np.pi/2,
                    )
        self.gaze_transform.transform.rotation.x = q[0]
        self.gaze_transform.transform.rotation.y = q[1]
        self.gaze_transform.transform.rotation.z = q[2]
        self.gaze_transform.transform.rotation.w = q[3]




    def delete(self):

        if not self.ready:
            return

        self.get_logger().info(
            "Face [%s] lost. It remained visible for %s frames"
            % (self, self.nb_frames_visible)
        )
        ##TODO check destroy method
        self.roi_pub.destroy()
        self.cropped_pub.destroy()
        self.aligned_pub.destroy()
        self.landmarks_pub.destroy()

        self.ready = False

    def __repr__(self):
        return self.id


class FaceDetector:
    def __init__(self, face_mesh=True, max_num_faces=10):

        self.face_mesh = face_mesh

        if not face_mesh:
            self.detector = mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )
        else:
            self.detector = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )

    def make_facial_landmarks_msg(self, detection, img_height, img_width):

        # PoI = Points of Interest
        poi = [
            NormalizedPointOfInterest2D(
                int(detection.landmark[ros4hri_to_mediapipe[idx]].x * img_width),
                int(detection.landmark[ros4hri_to_mediapipe[idx]].y * img_height),
                1,
            )
            for idx in range(68)
        ]

        # the last two facial landmarks represent the pupils
        # and Mediapipe does not provide pupils estimation.
        poi.append(NormalizedPointOfInterest2D(0, 0, 1))  # RIGHT_PUPIL
        poi.append(NormalizedPointOfInterest2D(0, 0, 1))  # LEFT_PUPIL

        landmarks = FacialLandmarks(landmarks=poi, height=img_height, width=img_width)

        return landmarks

    def detect(self, img, img_width, img_height):
        """img is expected as RGB"""
        img_rows, img_cols, _ = img.shape

        detections = self.detector.process(img).detections
        if not detections:
            return []

        results = []

        for detection in detections:

            face = {}

            face["bb"] = self.get_boundingbox(detection, img_cols, img_rows)

            landmarks = FacialLandmarks()
            points_of_interest = NormalizedPointOfInterest2D()
            points_of_interest.x = 0.0                                          
            points_of_interest.y = 0.0
            points_of_interest.c = 1.0
            landmarks.landmarks = [points_of_interest  for _ in range(70)]
            landmarks.height = img_height
            landmarks.width = img_width


            nose_tip = mp_face_detection.get_key_point(
                detection, mp_face_detection.FaceKeyPoint.NOSE_TIP
            )
            face["nose_tip"] = (nose_tip.x, nose_tip.y)

            nose_points = NormalizedPointOfInterest2D()
            nose_points.x = min(nose_tip.x * img_width, img_width)
            nose_points.y = min(nose_tip.y * img_height, img_height)
            nose_points.c = 1.0

            landmarks.landmarks[FacialLandmarks.NOSE] = nose_points

            right_eye = mp_face_detection.get_key_point(
                detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE
            )
            face["right_eye"] = (right_eye.x, right_eye.y)
            right_eye_points = NormalizedPointOfInterest2D()
            right_eye_points.x = min(right_eye.x * img_width, img_width)
            right_eye_points.y = min(right_eye.y * img_height, img_height)
            right_eye_points.c = 1.0
            landmarks.landmarks[
                FacialLandmarks.RIGHT_PUPIL
            ] = right_eye_points

            left_eye = mp_face_detection.get_key_point(
                detection, mp_face_detection.FaceKeyPoint.LEFT_EYE
            )
            face["left_eye"] = (left_eye.x, left_eye.y)

            left_eye_points = NormalizedPointOfInterest2D()
            left_eye_points.x = min(left_eye.x * img_width, img_width)
            left_eye_points.y = min(left_eye.y * img_height, img_height)
            left_eye_points.c = 1.0
            
            landmarks.landmarks[
                FacialLandmarks.LEFT_PUPIL
            ] = left_eye_points

            mouth_center = mp_face_detection.get_key_point(
                detection, mp_face_detection.FaceKeyPoint.MOUTH_CENTER
            )
            face["mouth_center"] = (mouth_center.x, mouth_center.y)

            mouth_points = NormalizedPointOfInterest2D()
            mouth_points.x = min(mouth_center.x * img_width, img_width)
            mouth_points.y = min(mouth_center.y * img_height, img_height)
            mouth_points.c = 1.0

            landmarks.landmarks[
                FacialLandmarks.MOUTH_INNER_TOP_2
            ] = mouth_points 

            right_ear_tragion = mp_face_detection.get_key_point(
                detection, mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION
            )
            face["right_ear_tragion"] = (right_ear_tragion.x, right_ear_tragion.y)

            right_ear_points = NormalizedPointOfInterest2D()
            right_ear_points.x = min(right_ear_tragion.x * img_width, img_width)
            right_eye_points.y = min(right_ear_tragion.y * img_height, img_height)
            right_eye_points.c = 1.0

            landmarks.landmarks[
                FacialLandmarks.RIGHT_EAR
            ] = right_ear_points

            left_ear_tragion = mp_face_detection.get_key_point(
                detection, mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION
            )
            face["left_ear_tragion"] = (left_ear_tragion.x, left_ear_tragion.y)

            left_ear_points = NormalizedPointOfInterest2D()
            left_ear_points.x = min(left_ear_tragion.x * img_width, img_width)
            left_ear_points.y = min(left_ear_tragion.y * img_height, img_height)
            left_ear_points.c = 1.0

            landmarks.landmarks[FacialLandmarks.LEFT_EAR] = left_ear_points
            face["facial_landmarks_msg"] = landmarks

            results.append(face)

        return results

    def detect_face_mesh(self, img, img_height, img_width):
        """img is expected as RGB"""
        img_rows, img_cols, _ = img.shape

        detections = self.detector.process(img)
        if not detections or not detections.multi_face_landmarks:
            return []

        results = []

        for detection in detections.multi_face_landmarks:

            face = {}

            x_min = 1
            y_min = 1
            x_max = 0
            y_max = 0

            for idx, landmark in enumerate(detection.landmark):
                if landmark.x < x_min:
                    x_min = landmark.x
                if landmark.y < y_min:
                    y_min = landmark.y
                if landmark.x > x_max:
                    x_max = landmark.x
                if landmark.y > y_max:
                    y_max = landmark.y
                if idx == 1:
                    face["nose_tip"] = (landmark.x, landmark.y)
                if idx == 13:
                    face["mouth_center"] = (landmark.x, landmark.y)
                if idx == 159:
                    face["right_eye"] = (landmark.x, landmark.y)
                if idx == 234:
                    face["right_ear_tragion"] = (landmark.x, landmark.y)
                if idx == 386:
                    face["left_eye"] = (landmark.x, landmark.y)
                if idx == 454:
                    face["left_ear_tragion"] = (landmark.x, landmark.y)

            x, y = normalized_to_pixel_coordinates(x_min, y_min, img_cols, img_rows)
            w, h = normalized_to_pixel_coordinates(
                x_max - x_min, y_max - y_min, img_cols, img_rows
            )

            face["bb"] = (x, y, w, h)

            face["facial_landmarks_msg"] = self.make_facial_landmarks_msg(
                detection, img_height, img_width
            )

            results.append(face)

        return results

    def get_boundingbox(self, detection, image_cols, image_rows):
        """
        Based on https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/drawing_utils.py
        """

        bb = detection.location_data.relative_bounding_box
        x, y = normalized_to_pixel_coordinates(bb.xmin, bb.ymin, image_cols, image_rows)
        w, h = normalized_to_pixel_coordinates(
            bb.width, bb.height, image_cols, image_rows
        )

        return (x, y, w, h)

    def __str__(self):
        return "Google mediapipe face detector"


class RosFaceDetector(Node):
    def __init__(self, debug=False, face_mesh=True, max_num_faces=4):
        super().__init__('hri_face_detect')


        self.declare_parameter('~debug', False)
        self.declare_parameter('~face_mesh', False)
        self.declare_parameter("~max_num_faces", 4)

        max_num_faces = self.get_parameter('~max_num_faces').get_parameter_value().integer_value

        self.is_shutting_down = False
        self.debug = debug = self.get_parameter('~debug').get_parameter_value().bool_value
        self.face_mesh = face_mesh = self.get_parameter('~face_mesh').get_parameter_value().bool_value
 
        semaphore_pub = self.create_publisher(
            Empty,
            "/hri_face_detect/ready",
             1
            )
        self.faces_pub = self.create_publisher(
            IdsList,
            "/humans/faces/tracked",             
             1)

        self.facedetector = FaceDetector(face_mesh, max_num_faces)
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
            "Ready. Waiting for images to be published on %s." % self.image_sub.topic
        )
        semaphore_pub.publish(Empty())

        # last-used face id
        self.last_id = 1

        # list (map ID -> Face) of Face instances, corresponding to the currently tracked faces
        self.knownFaces = {}

        self.tb = TransformBroadcaster(self)

    def distance_rois(self, bb1, bb2):
        x1, y1 = bb1.x_offset + bb1.width / 2, bb1.y_offset + bb1.height / 2
        x2, y2 = bb2.x_offset + bb2.width / 2, bb2.y_offset + bb2.height / 2

        return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

    def find_previous_match(self, bb):
        for _, face in self.knownFaces.items():
            prev_bb = face.bb
            if (
                self.distance_rois(prev_bb, bb) < MAX_ROIS_DISTANCE * MAX_ROIS_DISTANCE
                and 1 / MAX_SCALING_ROIS < prev_bb.width / bb.width < MAX_SCALING_ROIS
                and 1 / MAX_SCALING_ROIS < prev_bb.height / bb.height < MAX_SCALING_ROIS
            ):
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
        self.get_logger().info("Got into the callback, is_shut: %s"% str(self.is_shutting_down))
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
            bb.y_offset =  max(0, y)
            bb.width =  min(img_height - y, h)
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
                face.nose_tip = detection["nose_tip"]
                face.mouth_center = detection["mouth_center"]
                face.right_eye = detection["right_eye"]
                face.left_eye = detection["left_eye"]
                face.right_ear_tragion = detection["right_ear_tragion"]
                face.left_ear_tragion = detection["left_ear_tragion"]
                face.facial_landmarks_msg = detection["facial_landmarks_msg"]

            else:
                face = Face()
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
                        face.compute_6d_pose(self.K, image, camera_optical_frame)

                        self.tb.sendTransform(face.head_transform)
                        self.tb.sendTransform(face.gaze_transform)
        faces_msg = IdsList()
        faces_msg.header = rgb_msg.header
        faces_msg.ids = [face.id for face in self.knownFaces.values() if face.ready]
        self.faces_pub.publish(faces_msg)
        faces_msg = None

        if self.debug:
            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            self.get_logger().info("%s faces detected" % len(self.knownFaces))
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

            cv2.imshow("MediaPipe Face Detection", image)
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
        self.get_logger().info("Stopped publishing faces.")
        rate= self.create_rate(10)
        rate.sleep()
        rate.destroy()
        
    
def main(args=None):
    rclpy.init(args=args)
    # rospy.init_node("hri_face_detect")
    detector = RosFaceDetector()
    # detector = RosFaceDetector(debug, face_mesh, max_num_faces)
    #rclpy.context.Context().on_shutdown(detector.close())
    rclpy.spin(detector)

if __name__ == "__main__":

    main()
    
