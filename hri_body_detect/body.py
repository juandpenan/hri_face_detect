from .utils import quaternion_from_euler, body_pose_estimation, normalized_to_pixel_coordinates
import random
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge
from PIL import Image as PILImage
import math
from sensor_msgs.msg import Image, RegionOfInterest
import numpy as np
import cv2


class Body():

    last_id = 0
    # todo (juandpenan) set deterministc id as a ros param

    def __init__(self, node, deterministic_id=False, width=300, height=300, id=''):

        self.node = node
        self.cropped_body_width = width
        self.cropped_body_height = height

        if self.cropped_body_height != self.cropped_body_width:
            self.get_logger.error(
                'The /humans/bodies/width and /humans/bodies/height must be equal. Continuing with width = height = %spx' %
                self.cropped_body_width)
            self.cropped_body_height = self.cropped_body_width

        # generate IDs
        if id != '':
            self.id = id
        else:
            if deterministic_id:
                self.id = "f%05d" % Body.last_id
                Body.last_id = (Body.last_id + 1) % 10000
            else:
                self.id = "".join(
                    random.choices("abcdefghijklmnopqrstuvwxyz", k=5)
                )  # for a 5 char long ID

        self.nb_frames_visible = 1

        self.bb = None  # a RegionOfInterest instance
      
        self.head_transform = None
        self.gaze_transform = None

        self.roi_pub = None
        self.cropped_pub = None
        self.aligned_pub = None

        # True once the publishers are initialised
        self.ready = False

    def initialise_publishers(self):
        """Initialises all the publishers for this body.

        Not done in the constructor as we typically wait for a few frames
        before publishing anything (to avoid creating too many spurious bodies
        due to detection noise).
        """

        # already initialised?
        if self.ready:
            return

        self.roi_pub = self.node.create_publisher(
            RegionOfInterest,
            "/humans/bodies/%s/roi" % self.id,
            1,
        )

        self.cropped_pub = self.node.create_publisher(
            Image,
            "/humans/bodies/%s/cropped" % self.id,
            1,
        )

        self.aligned_pub = self.node.create_publisher(
            Image,
            "/humans/bodies/%s/aligned" % self.id,
            1,
        )

        self.node.get_logger().info('New body: %s' % self)
        self.ready = True

    def publish(self):

        if not self.ready:
            self.node.get_logger().error(
                "Trying to publish body information but publishers have not been created yet!"
            )
            return

        self.roi_pub.publish(self.bb)

    def publish_images(self, src_image):

        if not self.ready:
            self.node.get_logger().error(
                "Trying to publish body images but publishers have not been created yet!"
            )
            return

        self.publish_cropped_body(src_image)
        self.publish_aligned_body(src_image)

    def publish_cropped_body(self, src_image):

        # no-one interested in the body image? skip it!

        if self.cropped_pub.get_subscription_count() == 0:
            return

        roi = src_image[
            self.bb.y_offset: self.bb.y_offset + self.bb.height,
            self.bb.x_offset: self.bb.x_offset + self.bb.width,
        ]

        sx = self.cropped_body_width * 1.0 / self.bb.width
        sy = self.cropped_body_height * 1.0 / self.bb.height

        scale = min(sx, sy)

        scaled = cv2.resize(roi, None, fx=scale, fy=scale)
        scaled_h, scaled_w = scaled.shape[:2]

        output = np.zeros(
            (self.cropped_body_width,
             self.cropped_body_height,
             3),
            np.uint8)

        x_offset = int((self.cropped_body_width - scaled_w) / 2)
        y_offset = int((self.cropped_body_height - scaled_h) / 2)

        output[y_offset: y_offset + scaled_h,
               x_offset: x_offset + scaled_w] = scaled

        self.cropped_pub.publish(
            CvBridge().cv2_to_imgmsg(
                output, encoding="bgr8"))

    def publish_aligned_body(self, src_image):
        """Aligns given body in img based on left and right eye coordinates.

        This function is adapted from MIT-licensed DeepFace.
        Author: serengil
        Original source: https://github.com/serengil/deepface/blob/f07f278/deepface/detectors/FaceDetector.py#L68
        """

        # no-one interested in the body image? skip it!
        if self.aligned_pub.get_subscription_count() == 0:
            return

        img_height, img_width, _ = src_image.shape
        x, y, w, h = self.bb.x_offset, self.bb.y_offset, self.bb.width, self.bb.height

        # expand the ROI a little to ensure the rotation does not introduce
        # black zones
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

            # convert to a PIL image to rotate it
            img = PILImage.fromarray(preroi)
            preroi = np.array(
                img.rotate(-direction * angle, PILImage.BILINEAR))

        roi = preroi[y - ym1: y - ym1 + h, x - xm1: x - xm1 + w]

        sx = self.cropped_body_width * 1.0 / self.bb.width
        sy = self.cropped_body_height * 1.0 / self.bb.height

        scale = min(sx, sy)

        scaled = cv2.resize(roi, None, fx=scale, fy=scale)
        scaled_h, scaled_w = scaled.shape[:2]

        output = np.zeros(
            (self.cropped_body_width,
             self.cropped_body_height,
             3),
            np.uint8)

        x_offset = int((self.cropped_body_width - scaled_w) / 2)
        y_offset = int((self.cropped_body_height - scaled_h) / 2)

        output[y_offset: y_offset + scaled_h,
               x_offset: x_offset + scaled_w] = scaled

        self.aligned_pub.publish(
            CvBridge().cv2_to_imgmsg(
                output, encoding="bgr8"))


    def delete(self):

        if not self.ready:
            return

        self.node.get_logger().info(
            "body [%s] lost. It remained visible for %s frames"
            % (self, self.nb_frames_visible)
        )
        self.node.destroy_publisher(self.roi_pub)
        self.node.destroy_publisher(self.cropped_pub)
        self.node.destroy_publisher(self.aligned_pub)

        self.ready = False

    def __repr__(self):
        return self.id
