from hri_msgs.msg import IdsList
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import Header
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from .body import Body
from ultralytics import YOLO
from ultralytics.tracker import BOTSORT, BYTETracker
from ultralytics.tracker.trackers.basetrack import BaseTrack
from ultralytics.yolo.utils import IterableSimpleNamespace, yaml_load
from ultralytics.yolo.utils.checks import check_requirements, check_yaml
import torch
import random
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


class RosBodyDetector(Node):
    def __init__(self,):
        super().__init__('hri_body_detect')

        self.declare_parameter('/humans/bodies/width', 128)
        self.declare_parameter('/humans/bodies/height', 128)
        self.declare_parameter('/humans/bodies/max_num_bodies', 4)

        self.yolo = YOLO('yolov8m.pt')
        self.yolo.fuse()
        self.yolo.to("cuda:0")

        self.max_num_bodies = self.get_parameter(
            '/humans/bodies/max_num_bodies').get_parameter_value().integer_value
        self.body_width = self.get_parameter(
            '/humans/bodies/width').get_parameter_value().integer_value
        self.body_height = self.get_parameter(
            '/humans/bodies/height').get_parameter_value().integer_value
        self.is_shutting_down = False

        self.bodies_pub = self.create_publisher(
            IdsList,
            "/humans/bodies/tracked",
            1)

        self.image_sub = self.create_subscription(
            Image,
            "/head_front_camera/rgb/image_raw",
            self.image_callback,
            1)

        # self.bb_sub = self.create_subscription(
        #     BoundingBoxes,
        #     "/darknet_ros/bounding_boxes",
        #     self.bb_callback,
        #     1)

        self.get_logger().info(
            "Ready. Waiting for images to be published on %s." %
            self.image_sub.topic)
        
        self.create_timer(0.2, self.bb_callback)
        # last-used face id
        self.last_id = 1
        self.last_image = None
        self.bridge = CvBridge()

        # list (map ID -> body) of Body instances, corresponding to the
        # currently tracked bodies
        self.knownBodies = {}
        self.tracker = self.create_tracker("bytetrack.yaml")

    def distance_rois(self, bb1, bb2):
        x1, y1 = bb1.x_offset + bb1.width / 2, bb1.y_offset + bb1.height / 2
        x2, y2 = bb2.x_offset + bb2.width / 2, bb2.y_offset + bb2.height / 2

        return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)

    def find_previous_match(self, bb):
        for _, body in self.knownBodies.items():
            prev_bb = body.bb
            if (self.distance_rois(prev_bb, bb) < MAX_ROIS_DISTANCE *
                MAX_ROIS_DISTANCE and 1 /
                MAX_SCALING_ROIS < prev_bb.width /
                bb.width < MAX_SCALING_ROIS and 1 /
                MAX_SCALING_ROIS < prev_bb.height /
                    bb.height < MAX_SCALING_ROIS):
                return body
        return None

    def image_callback(self, msg):
        if self.is_shutting_down:
            return
        self.last_header = msg.header
        self.last_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    


    def bb_callback(self):

        if self.is_shutting_down or self.last_image is None:
            return
        # copy the list of face ID before iterating over detection, so that we
        # can delete non-existant faces at the end.
        knownIds = list(self.knownBodies.keys())


        img_height, img_width, _ = self.last_image.shape
        # camera_optical_frame = msg.image_header.frame_id

        results = self.yolo.predict(
                source=self.last_image,
                verbose=False,
                stream=False,
                conf=0.1,
                mode='track'
            )

       
        det = results[0].boxes.cpu().numpy()

        if len(det) > 0:
            im0s = self.yolo.predictor.batch[2]
            im0s = im0s if isinstance(im0s, list) else [im0s]

            tracks = self.tracker.update(det, im0s[0])
            if len(tracks) > 0:
                results[0].update(boxes=torch.as_tensor(tracks[:, :-1]))

        results = results[0].cpu()
        currentIds = []

        for b in results.boxes:
            label = self.yolo.names[int(b.cls)]
            score = float(b.conf)

            if score < 0.5 or label != 'person':
                continue

            if b.id is not None:
                track_id = 'b' + str(int(b.id))
            else:
                return

            box = b.xywh[0]

            bb = RegionOfInterest()
            
            bb.x_offset = max(0, int(box[0] - int(box[2]/2)))
            bb.y_offset = max(0, int(box[1] - int(box[3]/2)))

            bb.width = int(box[2])
            bb.height = int(box[3])
            bb.do_rectify = True

            # body = self.find_previous_match(bb)

            if track_id in knownIds:
                print("same body")
                # we re-detect a face!
                body = self.knownBodies[track_id]
                currentIds.append(body.id)
                body.nb_frames_visible += 1

                # if it is a 3nd frame, we create all the required publishers.

                if body.nb_frames_visible == 3:
                    body.initialise_publishers()

                # update the face with its current position and landmarks
                body.bb = bb

            else:
                print("creating new body")
                body = Body(
                    node=self,
                    deterministic_id=True,
                    width=self.body_width,
                    height=self.body_height,
                    id=track_id)
                body.bb = bb
                currentIds.append(body.id)
                self.knownBodies[body.id] = body

        # iterate over faces not seen anymore,
        # and unregister corresponding publishers
        for id in knownIds:
            if id not in currentIds:
                self.knownBodies[id].delete()
                del self.knownBodies[id]
        
        # knownIds = list(self.knownBodies.keys())
        # print(self.knownBodies.items())
        for _, body in self.knownBodies.items():
            if body.ready:
                if not self.is_shutting_down:
                    body.publish()
                if not self.is_shutting_down:
                    body.publish_images(self.last_image)


        bodies_msg = IdsList()
        bodies_msg.header = self.last_header
        bodies_msg.ids = [
            body.id for body in self.knownBodies.values() if body.ready]
        self.bodies_pub.publish(bodies_msg)
        body_msg = None


    def close(self):

        self.get_logger().info("Stopping face publishing...")

        self.is_shutting_down = True

        for _, body in self.knownBodies.items():
            body.delete()

        h = Header()
        h.stamp = self.get_clock().now().to_msg()
        ids_list = IdsList()
        ids_list.header = h
        ids_list.ids = []
        self.bodies_pub.publish(ids_list)
        self.get_logger().info('Stopped publishing bodies.')
        rate = self.create_rate(10)
        rate.sleep()
        rate.destroy()

    def create_tracker(self, tracker_yaml) -> BaseTrack:

        TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        check_requirements("lap")  # for linear_assignment

        tracker = check_yaml(tracker_yaml)
        cfg = IterableSimpleNamespace(**yaml_load(tracker))

        assert cfg.tracker_type in ["bytetrack", "botsort"], \
            f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=1)
        return tracker


def main(args=None):
    rclpy.init(args=args)
    detector = RosBodyDetector()
    rclpy.spin(detector)


if __name__ == "__main__":
    main()
