import argparse
import math
import cv2
from PIL import Image as PILImage
import numpy as np
import pathlib
import random

import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# default size in pixels for the faces
cropped_face_width = 150
cropped_face_height = 150


def normalized_to_pixel_coordinates(
    normalized_x: float,
    normalized_y: float,
    image_width: int,
    image_height: int,
):

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


nb_frames = 0
nb_frames_recorded = 0

# only record one frame every `subsample` frames
subsample = 10


class Face:
    def __init__(self):

        self.bb = None  # a RegionOfInterest instance

        self.left_eye = (0, 0)
        self.right_eye = (0, 0)

    def save_aligned_face(self, path, src_image):
        """Aligns given face in img based on left and right eye coordinates.

        This function is adapted from MIT-licensed DeepFace.
        Author: serengil
        Original source: https://github.com/serengil/deepface/blob/f07f278/deepface/detectors/FaceDetector.py#L68
        """
        global nb_frames_recorded, nb_frames

        nb_frames += 1

        if nb_frames % subsample != 0:
            return

        img_height, img_width, _ = src_image.shape
        x, y, h, w = self.bb

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

        sx = cropped_face_width * 1.0 / w
        sy = cropped_face_height * 1.0 / h

        scale = min(sx, sy)

        scaled = cv2.resize(roi, None, fx=scale, fy=scale)
        scaled_h, scaled_w = scaled.shape[:2]

        output = np.zeros((cropped_face_width, cropped_face_height, 3), np.uint8)

        x_offset = int((cropped_face_width - scaled_w) / 2)
        y_offset = int((cropped_face_height - scaled_h) / 2)

        output[y_offset : y_offset + scaled_h, x_offset : x_offset + scaled_w] = scaled

        cv2.imwrite(str(path / ("%s.jpg" % self.random_name())), output)
        nb_frames_recorded += 1

    def random_name(self):
        return "".join(random.sample("abcdefghijklmnopqrstuvwxyz", 5))


class FaceDetector:
    def __init__(self, max_num_faces=1):

        self.detector = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def detect(self, img):
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
                if idx == 159:
                    face["right_eye"] = (landmark.x, landmark.y)
                if idx == 386:
                    face["left_eye"] = (landmark.x, landmark.y)

            x, y = normalized_to_pixel_coordinates(x_min, y_min, img_cols, img_rows)
            w, h = normalized_to_pixel_coordinates(
                x_max - x_min, y_max - y_min, img_cols, img_rows
            )

            face["bb"] = (x, y, w, h)

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


class FaceRecorder:
    def __init__(self, person_name, path, max_num_faces=1):
        global nb_frames_recorded

        self.facedetector = FaceDetector()

        self.person_name = person_name

        self.path = pathlib.Path(path) / person_name
        self.path.mkdir(parents=True, exist_ok=True)

        print("Dataset will be stored in %s" % self.path)

    def run(self):

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        is_recording = False

        face = None
        while True:
            ret, image = cap.read()

            debug_image = image.copy()

            img_height, img_width, _ = image.shape

            detections = self.facedetector.detect(image)

            for detection in detections:
                x, y, w, h = detection["bb"]
                bb = (
                    max(0, x),
                    max(0, y),
                    min(img_height - y, h),
                    min(img_width - x, w),
                )

                face = Face()

                # update the face with its current position and landmarks
                face.bb = bb
                face.right_eye = detection["right_eye"]
                face.left_eye = detection["left_eye"]

                # Draw the face detection annotations on the image.
                image.flags.writeable = True

                x, y, h, w = face.bb
                cv2.rectangle(
                    debug_image,
                    (x, y),
                    (x + w, y + h),
                    (190, 100, 30),
                    2,
                )

            cv2.putText(
                debug_image,
                self.person_name,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            if face:
                cv2.putText(
                    debug_image,
                    "# frames recorded: %d. Press Space to toggle recording"
                    % nb_frames_recorded,
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            if is_recording:
                cv2.circle(
                    debug_image,
                    (50, 100),
                    40,
                    (0, 0, 255),
                    -1,
                )

                face.save_aligned_face(self.path, image)

            cv2.imshow("Face Detection", debug_image)
            key = cv2.waitKey(5)

            if key == 32:
                is_recording = not is_recording
            elif key == 27:
                break

def main():

    parser = argparse.ArgumentParser(
        description="Record faces for pre-training face recognition"
    )
    parser.add_argument(
        "-n", "--name", type=str, nargs="?", help="name of the person to record"
    )

    parser.add_argument(
        "path",
        type=str,
        nargs="?",
        default="/tmp/face_dataset",
        help="path where the face images will be stored",
    )

    args = parser.parse_args()

    cv2.namedWindow("Face Detection", cv2.WINDOW_FULLSCREEN & cv2.WINDOW_KEEPRATIO)

    if not args.name:
        person_name = input("Enter your name:")
    else:
        person_name = args.name

    detector = FaceRecorder(person_name, args.path)

    detector.run()

if __name__ == "__main__":

    main()

