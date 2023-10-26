import mediapipe as mp
from .utils import normalized_to_pixel_coordinates
from hri_msgs.msg import FacialLandmarks, NormalizedPointOfInterest2D

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh


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
        # todo(juandpenan) change to list comprehension
        poi = []
        for idx in range(68):
            msg = NormalizedPointOfInterest2D()
            msg.x = detection.landmark[ros4hri_to_mediapipe[idx]].x * img_width
            msg.y = detection.landmark[ros4hri_to_mediapipe[idx]
                                       ].y * img_height
            msg.c = 1.0
            poi.append(msg)

        # poi = [
        #     NormalizedPointOfInterest2D(
        #         int(detection.landmark[ros4hri_to_mediapipe[idx]].x * img_width),
        #         int(detection.landmark[ros4hri_to_mediapipe[idx]].y * img_height),
        #         1,
        #     )
        #     for idx in range(68)
        # ]

        # the last two facial landmarks represent the pupils
        # and Mediapipe does not provide pupils estimation.
        r_pupil = NormalizedPointOfInterest2D()
        r_pupil.x = 0.0
        r_pupil.y = 0.0
        r_pupil.c = 1.0

        l_pupil = NormalizedPointOfInterest2D()
        l_pupil.x = 0.0
        l_pupil.y = 0.0
        l_pupil.c = 1.0
        poi.append(r_pupil)  # RIGHT_PUPIL
        poi.append(l_pupil)  # LEFT_PUPIL

        landmarks = FacialLandmarks(
            landmarks=poi,
            height=img_height,
            width=img_width)

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
            landmarks.landmarks = [points_of_interest for _ in range(70)]
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
            face["right_ear_tragion"] = (
                right_ear_tragion.x, right_ear_tragion.y)

            right_ear_points = NormalizedPointOfInterest2D()
            right_ear_points.x = min(
                right_ear_tragion.x * img_width, img_width)
            right_eye_points.y = min(
                right_ear_tragion.y * img_height, img_height)
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
            left_ear_points.y = min(
                left_ear_tragion.y * img_height, img_height)
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

            x, y = normalized_to_pixel_coordinates(
                x_min, y_min, img_cols, img_rows)
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
        x, y = normalized_to_pixel_coordinates(
            bb.xmin, bb.ymin, image_cols, image_rows)
        w, h = normalized_to_pixel_coordinates(
            bb.width, bb.height, image_cols, image_rows
        )

        return (x, y, w, h)

    def __str__(self):
        return "Google mediapipe face detector"
