import math
import numpy as np
import cv2


# face key points
P3D_RIGHT_EYE = (-20.0, -65.5, -5.0)
P3D_LEFT_EYE = (-20.0, 65.5, -5.0)
P3D_RIGHT_EAR = (-100.0, -77.5, -6.0)
P3D_LEFT_EAR = (-100.0, 77.5, -6.0)
P3D_NOSE = (21.0, 0.0, -48.0)
P3D_STOMION = (10.0, 0.0, -75.0)


def normalized_to_pixel_coordinates(
    normalized_x: float,
    normalized_y: float,
    image_width: int,
    image_height: int,
):

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


points_3D = np.array([P3D_NOSE,
                      P3D_RIGHT_EYE,
                      P3D_LEFT_EYE,
                      P3D_STOMION,
                      P3D_RIGHT_EAR,
                      P3D_LEFT_EAR])


def face_pose_estimation(points_2D, K):
    success, rot_vec, trans_vec = cv2.solvePnP(
        points_3D,
        points_2D,
        K,
        None,
        tvec=np.array([0.0, 0.0, 1000.0]),
        useExtrinsicGuess=True,
        flags=4,
    )
    rmat, jac = cv2.Rodrigues(rot_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    if np.isnan(trans_vec).any():
        trans_vec[:] = 0.0
    return trans_vec, angles


def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    q = np.empty((4, ))
    q[0] = cj * sc - sj * cs
    q[1] = cj * ss + sj * cc
    q[2] = cj * cs - sj * sc
    q[3] = cj * cc + sj * ss

    return q
