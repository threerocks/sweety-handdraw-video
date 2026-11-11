"""Face landmarks of one portrait image, as JSON on stdout (pixel coordinates of that image).

Runs in its own environment (MediaPipe needs numpy < 2, the engines use numpy 2):
    ~/.local/share/sweety-handdraw-video/venv-face/bin/python landmarks.py IMAGE

make.py calls it; nothing else should need to. Output keys (all [x, y] pixel pairs):
    ok, size, oval (36 points round the face), eyes (two, image left first: contour, iris centre, iris radius),
    brows (two), lips_outer, lips_inner, nose (tip, bridge, wings, base), score
"""
import json
import sys

import numpy as np
from PIL import Image

OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150,
        136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]       # image left
EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]   # image right
IRIS_R, IRIS_L = [468, 469, 470, 471, 472], [473, 474, 475, 476, 477]
BROW_R = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_L = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
LIPS_OUT = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_IN = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
NOSE = {'tip': 1, 'bridge': 6, 'wing_r': 129, 'wing_l': 358, 'base': 2}


def detect(path):
    import mediapipe as mp
    img = np.asarray(Image.open(path).convert('RGB'))
    h, w = img.shape[:2]
    out = {'ok': False, 'size': [w, h]}
    # a face can be small in a half-length portrait: try the whole picture, then the upper part enlarged
    for crop in (None, (0, 0, w, int(h * .6))):
        sub = img if crop is None else img[crop[1]:crop[3], crop[0]:crop[2]]
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                             min_detection_confidence=.3) as fm:
            res = fm.process(np.ascontiguousarray(sub))
        if res.multi_face_landmarks:
            sh, sw = sub.shape[:2]
            ox, oy = (0, 0) if crop is None else crop[:2]
            P = np.array([[p.x * sw + ox, p.y * sh + oy] for p in res.multi_face_landmarks[0].landmark])
            break
    else:
        return out

    def pts(ix):
        return P[ix].round(1).tolist()

    eyes = []
    for contour, iris in ((EYE_R, IRIS_R), (EYE_L, IRIS_L)):
        c = P[iris[0]]
        r = float(np.mean(np.linalg.norm(P[iris[1:]] - c, axis=1)))
        eyes.append({'contour': pts(contour), 'iris': c.round(1).tolist(), 'iris_r': round(r, 1)})
    if eyes[0]['iris'][0] > eyes[1]['iris'][0]:
        eyes = eyes[::-1]
    brows = [pts(BROW_R), pts(BROW_L)]
    if brows[0][0][0] > brows[1][0][0]:
        brows = brows[::-1]
    out.update(ok=True, oval=pts(OVAL), eyes=eyes, brows=brows, lips_outer=pts(LIPS_OUT), lips_inner=pts(LIPS_IN),
               nose={k: P[i].round(1).tolist() for k, i in NOSE.items()})
    return out


if __name__ == '__main__':
    print(json.dumps(detect(sys.argv[1])))
