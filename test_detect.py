from helpers.bounding_box import BoundingBox
import cv2

face_detector = BoundingBox("models/best.pt")

def get_face(img):
    face = face_detector.crop_boudingbox(img)
    return face

def bounding_box(img):
    face = face_detector.draw_boudingbox(img)
    return face

img = cv2.imread("test_file/testdd.png")
face = bounding_box(img)
cv2.imshow("face", face)
cv2.waitKey(0)