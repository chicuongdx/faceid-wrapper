from helpers.facenet import FaceNet
from helpers.bounding_box import BoundingBox
import helpers.sha256 as sha256
import cv2

yolov5_url = 'models/best.pt'
facenet_url = 'models/facenet.pt'

fn_model = FaceNet('cpu')

img1 = cv2.imread("test_file/itsme.jpg")
img2 = cv2.imread("test_file/hpd.jpg")

bd_model = BoundingBox(yolov5_url)

face1 = bd_model.crop_boudingbox(img1)
face2 = bd_model.crop_boudingbox(img2)

feature1 = fn_model.get_embedding(face1)
feature2 = fn_model.get_embedding(face2)

print(sha256.compare_embedding(feature1, feature2))