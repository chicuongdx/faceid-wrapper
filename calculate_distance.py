from helpers.facenet import FaceNet
from helpers.bounding_box import BoundingBox
import helpers.sha256 as sha256
import cv2
import os

yolov5_url = 'models/yolov5l.pt'
fn_model = FaceNet('cpu')
bd_model = BoundingBox(yolov5_url)

# Get all images in the folder "test_file/me"
# and calculate the distance between them
# and the image "test_file/me.jpg"
# img1 = cv2.imread("test_file/tuyen.jpg")
# face1 = bd_model.crop_boudingbox(img1)

img1 = cv2.imread("test_file/tuyen.jpg")
face1 = bd_model.crop_boudingbox(img1)

embedding1 = fn_model.get_embedding(face1)

result = []
for file in os.listdir("test_file/me"):
    img2 = cv2.imread("test_file/me/" + file)
    face2 = bd_model.crop_boudingbox(img2)
    embedding2 = fn_model.get_embedding(face2)
    #dis = fn_model.get_distance(embedding1, embedding2)

    #dis = sha256.compare_embedding(embedding1, embedding2)
    dis = fn_model.get_distance(embedding1, embedding2)
    result.append(dis)
    print(file, dis)

print(sum(result) / len(result))
print(max(result))