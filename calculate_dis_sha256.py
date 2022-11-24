import helpers.sha256 as sha256
import helpers.bounding_box as bounding_box
import cv2, os

img1 = cv2.imread("test_file/tuyen.jpg")
feature1 = sha256.get_face_hash(img1)

result = []
for file in os.listdir("test_file/me"):
    img2 = cv2.imread("test_file/me/" + file)
    feature2 = sha256.get_face_hash(img2)

    distance = sha256.hamming(feature1, feature2)
    result.append(distance)
    print(file, distance)

print(sum(result) / len(result))