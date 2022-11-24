import helpers.sha256
import cv2

img1 = cv2.imread("test_file/itsme.jpg")
img2 = cv2.imread("test_file/me/me.jpg")
print(helpers.sha256.feature_matching(img1, img2))