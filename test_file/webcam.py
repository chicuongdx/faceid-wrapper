import pafy
import sys
import cv2, time
import torch
from threading import Thread
import phu


modelpath = '.\helmet_yolov5s.pt'
try:
	modelpath = sys.argv[1]
except:
	pass
model = torch.hub.load( './yolov5', 'custom', path=modelpath, source='local' )


if __name__ == '__main__':
	maincam = phu.Camera()
	maindetective = phu.Detective(maincam, model )
	while maincam.isOpened():
		maindetective.run()


		# ret, img = cap.read()
		# img = phu.detect( img ,model  )
		# cv2.imshow("asfdghnkxcvblhj", img )
		# if cv2.waitKey(1) == ord('q'):
		# 	break


