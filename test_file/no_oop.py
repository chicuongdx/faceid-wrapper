import cv2, time
import numpy as np
from threading import Thread
import os

def same_object( box1, box2, thresh=0.5 ):
	x1,y1,w1,h1 = box1
	x2,y2,w2,h2 = box2

	inner = min( x1+w1-x2 , x2+w2-x1 ) * min( y1+h1-y2, y2+h2-y1 )
	total = w1*h1 + w2*h2 - inner
	fra = inner / total

	return fra >= thresh

def detection_filter( detection ):
	hel = []
	nohel = []

	# split hel boxes anh nohel boxes
	for box in detection:
		if box[5] == 0:
			nohel.append( box )
		else:
			hel.append( box )

	# two arrays below contains indexes locate boxes we wanna take
	hel_av = []
	nohel_av = [ i for i in range(len(nohel)) ]
	for i in range( len(hel) ):
		hel_con = hel[i][4]
		hel_box = (hel[i][0],hel[i][1],hel[i][2],hel[i][3])
		av = True
		nav = nohel_av
		for ni in nav:
			noh_con = nohel[ni][4]
			noh_box = (nohel[ni][0],nohel[ni][1],nohel[ni][2],nohel[ni][3])
			if same_object( hel_box, noh_box ):
				if hel_con >= noh_con:
					nohel_av.remove(ni)
				else:
					av = False
		if av:
			hel_av.append( i )

	return [hel[i] for i in hel_av] + [nohel[x] for x in nohel_av]

def draw_boxes( img, detection ):
	de = [box.astype(int) for box in detection]
	colors = [(255,0,0), (0,255,0)]
	h,w,_ = img.shape

	thick = h//700 + 2
	for box in de:
		img = cv2.rectangle( img, (box[0],box[1]), (box[2],box[3]), colors[box[5]] , thick )
	return img

def count_class( detection, target=0 ):
	count = 0
	for box in detection :
		if box[5] == target:
			count += 1
	return count

class Capture( object ):
	def __init__(self, model,url =0, catch=False) :
		self.model = model
		self.cap = cv2.VideoCapture( url )
		self.cap.set( cv2.CAP_PROP_BUFFERSIZE, 2)
		self.FPS = 30
		# time( milisecond per frame )
		self.MSPF = 1000//self.FPS

		#threading
		self.threading = Thread( target=self.update, args=() )
		self.threading.daemon = True
		self.threading.start()


		# count nohelmet
		self.catch = catch
		self.count = 0

	def update(self):
		while True:
			if self.cap.isOpened():
				self.ret, self.frame = self.cap.read()
			time.sleep( self.MSPF/1000 )

	def show(self):
		img = self.frame
		img, co = detect( img, self.model, get_count=True )

		# put on count number
		font = cv2.FONT_HERSHEY_SIMPLEX
		img = cv2.putText( img, str(co), (20,40), font, 1,(0,0,255), 2, cv2.LINE_AA)

		# check event to catch
		if co>self.count and self.catch :
			# make directory to save catched image
			savedir = 'catch/'
			if not os.path.isdir( savedir ):
				os.mkdir( savedir )

			# catch image's image name formating
			tag = 'catch_'
			index = 0
			while os.path.isfile(savedir+tag+str(index)+".jpg"):
				index += 1
			file_name = savedir+tag+str(index)+".jpg"

			# save image
			cv2.imwrite( file_name, img )
			print( "save to "+file_name )
		self.count = co

		#show img
		cv2.imshow("asdfglzkxjv", img )
		cv2.waitKey( self.MSPF )



class Camera( object ):
	def __init__(self, url=0 ):
		self.cap = cv2.VideoCapture(url)
		self.detection = []
		self.opened = True
		self.FPS = 30
		_, self.frame = self.cap.read()

		# run threading
		self.threading = Thread( target=self.run, args=() )
		self.threading.daemon = True 
		self.threading.start()
	
	def run( self ):
		while 1 :
			if self.cap.isOpened():
				# take frame from camera
				ret, self.frame = self.cap.read()

				# draw boxes on to frame
				self.frame = cv2.cvtColor( self.frame, cv2.COLOR_BGR2RGB )
				self.frame = draw_boxes( self.frame, self.detection )
				self.frame = cv2.cvtColor( self.frame, cv2.COLOR_BGR2RGB )

				# show self.frame
				cv2.imshow( "asfgasdfg", self.frame )

				if cv2.waitKey(1000//self.FPS) == ord('q'):
					self.opened = False
					self.cap.release()
					cv2.destroyAllWindows()
					break

	def update_detection( self, de ):
		self.detection = de
	
	def isOpened( self ):
		return self.opened

class Detective( object ):
	def __init__( self, cam, model ):
		self.cam = cam
		self.model = model

		# object threading
		# self.threading = Thread( target=self.run, args=() )
		# self.threading.start()
	
	def run( self ):
		if self.cam.isOpened():
			# take img from camera
			img = self.cam.frame

			# detect to get boxes
			detection = get_detection( img, self.model )

			# update camera's boxes
			self.cam.update_detection( detection )



# Detection format
# xmin ymin xmax ymax confidence class
# 0.x  0.y  0.x  0.y  0.c        0( or 1,2,... )  


# general
def get_detection( img, model ):
	# convert to yolo format anh predict
	img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
	res = model(img)

	# get detection for filter anh draw boxes
	detection = np.array( res.xyxy[0] )
	return detection_filter( detection )



def detect( img, model, get_count = False  ):
	# convert to yolo format anh predict
	img = cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
	res = model(img)

	# get detection for filter anh draw boxes
	detection = np.array( res.xyxy[0] )
	detection = detection_filter( detection )
	img = draw_boxes( img, detection)

	# return cv2 img has boxes on it
	if not get_count:
		return cv2.cvtColor( img, cv2.COLOR_BGR2RGB )
	else :
		return cv2.cvtColor( img, cv2.COLOR_BGR2RGB ), count_class( detection )