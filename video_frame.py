from helpers.bounding_box import BoundingBox
import cv2

class VideoFrame:
    def __init__(self, cap_url, bd_box_url):
        self.cap = cv2.VideoCapture(cap_url)
        self.bd_box = BoundingBox(bd_box_url)
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            return None
    
    def draw_boudingbox(self, frame):
        return self.bd_box.draw_boudingbox(frame)

    def crop_boudingbox(self, frame, location=None):
        return self.bd_box.crop_boudingbox(frame, location)

    def  run(self):
        while True:
            frame = self.get_frame()
            if frame is None:
                break
            frame = self.draw_boudingbox(frame)
            # try:
            #     frame = self.crop_boudingbox(frame)
            # except:
            #     continue
            #frame = self.crop_boudingbox(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.release()

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()