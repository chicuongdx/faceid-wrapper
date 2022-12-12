from helpers.object_detection import Yolov5 as yolov5
import cv2
from imutils import face_utils

class BoundingBox(yolov5):
    def __init__(self, url="models/best.pt"):
        self.model = yolov5(url)
        self.classes = self.model.classes
        self.device = self.model.device

    def is_same_location(self, box1, box2, thresh=0.5):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        #calculate the overlap area
        x_overlap = max(0, min(x2, x4) - max(x1, x3))
        y_overlap = max(0, min(y2, y4) - max(y1, y3))
        overlap_area = x_overlap * y_overlap
        #calculate the union area
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - overlap_area
        #calculate the IoU
        iou = overlap_area / union_area
        if iou > thresh:
            return True
        return False

    def draw_boudingbox(self, img):
        results = self.model.detect(img)
        #delete box in boxes with low confidence if they same location
        boxes = results.xyxy[0].tolist()
        if len(boxes) == 0:
            return img
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i != j:
                    box1 = boxes[i]
                    box2 = boxes[j]
                    print(box1)
                    print(box2)
                    try:
                        if box1 is not None and box2 is not None:
                            if box1[4] < box2[4] and self.is_same_location(box1, box2):
                                boxes[i] = None
                                break
                    except:
                        print("Exception due to don't found box2")
                        pass
        boxes = [box for box in boxes if box is not None]

        for xyxy in boxes:
            
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return img
    
    def crop_boudingbox(self, img, location=None):
        if location is None or len(location) == 0:
            results = self.model.detect(img)
            location = results.xyxy[0].tolist()

        if len(location) == 0:
            return img

        if len(location) != 1:
            print("More than 1 face")
            return img

        xyxy = location[0]
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        crop_img = img[y1:y2, x1:x2]
        return crop_img
    
    def get_boudingbox(self, img):
        results = self.model.detect(img)
        boxes = results.xyxy[0].tolist()

        if len(boxes) == 0:
            return None
        
        if len(boxes) != 1:
            print("More than 1 face")
            return None
            
        return boxes