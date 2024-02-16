import os
from helpers.bounding_box import BoundingBox
import cv2

class LocalizeEvaluator():
    def __init__(self, url="test_file/tuyen"):
        self._URL_ = url
        self.model = BoundingBox("models/best.pt")
    
    def load_localize(self):
        result = dict()
        lst_true = []
        lst_predict = []
        for file in os.listdir(self._URL_):
            if file.endswith(".txt"):
                with open(self._URL_ + "/" + file, "r") as f:
                    line = f.readline()
                    line = line.split(" ")
                    x, y, w, h = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                    #read file image with same name to get width and height
                    img = cv2.imread(self._URL_ + "/" + file.split(".")[0] + ".jpg")

                    width, height = img.shape[1], img.shape[0]

                    #convert from yolo format to x1, y1, x2, y2
                    x1 = int((x - w / 2) * width)
                    y1 = int((y - h / 2) * height)
                    x2 = int((x + w / 2) * width)
                    y2 = int((y + h / 2) * height)

                    lst_true.append([x1, y1, x2, y2])

                    boxes_predict = self.model.get_boudingbox(img)
                    if boxes_predict is not None:
                        lst_predict.append(boxes_predict)
                    
                    result['true'] = lst_true
                    result['predict'] = lst_predict
        return result

if __name__ == "__main__":
    localize = LocalizeEvaluator()
    result = localize.load_localize()
    print("True:", len(result["true"]))
    print("Predict:", len(result["predict"]))
