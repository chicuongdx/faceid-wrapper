import torch
from imutils import face_utils
import cv2
import numpy as np

class Yolov5:
    def __init__(self, url):
        self._URL_ = url
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cpu'

    def load_model(self):
        model = torch.hub.load('./yolov5', 'custom', path=self._URL_, source='local')
        return model
    
    def detect(self, img):
        results = self.model(img)
        return results

