from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import os, cv2
import torch
import torch.nn as nn
import numpy as np

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
class FaceNet:
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(
            classify=False,
            pretrained='vggface2'
        ).to(self.device)
        self.model.eval()
    
    def get_embedding(self, img):
        # formula for fixed_image_standardization
        # x = (x - mean) / std
        img = cv2.resize(img, (512, 512))
        img = fixed_image_standardization(img)
        img = torch.from_numpy(np.array(img, dtype=np.float32)).to(self.device)
        embedding = self.model(img.unsqueeze(0).permute(0, 3, 1, 2))
        return embedding
    
    def get_distance(self, embedding1, embedding2):
        distance = torch.dist(embedding1, embedding2, 2)
        #distance = cos(embedding1, embedding2)
        return distance.item()
    
    def cosine_similarity(self, embedding1, embedding2):
        return cos(embedding1, embedding2).item()

    #region feature matching
    def feature_matching_cosine(self, img1, img2, thresh=0.9):
        embedding1 = self.get_embedding(img1)
        embedding2 = self.get_embedding(img2)
        cos_sim = self.cosine_similarity(embedding1, embedding2)
        print(cos_sim)
        if cos_sim > thresh:
            return True
        return False

    def feature_matching(self, img1, img2, thresh=0.9):
        embedding1 = self.get_embedding(img1)
        embedding2 = self.get_embedding(img2)
        distance = self.get_distance(embedding1, embedding2)
        print(distance)
        if distance < thresh:
            return True
        return False

    def feature_matching_embedding(self, embedding1, embedding2, thresh=1):
        distance = self.get_distance(embedding1, embedding2)
        print(distance)
        if distance < thresh:
            return True, distance
        return False, distance
    
    def feature_matching_embedding_cosine(self, embedding1, embedding2, thresh=0.8):
        cos_sim = self.cosine_similarity(embedding1, embedding2)
        if cos_sim > thresh:
            return True, cos_sim
        return False, cos_sim
    #endregion