import torch
import pymongo
from helpers.facenet import FaceNet
from helpers.bounding_box import BoundingBox
import cv2
import numpy as np

#monggoDB
file = open("constants/mongoDB.txt", "r")
mongoUri = file.read()
file.close()

client = pymongo.MongoClient(mongoUri)
#link collection face_recognition.faceid
db = client["face_recognition"]

fn_model = FaceNet('cpu')
bd_model = BoundingBox('models/best.pt')
#what is ground truth
def get_ground_truth(id1, id2):
    #get ground truth
    ground_truth = db.faceid.find_one({"faceid": id1})
    ground_truth = ground_truth['ground_truth']
    if id2 in ground_truth:
        return True
    else:
        return False

#precision
def face_matching(face):
    #get ground truth
    faceids = db.faceid.find()
    #get embedding of face
    
    embedding = fn_model.get_embedding(face)
    #feature matching
    
    query_face = []
    for faceid in faceids:
        #list to numpy array
        faceid_feature = np.array(faceid['feature'])
            #reshape array [n] -> [1,n]
        faceid_feature = faceid_feature.reshape(1,faceid_feature.shape[0])
            #convert array to tensor
        faceid_feature = torch.from_numpy(faceid_feature)

        matching, cosine = fn_model.feature_matching_embedding_cosine(faceid_feature, embedding)
        if matching:
            query_face.append(faceid)
    return query_face, list(faceids)

#main
if __name__ == '__main__':
    #read image test/bankgirl.jpg
    img = cv2.imread('test_file/bankgirl.jpg')
    face = bd_model.crop_boudingbox(img)
    query_face, ground_truth = face_matching(face)
    print(len(query_face), len(ground_truth))
