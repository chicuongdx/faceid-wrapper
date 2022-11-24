import hashlib
import cv2
from helpers.bounding_box import BoundingBox

#get face from image
def get_face(img):
    face_detector = BoundingBox("models/best.pt")
    face = face_detector.crop_boudingbox(img)
    return face

#use sha256 to hash image
def hash_image(img):
    # fixed ValueError: ndarray is not C-contiguous
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    return hashlib.sha256(img).hexdigest()

#crop face from image and calculate sha256
def get_face_hash(img):
    face = get_face(img)
    if face is None:
        return None
    return hash_image(face)

def hamming(s1, s2):
    #decoder hash image to binary
    s1 = bin(int(s1, 16))[2:]
    s2 = bin(int(s2, 16))[2:]

    #padding 0 to make two string have the same length
    if len(s1) > len(s2):
        s2 = s2.zfill(len(s1))
    else:
        s1 = s1.zfill(len(s2))
    
    #calculate hamming distance
    distance = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            distance += 1
    return distance / len(s1)

    # assert len(s1) == len(s2)
    # return float(sum(c1 != c2 for c1, c2 in zip(s1, s2))) / float(len(s1))

def compare_faces(img1, img2):
    '''
    Compare two faces.
    '''
    hash1 = get_face_hash(img1)
    hash2 = get_face_hash(img2)
    if hash1 is None or hash2 is None:
        return None
    return hamming(hash1, hash2)

#feature matching with hamming distance
def feature_matching(img1, img2, thresh=0.9):
    distance = compare_faces(img1, img2)
    print(distance)
    if distance is None:
        return False
    if distance < thresh:
        return True
    return False

#calculate sha256 of vector embedding from facenet model (1, 512)
def hash_embedding(embedding):
    #tensor to numpy
    embedding = embedding.detach().numpy()
    return hashlib.sha256(embedding).hexdigest()

#compare two vector embedding from facenet model (1, 512)
def compare_embedding(embedding1, embedding2):
    hash1 = hash_embedding(embedding1)
    hash2 = hash_embedding(embedding2)
    print(hash1)
    print(hash2)
    return hamming(hash1, hash2)

#feature matching with hamming distance
# def feature_matching(embedding1, embedding2, thresh=1.0):
#     distance = compare_embedding(embedding1, embedding2)
#     print(distance)
#     if distance is None:
#         return False
#     if distance < thresh:
#         return True, distance
#     return False, distance

# def feature_matching(hash1, hash2, thresh=0.5):
#     distance = hamming(hash1, hash2)
#     print(distance)
#     if distance is None:
#         return False
#     if distance < thresh:
#         return True, distance
#     return False, distance