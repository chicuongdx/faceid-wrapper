import fastapi
import uvicorn
from helpers.bounding_box import BoundingBox
from helpers.facenet import FaceNet
from fastapi import UploadFile, File, Form
import numpy as np
from PIL import Image
from io import BytesIO
import pymongo
import torch

bd_model = BoundingBox('models/best.pt')
app = fastapi.FastAPI()

#monggoDB
file = open("constants/mongoDB.txt", "r")
dirDB = file.read()
file.close()

client = pymongo.MongoClient(dirDB)
#link collection face_recognition.faceid
db = client["face_recognition"]

def read_imagefile(data) -> Image.Image:
    image = Image.open(BytesIO(data))
    return image

#handle wrong type of image exception
@app.exception_handler(fastapi.exceptions.RequestValidationError)
def validation_exception_handler(request, exc):
    return fastapi.responses.JSONResponse(
        status_code=422,
        content={"message": "Wrong type of image", "status": "fail"},
    )
#handle 500 error
@app.exception_handler(fastapi.exceptions.HTTPException)
def http_exception_handler(request, exc):
    return fastapi.responses.JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "status": "fail"},
    )

@app.post('/api/feature_matching')
async def feature_matching(img1: UploadFile = File(...), img2: UploadFile = File(...)):
    #byte to cv2 image
    nparr1 = np.array(read_imagefile(await img1.read()))
    nparr2 = np.array(read_imagefile(await img2.read()))

    face1 = bd_model.crop_boudingbox(nparr1)
    face2 = bd_model.crop_boudingbox(nparr2)
    fn_model = FaceNet('cpu')

    #get embedding
    embedding1 = fn_model.get_embedding(face1)
    embedding2 = fn_model.get_embedding(face2)

    matching, cosine = fn_model.feature_matching_embedding_cosine(embedding1, embedding2)

    #get distance
    if matching[0]:
        distance = fn_model.get_distance(embedding1, embedding2)
        return {"status": "success", "matching": matching, "distance": distance}
    else:
        return {"status": "success", "matching": matching, "cosine": cosine}
    
@app.post('/api/face_register')
async def face_register(faceid: str = Form(...) ,img: UploadFile = File(...)):

    try:
        #mongoDB find all object with faceid = faceid
        faceid_db = db.faceid.find_one({"faceid": faceid})
        if faceid_db is not None:
            return {"message": "Faceid is exist", "status": "fail"}

        nparr = np.array(read_imagefile(await img.read()))
        face = bd_model.crop_boudingbox(nparr)
        fn_model = FaceNet('cpu')

        feature = fn_model.get_embedding(face)

        #tensor to array
        feature = feature.detach().numpy()
        #reshape array [1,n] -> [n]
        feature = feature.reshape(feature.shape[1],)
        feature = feature.tolist()

        #face_sha256 = sha256.hash_embedding(feature)
        #post faceid and feature to mongoDB
        post = {"faceid": faceid, "feature": feature}
        db.faceid.insert_one(post)

        return { "faceid": faceid, "status": "success"}
    except Exception as e:
        return {"msg": e, "status": "fail"}

@app.post('/api/face_indentity')
async def face_indentity(img: UploadFile = File(...)):
    try:
        nparr = np.array(read_imagefile(await img.read()))
        face = bd_model.crop_boudingbox(nparr)
        fn_model = FaceNet('cpu')

        feature = fn_model.get_embedding(face)
        #query_sha256 = sha256.hash_embedding(feature)
        #get all faceid in mongoDB
        faceids = db.faceid.find()

        face_matching = []
        #compare feature with all faceid in mongoDB
        for faceid in faceids:
            #version embedding
            #convert list to array
            faceid_feature = np.array(faceid['feature'])
            #reshape array [n] -> [1,n]
            faceid_feature = faceid_feature.reshape(1,faceid_feature.shape[0])
            #convert array to tensor
            faceid_feature = torch.from_numpy(faceid_feature)
            match, cosine = fn_model.feature_matching_embedding_cosine(faceid_feature, feature)
            #version sha256
            #match, distance = sha256.feature_matching(query_sha256, faceid['feature'])
            if match:
                distance = fn_model.get_distance(faceid_feature, feature)
                print(faceid['faceid'], distance, 'alpha =', cosine, 'Matching')
                face_matching.append({"faceid": faceid['faceid'], "distance": distance})
            else:
                print(faceid['faceid'], 'alpha = ', cosine, 'Not Matching')

        if len(face_matching) == 0:
            return {"faceid": "unknown", "status": "success"}

        #sort face_matching by distance
        face_matching = sorted(face_matching, key = lambda i: i['distance'])
        
        #return all faceid with distance
        return {"faceids": face_matching, "status": "success"}
    except Exception as e:
        return {"msg": e, "status": "fail"}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=True)