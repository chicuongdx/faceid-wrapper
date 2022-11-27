import pymongo


file = open("constants/mongoDB.txt", "r")
mongoUri = file.read()
file.close()

client = pymongo.MongoClient(mongoUri)
#link collection face_recognition.faceid
db = client["face_recognition"]

db.faceid.delete_many({})