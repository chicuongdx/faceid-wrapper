import requests
import json


url = "http://aa6e-34-124-207-46.ngrok.io"
instance = '/api/face_indentity'

requests.Timeout = 30
def indentity_face(name):
    fileType = name.split(".")[-1]
    files = {'img': (name, open(name, 'rb'), 'image/' + fileType)}

    #encode to muiltpart/form-data
    r = requests.post(url + instance, files=files)
    json_string = r.text
    obj = json.loads(json_string)
    print(r.text)
    return obj

obj_response = indentity_face("test_file/tuyen.jpg")

print(obj_response)
print(len(obj_response['faceids']))