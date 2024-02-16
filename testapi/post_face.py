import urllib3
import requests
import os

url = "http://c70c-35-240-237-49.ngrok.io"
instance = '/api/face_register'

#name_file = "test_file/itsme.jpg"
#faceid = "cuong"
requests.Timeout = 30
def register_face(faceid, name):
    fileType = name.split(".")[-1]
    files = {'img': (name, open(name, 'rb'), 'image/' + fileType)}
    data = {'faceid': faceid}

    #encode to muiltpart/form-data
    r = requests.post(url + instance, files=files, data=data)
    print(r.text)

base_id = "tuyen"
base_name = "test_file/tuyen/"
for idx, name in enumerate(os.listdir(base_name)):
    register_face(str(name[:-4]), base_name + name)