import requests

url = "http://3653-34-83-237-128.ngrok.io"
instance = '/api/face_indentity'

requests.Timeout = 30
def indentity_face(name):
    fileType = name.split(".")[-1]
    files = {'img': (name, open(name, 'rb'), 'image/' + fileType)}

    #encode to muiltpart/form-data
    r = requests.post(url + instance, files=files)
    print(r.text)

indentity_face("test_file/itsme.jpg")
