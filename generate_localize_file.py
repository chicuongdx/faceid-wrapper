from helpers.bounding_box import BoundingBox
import cv2

bd = BoundingBox("models/best.pt")
def generate_localize_file(img_path, location=None):
    img = cv2.imread(img_path)
    if location is None:
        location = bd.get_boudingbox(img)
        if location is None:
            raise Exception("No face detected")
        if len(location) != 1:
            raise Exception("More than 1 face")
        
    x1, y1, x2, y2 = int(location[0][0]), int(location[0][1]), int(location[0][2]), int(location[0][3])

    #replace file name with .txt
    txt_path = img_path.split(".")[0] + ".txt"
    with open(txt_path, "w") as f:
        #convert to yolo format / img.shape[1] = width, img.shape[0] = height
        x = (x1 + x2) / 2 / img.shape[1]
        y = (y1 + y2) / 2 / img.shape[0]
        w = (x2 - x1) / img.shape[1]
        h = (y2 - y1) / img.shape[0]
        f.write("0 " + str(x) + " " + str(y) + " " + str(w) + " " + str(h))
    
    return txt_path

def generate_localize_file_from_folder(folder_path):
    import os
    for file in os.listdir(folder_path):
        if file.endswith(".jpg") or file.endswith(".png"):
            try:
                generate_localize_file(folder_path + "/" + file)
            except Exception as e:
                print(e)
                pass

if __name__ == "__main__":
    generate_localize_file_from_folder("test_file/tuyen")
        