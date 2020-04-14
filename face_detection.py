import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
model=load_model("my_model.h5") 
conf_threshold = 0.8
DNN = "TF"
if DNN == "CAFFE":
    modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "models/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "models/opencv_face_detector_uint8.pb"
    configFile = "models/opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
def detectFaceOpenCVDnn(img):
    frameHeight = img.shape[0]
    frameWidth = img.shape[1]
    blob = cv2.dnn.blobFromImage(img, 1.0, (1000, 1000), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    faces=[]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            face_image = img[y1:y2, x1:x2]
            faces.append(face_image)
    return faces
#%%
def prediction(faces):
    genders=[]
    for i in range(len(faces)):
        if faces[i].shape[0]!=0 and faces[i].shape[1]!=0 and faces[i].shape[2]!=0:
            img=faces[i]
            img=cv2.resize(img,(96,96))/255
            img= img_to_array(img)
            img=img.reshape((1,96,96,3))
            re=model.predict(img)
            p=np.argmax(re)
            if p==0:
                ti="man"
            elif p==1:
                ti="woman"
            genders.append(ti)   
    return genders   
#%%
def main(img):
        faces= detectFaceOpenCVDnn(img)
        genders=prediction(faces) 
        return genders
img=cv2.imread("ra.jpg")
gender=main(img)
