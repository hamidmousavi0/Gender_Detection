#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101
import sys
import time
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from utils import label_map_util
from utils import visualization_utils_color as vis_util
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import multiprocessing
packages=sys.modules

min_score_thresh=.5
model=load_model("my_model.h5")    
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'
PATH_TO_LABELS = './protos/face_label_map.pbtxt'
NUM_CLASSES = 2
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
with detection_graph.as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

def face_detection(frame):
    with tf.Session(graph=detection_graph, config=config) as sess:
        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        faces=[]       
        i=0
        while (scores[0,i]>min_score_thresh):
            y_min=int(boxes[0,i,0]*frame.shape[0])
            x_min=int(boxes[0,i,1]*frame.shape[1])
            y_max=int(boxes[0,i,2]*frame.shape[0])
            x_max=int(boxes[0,i,3]*frame.shape[1])
            face=frame[y_min:y_max,x_min:x_max]
            faces.append(face)
            i=i+1
        return faces    
def prediction(faces,model):
    genders=[]
    for i in range(len(faces)):
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
def main(img):
        faces= face_detection(img) 
       	genders=prediction(faces,model) 
        return genders
img=cv2.imread("ra.jpg")
genders=main(img)

    
    

