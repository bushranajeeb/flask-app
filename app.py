#Import necessary libraries
from flask import Flask, render_template, request, session, Response
import pandas as pd
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandas.io.json import json_normalize
import csv
import base64
import json
import pickle
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import urllib


#Initialize the Flask app
app = Flask(__name__)


modelFile = "models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
configFile = "models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classFile = "models/coco_class_labels.txt"


with open(classFile) as fp:
    labels = fp.read().split("\n")
    
# Read the Tensorflow network
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

def detect_objects(net, im):
    dim = 300
    
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(im, 1.0, size=(dim, dim), mean=(0,0,0), swapRB=True, crop=True)

    # Pass blob to the network
    net.setInput(blob)
    
    # Peform Prediction
    objects = net.forward()
    return objects

def display_text(im, text, x, y):
    
    # Get text size 
    textSize = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]
            
    # Use text size to create a black rectangle    
    cv2.rectangle(im, (x,y-dim[1] - baseline), (x + dim[0], y + baseline), (0,0,0), cv2.FILLED);
    # Display text inside the rectangle
    cv2.putText(im, text, (x, y-5 ), FONTFACE, FONT_SCALE, (0, 255, 255), THICKNESS, cv2.LINE_AA)
    

import multiprocessing as mp
import shutil
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input, decode_predictions
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from PIL import Image
import xml.etree.ElementTree as ET
#import psutil
import random
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)



targetx = 224
targety = 224



model = tf.keras.models.load_model('dog_breed_classifier.h5')
behaviour_model = tf.keras.models.load_model('behaviour_classifier.h5')

behaviour_indices = {'Angry': 0,'Barking': 1,'Curious': 2,'Digging': 3,'Eating': 4,'Happy': 5,'Normal': 6,'Sad': 7, 'Sleeping': 8,'Tail_wagging': 9}

class_indices = {'n02085620-Chihuahua': 0, 'n02085782-Japanese_spaniel': 1, 'n02085936-Maltese_dog': 2, 'n02086079-Pekinese': 3, 'n02086240-Shih-Tzu': 4, 'n02086646-Blenheim_spaniel': 5, 
                 'n02086910-papillon': 6, 'n02087046-toy_terrier': 7, 'n02087394-Rhodesian_ridgeback': 8, 'n02088094-Afghan_hound': 9, 'n02088238-basset': 10, 'n02088364-beagle': 11, 'n02088466-bloodhound': 12, 'n02088632-bluetick': 13, 
                 'n02089078-black-and-tan_coonhound': 14, 'n02089867-Walker_hound': 15, 'n02089973-English_foxhound': 16, 'n02090379-redbone': 17, 'n02090622-borzoi': 18, 'n02090721-Irish_wolfhound': 19, 'n02091032-Italian_greyhound': 20, 'n02091134-whippet': 21, 'n02091244-Ibizan_hound': 22, 'n02091467-Norwegian_elkhound': 23, 'n02091635-otterhound': 24, 'n02091831-Saluki': 25, 'n02092002-Scottish_deerhound': 26, 'n02092339-Weimaraner': 27, 'n02093256-Staffordshire_bullterrier': 28, 'n02093428-American_Staffordshire_terrier': 29, 'n02093647-Bedlington_terrier': 30, 
                 'n02093754-Border_terrier': 31, 'n02093859-Kerry_blue_terrier': 32, 'n02093991-Irish_terrier': 33, 'n02094114-Norfolk_terrier': 34, 'n02094258-Norwich_terrier': 35, 'n02094433-Yorkshire_terrier': 36, 'n02095314-wire-haired_fox_terrier': 37, 'n02095570-Lakeland_terrier': 38, 'n02095889-Sealyham_terrier': 39, 'n02096051-Airedale': 40, 'n02096177-cairn': 41, 'n02096294-Australian_terrier': 42, 'n02096437-Dandie_Dinmont': 43, 'n02096585-Boston_bull': 44, 'n02097047-miniature_schnauzer': 45, 'n02097130-giant_schnauzer': 46, 'n02097209-standard_schnauzer': 47, 'n02097298-Scotch_terrier': 48, 
                 'n02097474-Tibetan_terrier': 49, 'n02097658-silky_terrier': 50, 'n02098105-soft-coated_wheaten_terrier': 51, 'n02098286-West_Highland_white_terrier': 52, 'n02098413-Lhasa': 53, 'n02099267-flat-coated_retriever': 54, 'n02099429-curly-coated_retriever': 55, 'n02099601-golden_retriever': 56, 'n02099712-Labrador_retriever': 57, 'n02099849-Chesapeake_Bay_retriever': 58, 'n02100236-German_short-haired_pointer': 59, 'n02100583-vizsla': 60, 'n02100735-English_setter': 61, 'n02100877-Irish_setter': 62, 'n02101006-Gordon_setter': 63, 'n02101388-Brittany_spaniel': 64, 'n02101556-clumber': 65, 'n02102040-English_springer': 66, 'n02102177-Welsh_springer_spaniel': 67, 
                 'n02102318-cocker_spaniel': 68, 'n02102480-Sussex_spaniel': 69, 'n02102973-Irish_water_spaniel': 70, 'n02104029-kuvasz': 71, 'n02104365-schipperke': 72, 'n02105056-groenendael': 73, 'n02105162-malinois': 74, 'n02105251-briard': 75, 'n02105412-kelpie': 76, 'n02105505-komondor': 77, 'n02105641-Old_English_sheepdog': 78, 'n02105855-Shetland_sheepdog': 79, 'n02106030-collie': 80, 'n02106166-Border_collie': 81, 'n02106382-Bouvier_des_Flandres': 82, 'n02106550-Rottweiler': 83, 'n02106662-German_shepherd': 84, 'n02107142-Doberman': 85, 'n02107312-miniature_pinscher': 86, 'n02107574-Greater_Swiss_Mountain_dog': 87, 'n02107683-Bernese_mountain_dog': 88, 'n02107908-Appenzeller': 89, 
                 'n02108000-EntleBucher': 90, 'n02108089-boxer': 91, 'n02108422-bull_mastiff': 92, 'n02108551-Tibetan_mastiff': 93, 'n02108915-French_bulldog': 94, 'n02109047-Great_Dane': 95, 'n02109525-Saint_Bernard': 96, 'n02109961-Eskimo_dog': 97, 'n02110063-malamute': 98, 'n02110185-Siberian_husky': 99, 'n02110627-affenpinscher': 100, 'n02110806-basenji': 101, 'n02110958-pug': 102, 'n02111129-Leonberg': 103, 'n02111277-Newfoundland': 104, 'n02111500-Great_Pyrenees': 105, 'n02111889-Samoyed': 106, 'n02112018-Pomeranian': 107, 'n02112137-chow': 108, 'n02112350-keeshond': 109, 'n02112706-Brabancon_griffon': 110, 'n02113023-Pembroke': 111, 'n02113186-Cardigan': 112, 'n02113624-toy_poodle': 113, 'n02113712-miniature_poodle': 114, 'n02113799-standard_poodle': 115, 'n02113978-Mexican_hairless': 116, 'n02115641-dingo': 117, 'n02115913-dhole': 118, 'n02116738-African_hunting_dog': 119}

FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1
threshold = 0.2


camera = cv2.VideoCapture(0)
'''
for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
for local webcam use cv2.VideoCapture(0)
'''

def gen_frames():  
    while True:
        success, img = camera.read()  # read the camera frame
        if not success:
            break
        else:
            image= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            objects = detect_objects(net, img)


            rows = img.shape[0]; cols = img.shape[1]
            resize3 = cv2.resize(img, (224,224))
            resize4 = cv2.resize(image, (224,224))

            # For every Detected Object
            for i in range(objects.shape[2]):
                # Find the class and confidence 
                classId = int(objects[0, 0, i, 1])
                score = float(objects[0, 0, i, 2])

                # Recover original cordinates from normalized coordinates
                x = int(objects[0, 0, i, 3] * cols)
                y = int(objects[0, 0, i, 4] * rows)
                w = int(objects[0, 0, i, 5] * cols - x)
                h = int(objects[0, 0, i, 6] * rows - y)


                # Check if the detection is of good quality
                if score > threshold:

                    if classId == 18 or classId == 17 or classId == 17:
                    #             display_text(img, "{}".format(labels[classId]), x, y)
                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
                        try:
                            crop = img[y:y+h, x:x+w]
                            crop2= image[y:y+h, x:x+w]

                            resize = cv2.resize(crop, (224, 224))
                            resize2 = cv2.resize(crop2, (224, 224))
                            
                            probabilities = model.predict(preprocess_input(np.expand_dims(resize2, axis=0)))
                            breed_list = tuple(zip(class_indices.values(), class_indices.keys()))

                            behaviour_probabilities = behaviour_model.predict(preprocess_input(np.expand_dims(resize, axis=0)))
                            behaviour_list = tuple(zip(behaviour_indices.values(), behaviour_indices.keys()))
                            
                            max_index1 = np.argmax(probabilities[0])
                            predicted1 = breed_list[max_index1][1].split("-")[1]


                            max_index2 = np.argmax(behaviour_probabilities[0])
                            predicted2 = behaviour_list[max_index2][1]

                            display_text(img, "{}".format(predicted2), x+10, y+30)
                            display_text(img, "{}".format(predicted1), x+10, y+60)

                            
                        except Exception as e:
                            print("detection on original image")
                            probabilities = model.predict(preprocess_input(np.expand_dims(resize3, axis=0)))
                            breed_list = tuple(zip(class_indices.values(), class_indices.keys()))

                            behaviour_probabilities = behaviour_model.predict(preprocess_input(np.expand_dims(resize3, axis=0)))
                            behaviour_list = tuple(zip(behaviour_indices.values(), behaviour_indices.keys()))
                            
                            max_index1 = np.argmax(probabilities[0])
                            predicted1 = breed_list[max_index1][1].split("-")[1]


                            max_index2 = np.argmax(behaviour_probabilities[0])
                            predicted2 = behaviour_list[max_index2][1]

                            display_text(img, "{}".format(predicted2), x+10, y+30)
                            display_text(img, "{}".format(predicted1), x+10, y+60)

                            pass


            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')
	
@app.route('/about')
def about():
    return render_template('about.html',title='About',name='Passed by variable')

@app.route("/predict")
def predict():
    return render_template("predict.html",title="Predict")

    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    

    
	
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)