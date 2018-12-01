import cv2
import numpy as np
from PIL import Image
from keras import models
from glob import glob
import random
from sklearn.datasets import load_files       
from keras.utils import np_utils
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image     
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input, decode_predictions


def main():

    christmas_names = [item[28:-1] for item in sorted(glob("data\images\christmas/train/*/"))]
    print(christmas_names)
    model = models.load_model('saved_models/christmas_model.h5')

    video = cv2.VideoCapture(0)

    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    text_loc = (frame_height//2, frame_width//2 - 300)

    while True:
        _, frame = video.read()

        frontal_face(frame)

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into 1920x1080 because we trained the model with this image size.
        im = im.resize((224, 224))
        img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 1920x1080x3 into 1x1920x1080x3 
        img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict method on model to predict safety class on the image
        #prediction = int(model.predict(img_array)[0][0])

        #names = christmas_type_predictor(frame, img_array, model, christmas_names)
        
        ret_val_text, ret_val_pred = christmas_classifier(frame, img_array, model, christmas_names)
        #if prediction is 0, which means safe is missing on the image, then show the frame in gray color.
        #if prediction == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if(ret_val_text == "Did not detect frontal face"):
            cv2.putText(frame, ret_val_text, 
                text_loc, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA)
        else:
            cv2.putText(frame, ret_val_text, 
                text_loc, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA)

            text_loc = (frame_height//2 - 200, frame_width//2 - 100)

            pred_text = " You resemble: " +  christmas_names[0] + ": " + str(ret_val_pred[0]) + " " + christmas_names[1] + ": " + str(ret_val_pred[1]) + " " + christmas_names[2] + ": " + str(ret_val_pred[2])

            cv2.putText(frame, pred_text, 
                text_loc, 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA)

        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    video.release()
    #out.release()
    cv2.destroyAllWindows()

def face_detector(frame):
    face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')
    img = frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def path_to_tensor(frame):
    x = image.img_to_array(frame)
    return np.expand_dims(x, axis=0)


def frontal_face(img_array):

    face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')

    # convert BGR image to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # find faces in image
    faces = face_cascade.detectMultiScale(gray)

    # get bounding box for each detected face
    for (x,y,w,h) in faces:
        # add bounding box to color image
        cv2.rectangle(img_array,(x,y),(x+w,y+h),(255,0,0),2)
        
    # convert BGR image to RGB for plotting
    cv_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    return(cv_rgb)


def christmas_classifier(frame, img_array, model, christmas_names):
        
    if face_detector(frame):
        print("Human detected")
        ret_val_text, ret_val_pred = christmas_type_predictor(frame, img_array, model, christmas_names)
        print("Resembling christmas character is {}".format(ret_val_text))
    else:
        print("Did not detect frontal face")
        ret_val_text = "Did not detect frontal face"
        ret_val_pred = 0

    return ret_val_text, ret_val_pred

def christmas_type_predictor(frame, img_array, model, christmas_names):
    
    im = Image.fromarray(frame, 'RGB')
    im = im.resize((224, 224))
    frame = np.array(im)
    predicted_vector = model.predict(path_to_tensor(frame))
    cv_color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #print(predicted_vector)
    return christmas_names[np.argmax(predicted_vector)], predicted_vector.flat[0:3]

if __name__ == '__main__':
    main()

