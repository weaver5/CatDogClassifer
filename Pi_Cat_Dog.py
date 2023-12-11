import os
import numpy as np
import cv2
import tensorflow as tf
import time
from sense_hat import SenseHat

model_filepath = '/home/pi/Downloads/cat_dog_model2.h5' #loading the model in
model = tf.keras.models.load_model(model_filepath, compile=False)

def resize_then_show(img):
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)#resized to match what i did in the model
    cv2.imshow("frame", img)
    img_processed = img.reshape((1, 128, 128, 3)).astype('float32') / 255.0 #normalizing and sizing
    return img_processed

def disp_sensehat(sense, predicted_class):
    red = (255, 0, 0)   #CHRISTMAS COLORS
    green = (0, 255, 0)  

    sense.show_message(predicted_class, text_colour=red, back_colour=green)#showing sensehat messg

def prediction(catDogNN, img):
    img_processed = resize_then_show(img)#resizing then i am showing it
    predictions = catDogNN.predict(img_processed) # iwant to predict then 
    #this function is only called when i press c because of lagging if ran always
    predicted_class = np.argmax(predictions)#whats the class?
    confDog = predictions[0][0] * 100 #challenging to try to find balance
    confCat = 100 - confDog #whats the confidence? i had to play around with this
    #kept getting same answer each time before finding this
    #tried [00] and it did not work
    pet_prediction = "Cat" if confCat >=confDog else "Dog" #see if it is greater
    #the code above helps with determining and setting
    disp_sensehat(sense, pet_prediction)#sensehat displaying of the pred
    return pet_prediction, confDog, confCat

cap = cv2.VideoCapture(0) #open video feedback
sense = SenseHat() #sensehat intializaiton

while True:
    ret, frame = cap.read()
    cv2.imshow('Video Feed', frame)#showing output

    key = cv2.waitKey(1) & 0xFF #c means to capture the image
    if key == ord('c'):
        #call function and do prediction
        pet_prediction, confidence_dog, confidence_cat = prediction(model, frame)
        #printing below
        print(f"Pet: {pet_prediction}, My Cat-Conf: {confidence_cat:.2f}%, My Dog-Conf: {confidence_dog:.2f}%")
        accuracy = max(confidence_dog, confidence_cat)
        print(f"accuracy%: {accuracy:.2f}%")

    elif key == ord('q'):
        break #will quit!

cap.release()
cv2.destroyAllWindows()
