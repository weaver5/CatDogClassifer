#!/usr/bin/env python
# coding: utf-8

# # <center><font color=maroon>CPE4903 Project: Cats and Dogs Classifier</font> </center>
# 
# ### In this mini-project, you will develop a CNN model for the cat-and-dog classifer. 
# #### You will create `at least two models`, applying the various techniques we discussed for improving the performance. 
# 
# 1. Deeper Conv layers and/or FC layers
# 2. Image augmentation
# 3. Transfer learning
# 4. Regularization
# 5. Increasing image size
# 6. Increasing size of the train/validation/test dataset
# 
# * You will compare the performance of your models with the baseline VGG-5 model.  
# * <font color=red>Performance requirement: the accuracy on the test data needs to be better than 85% for at least one of your models </font>
# * You will save the parameters of the model at the end, which will be deployed on Raspberry Pi.

# ### Cats & Dogs Dataset
# 
# * #### You are given a zip file, `train.zip`, that contains 25,000 labelled images of cats and dogs (12,500 each) 
# * #### You will select a subset of $N$ image files from the dataset and store them in the following sub-directory structure, where $N$ can be anywhere between 6,000 to 25,000.
# * #### The train-validation-test data split is 60%-15%-25%.

# ### Load tool modules

# In[26]:


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from sklearn.model_selection import train_test_split

import os
import itertools
import time
import random
import shutil  # shutil is a utility for file system operations
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
from tensorflow import keras
from keras import layers


# ### Load CNN models

# In[27]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout


# ### Load the image processing tools 

# In[28]:


from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory


# ### Load and Process the dataset
# __Create the subdirectory structures per the requirement.__

# 

# ### Create directory and have 3 subdirectories

# In[29]:


#Source directory containing all images
source_dir = r"C:\Users\Michaela Crego\Downloads\train"
#Destination directory where you want to save the subset
train_sub_path = './cat_dog_6000'
#Create the subset directory if it doesn't exist
os.makedirs(train_sub_path, exist_ok=True)


# In[30]:


def make_subset (subset_name, st_index, end_index):
    for category in ("cat", "dog"):
        dest = train_sub_path + '/' + subset_name + '/' + category
        os.makedirs(dest, exist_ok=True)
        
        files = [f"{category}.{i}.jpg" for i in range(st_index, end_index)]  # list of catxxx.jpg, dogxxx.jpg
        for f in files:
            shutil.copy(src=source_dir + '/' + f, dst=dest + '/ '+ f)
            #shutil.copy(src=source_dir + '\\' + f, dst=dest + '\\' + f)


# In[31]:


make_subset("train", st_index=0, end_index=6000)
make_subset("validation", st_index=6000, end_index=7500)
make_subset("test", st_index=7500, end_index=10000)


# ### Display 2 input images: one for dog, and one for cat 

# In[32]:


#RANDOM DOG
directory = os.listdir(train_sub_path+'/test'+'/dog')
sample = random.choice(directory)

img = load_img(os.path.join(train_sub_path+'/test'+'/dog', sample))

plt.imshow(img)
plt.show()


# In[39]:


#RANDOM CAT
directory = os.listdir(train_sub_path+'/test'+'/cat')
sample = random.choice(directory)

img = load_img(os.path.join(train_sub_path+'/test'+'/cat', sample))

plt.imshow(img)
plt.show()


# ### Image data pipeline:

# In[34]:


# create data generator
datagen = ImageDataGenerator(rescale=1.0/255.0)

train_data = datagen.flow_from_directory(train_sub_path + '/train',
            class_mode='binary', batch_size=64, target_size=(64, 64))

val_data = datagen.flow_from_directory(train_sub_path + '/validation',
           class_mode='binary', batch_size=64, target_size=(64, 64))

test_data = datagen.flow_from_directory(train_sub_path + '/test',
            class_mode='binary', batch_size=64, target_size=(64, 64))


# # <font color=Orchid>Build CNN Model One!!!!!!!!!!!!!!!!!!!!!!!!!!!</font>
# __Use CONV, POOL and FC layers to construct your CNN model. You can also load pre-trained model, if transfer learning is used. You will train and test the model after this step.__
# 
# <font color=deeppink1>__I will use Deeper Conv layers and/or FC layers for model1__</font> 

# ## <font color=green>Define the CNN Model1</font> 

# In[35]:


model1 = Sequential()

# Layer 1
model1.add(Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model1.add(MaxPooling2D((2, 2)))
model1.add(Dropout(0.25))

# Layer 2
model1.add(Conv2D(128, (3, 3), activation='relu'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Dropout(0.25))

# Layer 3
model1.add(Conv2D(256, (3, 3), activation='relu'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Dropout(0.25))

model1.add(Flatten())

# FC Layers
model1.add(Dense(256, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(1, activation='sigmoid'))


# ### Print the model summary that shows the output shape and # of parameters for each layer.

# In[36]:


model1.summary()


# ### <font color=Salmon>Question: What are the total number of parameters for the model?</font>

# In[37]:


#2730625 total parameters


# ## <font color=green>Train the CNN Model1</font>
# 
# __Note: Display the history when running model.fit( )__

# In[38]:


import timeit
start_time = timeit.default_timer()
#start time----------------------------

model1.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model1.fit(train_data, epochs=30, batch_size=128, validation_data=val_data, verbose=1)

#end time------------------------------
end_time = timeit.default_timer()
elapsed_time = end_time - start_time
print(f"Total training time: {elapsed_time:.2f} seconds")


# ### <font color=Salmon>Question: What is the estimated total model training time?</font>

# In[15]:


#2377.53 seconds
#I was also trying to get colab going and doing other tasks on my computer at the same time, so it
#might have changed the time


# ### Compare Loss and Accuracy Performance for train and validation data
# 
# #### Plot the loss data, for both train and validation data

# In[16]:


J = history.history['loss']  # Loss data for Training 
J_val = history.history['val_loss']

plt.figure(figsize=(10,7))

plt.title('Model Loss Performance: Train vs. validation')
plt.plot(J, color='DodgerBlue', label='Train')
plt.plot(J_val, color='orange', label='Validation')

plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend()
plt.grid()
plt.show()


# #### Plot the accuracy data, for both train and validation data

# In[17]:


accu = history.history['accuracy']  # Loss data for Training 
accu_val = history.history['val_accuracy']

plt.figure(figsize=(10,7))

plt.title('Model Accuracy Performance: Train vs. validation')
plt.plot(accu, color='DodgerBlue', label='Train')
plt.plot(accu_val, color='orange', label='Validation')

plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend()
plt.grid()
plt.show()


# ## <font color=green>Test the CNN Model1</font>
# __Note: Display the history when running model.evaluate( )__

# In[40]:


loss, accuracy = model1.evaluate(test_data, verbose=1)

y_pred = model1.predict(test_data)


# ### <font color=Salmon>Question: What is the estimated inference (testing) time on test dataset?</font>

# In[24]:


#roughly 46 seconds


# ### Print the final loss and accuracy of the test data

# In[19]:


print(f"test loss: {loss:.4f}")
print(f"test accuracy: {accuracy:.4f}")


# ### Save the CNN model parameters

# In[42]:


model1.save('cat_dog_model1.h5')


# # <font color=Orchid>Build CNN Model Two!!!!!!!!!!!!!!!!!!</font>
# 
# __Use CONV, POOL and FC layers to construct your CNN model. You can also load pre-trained model, if transfer learning is used. You will train and test the model after this step.__
# 
# <font color=deeppink1>__I will use Image augmentation for model2__</font> 

# ## <font color=green>Define the CNN Model2</font> 
# 
# __I will use a different model__

# In[16]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# New target size for images
new_resolution = (128, 128)

# Create data generator w/ image augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(train_sub_path + '/train',
                                         class_mode='binary', batch_size=64, target_size=new_resolution)

val_data = val_datagen.flow_from_directory(train_sub_path + '/validation',
                                           class_mode='binary', batch_size=64, target_size=new_resolution)

test_data = val_datagen.flow_from_directory(train_sub_path + '/test',
                                            class_mode='binary', batch_size=64, target_size=new_resolution)

model2 = Sequential()

# Layer 1 with updated input_shape
model2.add(Conv2D(64, (3, 3), activation='relu', input_shape=(new_resolution[0], new_resolution[1], 3)))
model2.add(MaxPooling2D((2, 2)))

# Layer 2
model2.add(Conv2D(128, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2)))

# Layer 3
model2.add(Conv2D(264, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2)))

model2.add(Flatten())

# FC Layers
model2.add(Dense(264, activation='relu'))
model2.add(Dense(1, activation='sigmoid'))


# ### Print the model summary that shows the output shape and # of parameters for each layer.

# In[45]:


model2.summary()


# ### <font color=Salmon>Question: What are the total number of parameters for the model?</font>

# In[12]:


#2889625 parameters


# ## <font color=green>Train the CNN Model2</font>
# 
# __Note: Display the history when running model.fit( )__

# In[19]:


import timeit
start_time = timeit.default_timer()
#start time----------------------------

model2.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model2.fit(train_data, epochs=80, validation_data = val_data, verbose=1)

#end time------------------------------
end_time = timeit.default_timer()
elapsed_time = end_time - start_time
print(f"Total training time: {elapsed_time:.2f} seconds")


# ### <font color=Salmon>Question: What is the estimated total model training time?</font>

# In[34]:


#10813.19 seconds


# ### Compare Loss and Accuracy Performance for train and validation data
# 
# #### Plot the loss data, for both train and validation data

# In[43]:


J = history.history['loss']  # Loss data for Training 
J_val = history.history['val_loss']

plt.figure(figsize=(10,7))

plt.title('Model Loss Performance: Train vs. validation')
plt.plot(J, color='DodgerBlue', label='Train')
plt.plot(J_val, color='orange', label='Validation')

plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend()
plt.grid()
plt.show()


# #### Plot the accuracy data, for both train and validation data

# In[44]:


accu = history.history['accuracy']  # Loss data for Training 
accu_val = history.history['val_accuracy']

plt.figure(figsize=(10,7))

plt.title('Model Accuracy Performance: Train vs. validation')
plt.plot(accu, color='DodgerBlue', label='Train')
plt.plot(accu_val, color='orange', label='Validation')

plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend()
plt.grid()
plt.show()


# ## <font color=green>Test the CNN Model2</font>
# __Note: Display the history when running model.evaluate( )__

# In[20]:


loss, accuracy = model2.evaluate(test_data, verbose=1)

y_pred = model2.predict(test_data)


# ### <font color=Salmon>Question: What is the estimated inference (testing) time on test dataset?</font>

# In[38]:


#86 seconds


# ### Print the final loss and accuracy of the test data

# In[21]:


print(f"test loss: {loss:.4f}")
print(f"test accuracy: {accuracy:.4f}")


# ### Save the CNN model parameters

# In[22]:


model2.save('cat_dog_model2.h5')


# # <font color=Orchid>Build CNN Model Three!!!!!!!!!!!!!!!!!!</font>
# 
# __Use CONV, POOL and FC layers to construct your CNN model. You can also load pre-trained model, if transfer learning is used. You will train and test the model after this step.__
# 
# <font color=deeppink1>__I will use regularization usng dropout increase and add a layer for model3__</font> 

# ## <font color=green>Define the CNN Model3</font> 
# 
# __I will use a different model__

# In[53]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

model3 = Sequential()

# Layer 1
model3.add(Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model3.add(MaxPooling2D((2, 2)))
model3.add(Dropout(0.30))

# Layer 2
model3.add(Conv2D(128, (3, 3), activation='relu'))
model3.add(MaxPooling2D((2, 2)))
model3.add(Dropout(0.30))

# Layer 3
model3.add(Conv2D(128, (3, 3), activation='relu'))
model3.add(MaxPooling2D((2, 2)))
model3.add(Dropout(0.30))

# Layer 4
model3.add(Conv2D(256, (3, 3), activation='relu'))
model3.add(MaxPooling2D((2, 2)))
model3.add(Dropout(0.30))

model3.add(Flatten())

# FC Layers
model3.add(Dense(256, activation='relu'))
model3.add(Dropout(0.5))
model3.add(Dense(1, activation='sigmoid'))


# ### Print the model summary that shows the output shape and # of parameters for each layer.

# In[54]:


model3.summary()


# ### <font color=Salmon>Question: What are the total number of parameters for the model?</font>

# In[55]:


#781057  


# ## <font color=green>Train the CNN Model3</font>
# 
# __Note: Display the history when running model.fit( )__

# In[56]:


import timeit
start_time = timeit.default_timer()
#start time----------------------------

model3.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history = model3.fit(train_data, epochs=30, batch_size=64, validation_data=val_data, verbose=1)

#end time------------------------------
end_time = timeit.default_timer()
elapsed_time = end_time - start_time
print(f"Total training time: {elapsed_time:.2f} seconds")


# ### <font color=Salmon>Question: What is the estimated total model training time?</font>

# In[57]:


#2358.96  seconds


# ### Compare Loss and Accuracy Performance for train and validation data
# 
# #### Plot the loss data, for both train and validation data

# In[58]:


J = history.history['loss']  # Loss data for Training 
J_val = history.history['val_loss']

plt.figure(figsize=(10,7))

plt.title('Model Loss Performance: Train vs. validation')
plt.plot(J, color='DodgerBlue', label='Train')
plt.plot(J_val, color='orange', label='Validation')

plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend()
plt.grid()
plt.show()


# #### Plot the accuracy data, for both train and validation data

# In[59]:


accu = history.history['accuracy']  # Loss data for Training 
accu_val = history.history['val_accuracy']

plt.figure(figsize=(10,7))

plt.title('Model Accuracy Performance: Train vs. validation')
plt.plot(accu, color='DodgerBlue', label='Train')
plt.plot(accu_val, color='orange', label='Validation')

plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend()
plt.grid()
plt.show()


# ## <font color=green>Test the CNN Model3</font>
# __Note: Display the history when running model.evaluate( )__

# In[60]:


loss, accuracy = model3.evaluate(test_data, verbose=1)

y_pred = model3.predict(test_data)


# ### <font color=Salmon>Question: What is the estimated inference (testing) time on test dataset?</font>

# In[61]:


#25 seconds


# ### Print the final loss and accuracy of the test data

# In[62]:


print(f"test loss: {loss:.4f}")
print(f"test accuracy: {accuracy:.4f}")


# ### Save the CNN model parameters

# In[63]:


model3.save('cat_dog_model3')


# ## <font color=magenta>__✧Conclusion✧__</font>
# 
# ### You will fill out information in this table:
# | Model              | Accuracy | Number of Parameters | Training Time | Inference Speed |
# |-------------------- |----------|-----------------------|--------------- |------------------|
# | Baseline VGG-5     |   0.7585       |    683,329           |    142s            |       6s           |
# | Model One           |  0.8927        |     2730625                  |   2377.53s             |          46s        |
# | Model Two           |    0.9079      |        2889625                |     10813.19s           |           29s       |
# | Model Three         |   0.8664       |          781057            |      2358.96s          |   25s               |
# 
# 
# __You can also add comments on what you tried and observed while working on the project.__

#  <font color=magenta>__✧Model1 Comments✧__</font>

# In[68]:


#by increasing the parameters, I may risk overfitting, but it improved the accuracy but increased the time to train
#changing the layers to be more encompassing led to a better model, but it took significantly more time
#it was interesting to play with the hyperparameters 


#  <font color=magenta>__✧Model2 Comments✧__</font>

# In[66]:


#Image augmentation added to the model robustness by diversifying the training dataset
#and it avoided overfitting. It improved overall performance in image classification and accuraccy


#  <font color=magenta>__✧Model3 Comments✧__</font>

# In[67]:


#At first I included more regularization and edited the hyper parameters, and that led toan accuracy of 50%! 
#it was terrible and the graph indicated heavy overfitting
#similar in parameter size to the baseline, and it took the same amount of time as model1
#i think even though i added another layer, it was not as accurate as model1 because of the # of parameters


# ## <center><font color=maroon>Remember to turn in both the notebook and the pdf version.</font></center>
