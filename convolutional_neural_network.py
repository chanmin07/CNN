# Convolutional Neural Network

# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__

# Part 1 - Data Preprocessing

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,    #0~1: feature scaling - copy from templates in keras - keras API - Data Preprocessing - Image Data Preprocessing
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True) 
    # to avoid overfitting, apply transformation: geometric transformations - Image Argumentation
    # Check keras in google - keras API - Data Preprocessing - Image Data Preprocessing - for Image Argumentations
    # Example of using .flow_from_directory(directory):

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),    #size of input image to be used in CNN 
                                                 batch_size = 32,           #한번의 처리에 사용한 입력영상의 수 
                                                 class_mode = 'binary')     #cat or dog: 0 or 1
    #Check keras in google - keras API - Data Preprocessing - Image Data Preprocessing - for Image Argumentations
    
# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)                     #No Image Argumentation
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),     
                                            batch_size = 32,            
                                            class_mode = 'binary')      

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    # number of feature detectors, use relu unless the level is final step
    # input_shape is needed only in the inpoput layer
    
# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    #strides = 다음 풀링을 위해 얼마나 멀리 움직이니는가 

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    # units = number of hidden neurons

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)
    # It took about 10 ~ 15 minutes, 89.09% accuracy

# Part 4 - Making a single prediction

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)   #corresponding to batch number = 32
result = cnn.predict(test_image)
training_set.class_indices          # Out[7]: {'cats': 0, 'dogs': 1}
if result[0][0] == 1:               # result is also batch
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)

test_image = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices  
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)

test_image = image.load_img('dataset/single_prediction/cat_or_dog_3.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices  
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)

test_image = image.load_img('dataset/single_prediction/cat_or_dog_4.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices  
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)