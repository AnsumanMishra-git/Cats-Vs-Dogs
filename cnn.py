#Importing keras libraries and packages

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import MaxPooling2D

#step1 Initializing CNN
classifier = Sequential()

# step2 adding 1st Convolution layer and Pooling layer
classifier.add(Convolution2D(32,(3,3),input_shape = (64,64,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# step3 adding 2nd convolution layer and polling layer
classifier.add(Convolution2D(32,(3,3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))


#step4 Flattening the layers

classifier.add(Flatten())

#step5 Full_Connection

classifier.add(Dense(units=32,activation = 'relu'))

classifier.add(Dense(units=64,activation = 'relu'))

classifier.add(Dense(units=128,activation = 'relu'))

classifier.add(Dense(units=256,activation = 'relu'))

classifier.add(Dense(units=256,activation = 'relu'))


classifier.add(Dense(units=3,activation = 'softmax'))

#step6 Compiling CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#step7 Fitting CNN to images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, # To rescaling the image in range of [0,1]
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('train',
                                                target_size=(64,64),
                                                batch_size=12, #Total no. of batches
                                                shuffle=True,
                                                class_mode='categorical')

test_set = test_datagen.flow_from_directory('test',
                                            target_size=(64,64),
                                            batch_size=12,
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch=1461//12, # Total training images
                         epochs = 25, # Total no. of epochs
                         validation_data = test_set,
                         validation_steps= 519//12) # Total testing images



#prediction
import numpy as np
from keras.preprocessing import image
training_set.class_indices

def print_results(result) :
    if result[0][0] == 1:
      prediction = 'apple'
    elif result[0][1] == 1:
      prediction = 'banana'
    else:
      prediction = 'orange'
    
    return prediction
    
test_image=image.load_img('orange-500x500.jpg',target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print_results(result)

#step8 saving model 

#classifier.save("model.h5")



