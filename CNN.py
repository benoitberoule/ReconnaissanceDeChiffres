#Construction du CNN
import numpy as np

#imports
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#Initialiser CNN
classifier = Sequential()


#Etape 1 - Convolution
classifier.add(Convolution2D(filters=32
                             ,kernel_size=3
                             ,strides=1
                             ,input_shape=(28,28,3)
                             ,activation="relu"))
classifier.add(Dropout(rate=0.1))

#Etape 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Ajout d'un couche de convolution
classifier.add(Convolution2D(filters=32
                             ,kernel_size=3
                             ,strides=1
                             ,activation="relu"))
classifier.add(Dropout(rate=0.1))

classifier.add(MaxPooling2D(pool_size=(2,2)))

#Etape 3 - Flattening
classifier.add(Flatten())

#Etape 4 - Couche complète connectée
classifier.add(Dense(units=128,activation="relu"))
classifier.add(Dense(units=1,activation="sigmoid"))

#Compilation
classifier.compile(optimizer="adam", loss="binary_crossentropy",
                   metrics=["accuracy"])

#Entrainer le reseaux
from keras.preprocessing.image import ImageDataGenerator
import idx2numpy


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = idx2numpy.convert_from_file("train-images-idx3-ubyte")
train_generator = np.expand_dims(train_generator,axis=3)
train_generator = np.append(np.append(train_generator,train_generator,axis = 3),train_generator,axis = 3)

labels = idx2numpy.convert_from_file("train-labels-idx1-ubyte")

toto = np.array([[1,2],[3,4]])
titi = np.array([toto,np.array([5,6,7])])

validation_generator = idx2numpy.convert_from_file("t10k-images-idx3-ubyte")
validation_generator = np.expand_dims(validation_generator, axis=3)
validation_generator = np.append(np.append(validation_generator,validation_generator,axis = 3),validation_generator,axis = 3)

validation_generator = (validation_generator,idx2numpy.convert_from_file("t10k-labels-idx1-ubyte"))


# train_generator = train_datagen.flow_from_directory(
#         'dataset/training_set',
#         target_size=(128, 128),
#         batch_size=32,
#         class_mode='binary')

# validation_generator = test_datagen.flow_from_directory(
#         'dataset/test_set',
#         target_size=(128, 128),
#         batch_size=32,
#         class_mode='binary')

classifier.fit(
        x=train_generator,y = idx2numpy.convert_from_file("train-labels-idx1-ubyte"),
        #steps_per_epoch=250,
        epochs=10,
        validation_data=validation_generator,
        batch_size=32)

test1 = validation_generator[0][0]
soluce = classifier.predict(validation_generator[0])

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

for i in range(1,100):
    image = load_img("dataset/training_set/cats/cat."+str(i)+".jpg",target_size=(128,128))
    input_arr = img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    if classifier.predict(input_arr)[0][0] == 0 :
        print("dichat")
    else :
        print("didou")



