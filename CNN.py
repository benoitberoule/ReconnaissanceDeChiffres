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
classifier.add(Dense(units=10,activation="sigmoid"))

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

#Création des labels
labelsBruts = idx2numpy.convert_from_file("train-labels-idx1-ubyte")
labels = np.zeros((len(train_generator),10))

for i in range(0,len(train_generator)):
        labels[i][labelsBruts[i]] = 1
labels = labels.astype(int)

validation_generator = idx2numpy.convert_from_file("t10k-images-idx3-ubyte")
validation_generator = np.expand_dims(validation_generator, axis=3)
validation_generator = np.append(np.append(validation_generator,validation_generator,axis = 3),validation_generator,axis = 3)

labelsBrutsValidation = idx2numpy.convert_from_file("t10k-labels-idx1-ubyte")
labelsValidation = np.zeros((len(validation_generator),10))

for i in range(0,len(validation_generator)):
        labelsValidation[i][labelsBrutsValidation[i]] = 1
labelsValidation = labelsValidation.astype(int)


validation_generator = (validation_generator,labelsValidation)




classifier.fit(
        x=train_generator,y = labels,
        #steps_per_epoch=250,
        epochs=1,
        validation_data=validation_generator,
        batch_size=32)

#Vérification
res = np.zeros((len(validation_generator[0]),3))
correct = 0
pred = classifier.predict(validation_generator[0])
for i in range(0,len(validation_generator[0])):
    res[i][0] = labelsBrutsValidation[i]
    argMax = -1
    maxi = -1
    for j in range(0,10):
        if pred[i][j] > maxi:
            maxi = pred[i][j]
            argMax = j
    res[i][1] = argMax
    if res[i][0] == res[i][1]:
        res[i][2] = 1
        correct += 1
correct /= len(validation_generator[0])
        

# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array

# for i in range(1,100):
#     image = load_img("dataset/training_set/cats/cat."+str(i)+".jpg",target_size=(128,128))
#     input_arr = img_to_array(image)
#     input_arr = np.array([input_arr])  # Convert single image to a batch.
#     if classifier.predict(input_arr)[0][0] == 0 :
#         print("dichat")
#     else :
#         print("didou")



