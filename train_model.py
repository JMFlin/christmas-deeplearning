from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
#from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions
#import random
from PIL import ImageFile

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential

#from keras.callbacks import ModelCheckpoint  
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras import optimizers
from keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)
 
        # if we are using "channels first", update the input shape
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)
        else:
            input_shape = (width, height, depth) 

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
 
        # softmax classifier
        model.add(Dense(units= 133))
        model.add(Activation("softmax"))
 
        return model

    @staticmethod
    def evaluate(history):
        plt.figure(figsize=[8,6])
        plt.plot(history.history['loss'], marker='o', linestyle='--', color = 'r', linewidth=2.0, label="train_loss")
        plt.plot(history.history['val_loss'], marker='o', linestyle='--', color = 'b', linewidth=2.0, label="val_loss")
        plt.plot(history.history['acc'], marker='o', linestyle='--', color = 'r', linewidth=2.0, label="train_acc")
        plt.plot(history.history['val_acc'], marker='o', linestyle='--', color = 'b', linewidth=2.0, label="val_acc")
        plt.legend(loc="lower left")
        plt.xlabel('Epochs ', fontsize=16)
        plt.ylabel('Loss/Accuracy', fontsize=16)
        plt.title('Training loss and accuracy',fontsize=16)
        plt.savefig('Training_loss_and_accuracy.png', bbox_inches='tight')
        plt.close()

class Load:    
    @staticmethod    
    def load_dataset(path):
        data = load_files(path)
        files = np.array(data['filenames'])
        targets = np_utils.to_categorical(np.array(data['target']), 133)
        return files, targets

    @staticmethod
    def paths_to_tensor(img_paths):
        list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors)





def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

def create_model():
    model = Sequential()
    model.add(Conv2D(16, (2,2), input_shape=(224,224,3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(16, (3,3), activation= 'relu'))
    model.add(MaxPooling2D(pool_size =(2,2)))
    model.add(Conv2D(32, (3,3), activation= 'relu'))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=64, activation= 'relu'))
    model.add(Dropout(0.35))
    model.add(Dense(units= 133, activation = 'softmax'))
    #sgd = optimizers.SGD() #lr=0.05, decay=1e-6, momentum=0.5, nesterov=True
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_evaluation_curves(history):
    # Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'], marker='o', linestyle='--', color = 'r', linewidth=2.0, label="train_loss")
    plt.plot(history.history['val_loss'], marker='o', linestyle='--', color = 'b', linewidth=2.0, label="val_loss")
    plt.plot(history.history['acc'], marker='o', linestyle='--', color = 'r', linewidth=2.0, label="train_acc")
    plt.plot(history.history['val_acc'], marker='o', linestyle='--', color = 'b', linewidth=2.0, label="val_acc")
    plt.legend(loc="lower left")
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss/Accuracy', fontsize=16)
    plt.title('Training loss and accuracy',fontsize=16)
    plt.savefig('model-validation/Training_loss_and_accuracy.png', bbox_inches='tight')
    plt.close()
  

if __name__ == '__main__':


    print("[INFO] loading images...")
    train_files, train_targets = Load.load_dataset('data\images\christmas/train')
    valid_files, valid_targets = Load.load_dataset('data\images\christmas/valid')
    test_files, test_targets = Load.load_dataset('data\images\christmas/test')

    train_tensors = Load.paths_to_tensor(train_files).astype('float32')/255
    valid_tensors = Load.paths_to_tensor(valid_files).astype('float32')/255
    test_tensors = Load.paths_to_tensor(test_files).astype('float32')/255

    print("[INFO] compiling model...")
    model = LeNet.build(224, 224, 3, 3)

    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    print("[INFO] training network...")
    history = model.fit_generator(aug.flow(train_tensors, train_targets, batch_size=30),
    validation_data=(valid_tensors, valid_targets), 
    steps_per_epoch=len(train_tensors) // 30,
    validation_steps=len(test_tensors) // 30,
    epochs=40, 
    verbose=1)

    print("[INFO] evaluating network...")
    LeNet.evaluate(history)

    #plot_evaluation_curves(history)

    print("[INFO] serializing network...")
    model.save('saved-models/christmas_model.h5')


    ### Do NOT modify the code below this line.
    #print("[INFO] loading images...")
    #train_files, train_targets = load_dataset('data\images\christmas/train')
    #valid_files, valid_targets = load_dataset('data\images\christmas/valid')
    #test_files, test_targets = load_dataset('data\images\christmas/test')

    # pre-process the data for Keras
    #train_tensors = paths_to_tensor(train_files).astype('float32')/255
    #valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
    #test_tensors = paths_to_tensor(test_files).astype('float32')/255

    #print("[INFO] compiling model...")
    #model = create_model()

    #aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    #height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    #horizontal_flip=True, fill_mode="nearest")

    #print("[INFO] training network...")
    #history = model.fit_generator(aug.flow(train_tensors, train_targets, batch_size=30),
    #validation_data=(valid_tensors, valid_targets), 
    #steps_per_epoch=len(train_tensors) // 30,
    #validation_steps=len(test_tensors) // 30,
    #epochs=35, 
    #verbose=1)

    #plot_evaluation_curves(history)

    #print("[INFO] serializing network...")
    #model.save('saved_models/christmas_model.h5')

    ### Do NOT modify the code below this line.

    #checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
    #                               verbose=1, save_best_only=True)

    #model.fit(train_tensors, train_targets, 
    #          validation_data=(valid_tensors, valid_targets),
    #          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

    

    #model.load_weights('saved_models/weights.best.from_scratch.hdf5')

    # get index of predicted dog breed for each image in test set
    #dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

    # report test accuracy
    #test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
    #print('Test accuracy: %.4f%%' % test_accuracy)