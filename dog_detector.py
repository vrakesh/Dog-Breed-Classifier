from keras.preprocessing import image
from tqdm import tqdm
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from data_loader import *
import cv2
from glob import glob

from extract_bottleneck_features import *
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

class dog_breed_detector(object):
    """Using resnet50 detect the breed of the dog"""

    def __init__(self):

        ###Obtain bottleneck features from another pre-trained CNN.
        self.bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
        self.train_Resnet50 = self.bottleneck_features['train']
        self.valid_Resnet50 = self.bottleneck_features['valid']
        self.test_Resnet50 = self.bottleneck_features['test']
        self.train_targets = train_targets
        self.test_targets = test_targets
        self.valid_targets = valid_targets
        self.model = None
    def path_to_tensor(self, img_path):
        # loads RGB image as PIL.Image.Image type
        img = image.load_img(img_path, target_size=(224, 224))
        # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
        x = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
        return np.expand_dims(x, axis=0)

    def paths_to_tensor(self,img_paths):
        list_of_tensors = [self.path_to_tensor(img_path) for img_path in tqdm(img_paths)]
        return np.vstack(list_of_tensors)

    def ResNet50_predict_labels(self,img_path):
        # returns prediction vector for image located at img_path
        ResNet50_model = ResNet50(weights='imagenet')
        img = preprocess_input(self.path_to_tensor(img_path))
        return np.argmax(ResNet50_model.predict(img))

    def dog_detector(self,img_path):
        print(img_path)
        print("that was img path")
        prediction = self.ResNet50_predict_labels(img_path)
        return ((prediction <= 268) & (prediction >= 151))

    def dog_model(self):
        Resnet50_model = Sequential()
        Resnet50_model.add(GlobalAveragePooling2D(input_shape=self.train_Resnet50.shape[1:]))
        Resnet50_model.add(Dense(133, activation='softmax'))
        Resnet50_model.summary()
        Resnet50_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        self.model = Resnet50_model

    def train(self):
        """Train the model"""
        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Resnet50.hdf5',
                               verbose=1, save_best_only=True)

        self.model.fit(self.train_Resnet50, self.train_targets,
            validation_data=(self.valid_Resnet50, self.valid_targets),
            epochs=21, batch_size=20, callbacks=[checkpointer], verbose=1)
        # get index of predicted dog breed for each image in test set
        Resnet50_predictions = [np.argmax(self.model.predict(np.expand_dims(feature, axis=0))) for feature in self.test_Resnet50]

        # report test accuracy
        test_accuracy = 100*np.sum(np.array(Resnet50_predictions)==np.argmax(self.test_targets, axis=1))/len(Resnet50_predictions)
        print('Test accuracy: %.4f%%' % test_accuracy)

    def load_weights(self):
        self.model.load_weights('saved_models/weights.best.Resnet50.hdf5')

    def Resnet50_predict_breed(self, img_path):
        # extract bottleneck features
        bottleneck_feature = extract_Resnet50(self.path_to_tensor(img_path))
        # obtain predicted vector
        predicted_vector = self.model.predict(bottleneck_feature)
        # return dog breed that is predicted by the model
        return dog_names[np.argmax(predicted_vector)]

df = dog_breed_detector()
