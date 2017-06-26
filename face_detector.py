from keras.preprocessing import image
from tqdm import tqdm
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from data_loader import *
import cv2
from PIL import ImageFile
from glob import glob

ImageFile.LOAD_TRUNCATED_IMAGES = True
class face_detector(object):

        def __init__(self):
            """initializes files from human and dog database"""
            self.train_files = train_files
            self.train_targets = train_targets
            self.test_files = test_files
            self.test_targets = test_targets

            self.dog_names = dog_names
            self.human_files = human_files
            self.train_set, self.test_set, self.valid_set = None, None, None
            self.y_train, self.y_test, self.y_valid =  None, None, None
            self.model = None

        def path_to_tensor(self, img_path):
            """Create a 4D tensor from data"""
            # loads RGB image as PIL.Image.Image type
            img = image.load_img(img_path, target_size=(224, 224))
            # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
            x = image.img_to_array(img)
            # convert 3D tensor to 4D tensor with shape (1, 126, 126, 3) and return 4D tensor
            return np.expand_dims(x, axis=0)
        def paths_to_tensor(img_paths):
            """Create a list of 4D tensors"""
            list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
            return np.vstack(list_of_tensors)

        def format_human_data(self):
            """Convert human data to tensors"""
            #Get human files
            human_tensors = self.paths_to_tensor(self.human_files[:1500]).astype('float32')/255
            #get dog_tensors, used to show which are not human faces
            dog_tensors = self.paths_to_tensor(self.train_files[:1500]).astype('float32')/255
            #print(human_tensors)
            self.train_set = np.concatenate((human_tensors[:1000],dog_tensors[:1000]),axis=0)
            self.test_set = np.concatenate((human_tensors[1000:1250],dog_tensors[1250:1500]),axis=0)
            self.valid_set = np.concatenate((human_tensors[1000:1250],dog_tensors[1250:1500]),axis=0)

            #print(train_set.shape)
            self.y_train = np.concatenate((np.ones(1000).astype('float32'),np.zeros(1000).astype('float32')),axis=0)
            self.y_test = np.concatenate((np.ones(250).astype('float32'),np.zeros(250).astype('float32')),axis=0)
            self.y_valid = np.concatenate((np.ones(250).astype('float32'),np.zeros(250).astype('float32')),axis=0)
            pass

        def face_model(self):
            """model to detect faces"""
            face_model = Sequential()
            face_model.add(Conv2D(filters=16, kernel_size=2, strides=1, padding='same', activation='relu', input_shape=(224, 224, 3)))
            face_model.add(Conv2D(filters=32, kernel_size=3    , strides=1, padding='same', activation='relu'))
            face_model.add(MaxPooling2D(pool_size=2))
            face_model.add(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))
            face_model.add(MaxPooling2D(pool_size=2))
            face_model.add(Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu'))
            face_model.add(MaxPooling2D(pool_size=2))
            face_model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
            face_model.add(MaxPooling2D(pool_size=2))
            face_model.add(Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu'))
            face_model.add(MaxPooling2D(pool_size=2))
            face_model.add(Dropout(0.2))
            face_model.add(Flatten())
            face_model.add(Dropout(0.2))
            face_model.add(Dense(256,activation='relu'))
            face_model.add(Dropout(0.2))
            face_model.add(Dense(1, activation='sigmoid'))
            face_model.summary()
            #compile_face_model
            face_model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['binary_accuracy'])
            self.model = face_model
            return self.model
        def train(self):
            """Train our model"""
            if(self.model == None):
                print("Run model method before training")
                return
            # create and configure augmented image generator
            datagen = ImageDataGenerator(
                width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
                height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
                horizontal_flip=True) # randomly flip images horizontally

            # fit augmented image generator on data
            datagen.fit(train_set)
            epochs = 25

            ### Do NOT modify the code below this line.

            checkpointer = ModelCheckpoint(filepath='saved_models/face_detect.weights.best.from_scratch.hdf5',
                               verbose=1, save_best_only=True)

            self.model.fit_generator(datagen.flow(train_set, y_train, batch_size=20),steps_per_epoch=train_set.shape[0] // 20, validation_data=(valid_set, y_valid),epochs=epochs, callbacks=[checkpointer], verbose=1)
            self.model.load_weights('saved_models/face_detect.weights.best.from_scratch.hdf5')
            score = self.model.evaluate(test_set,y_test)
            print("Accuracy", score[-1])
        def load_weights(self):
            """load weights to model"""
            self.model.load_weights('saved_models/face_detect.weights.best.from_scratch.hdf5')

# returns "True" if face is detected in image stored at img_path
def face_detect_cv2(img_path):
    img = cv2.imread(img_path)
    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
#new face detector using our defined model to predict
def face_detect(img_path):
    fd = face_detector()
    fd.face_model()
    fd.load_weights()
    tensor = fd.path_to_tensor(img_path)
    y_hat = fd.model.predict(tensor)
    return int(y_hat)
if __name__ == '__main__':
    input_files = np.array(glob("*.jpg"))
    img_path = input_files[0]
    print(img_path)
    '''fd = face_detector()
    fd.face_model()
    fd.load_weights()
    tensor = fd.path_to_tensor(img_path)
    y_hat = fd.model.predict(tensor)'''
    y_hat = face_detect_cv2(img_path)
    print(int(y_hat))
