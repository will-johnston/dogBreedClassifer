import os
import pickle
import json
import re
import numpy as np
import pandas as pd
import cv2
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.applications import xception
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Dense, Dropout

# global vars
images_per_breed = 200
# image_threshold = 200
number_breeds = 120
image_size = 299
batch_size = 500
pooling = 'avg'

model_path = "trained_model_20epochs.h5"
labels_path = "labels.pickle"
breeds_ref_file = "breed_reference.csv"
validation_path = "data/val_data"
training_path = "data/train_data"


def main():

    # X_train, X_valid, y_train, y_valid, lb = read_data()
    breeds = get_breeds()
    model = create_model()
    train_size, val_size = get_dataset_size()

    # create generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        training_path,
        target_size=(image_size, image_size),
        batch_size=32,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_path,
        target_size=(image_size, image_size),
        batch_size=32,
        class_mode='categorical')


    print("Training...")
    model.fit_generator(
        train_generator,
        steps_per_epoch=int(train_size / batch_size),
        epochs=10,
        validation_data=validation_generator,
        validation_steps=int(val_size / batch_size))

    # model.fit_generator(generate_image_label_pair(training_path, breeds),
    #                     steps_per_epoch=train_size / 10,
    #                     epochs=20,
    #                     verbose=1,
    #                     validation_data=generate_image_label_pair(validation_path, breeds),
    #                     validation_steps=val_size / 10)

    # train top layer
    # print("Training...")
    # model.fit(X_train, y_train, epochs=1, validation_data=(X_valid, y_valid), verbose=1)
    #

    # save_model_and_labels(model, lb)
    print("Saving model and class indices...")
    model.save(model_path)
    with open('class_indices.txt', 'w') as file:
        file.write(json.dumps(train_generator.class_indices))



def generate_image_label_pair(dir_path, breeds):
    counter = 0
    cont = True
    while cont:
        # go through files in directory
        for filename in os.listdir(dir_path):
            # if counter > image_threshold: cont = False
            image = get_image(dir_path + "/" + filename, image_size, image_size)
            label = "_".join(filename.split("_")[:-1])
            image_4d = [image]

            # preprocess image
            image_4d = np.array(image_4d, np.float32) / 255  # divide so pixel value is in range [0,1]

            # convert label to categorical
            y = np.zeros(number_breeds)
            index = np.where(breeds[:] == label)[0]
            y[index] = 1
            y = np.array([y], np.int)
            yield (image_4d, y)
            counter += 1


def get_dataset_size():
    train_size = 0
    val_size = 0

    for path, dirs, files in os.walk(training_path):
        train_size += len(files)

    for path, dirs, files in os.walk(validation_path):
        val_size += len(files)

    return train_size, val_size

def get_breeds():
    breeds_df = pd.read_csv(breeds_ref_file, sep=",", quotechar="\'", header=None, engine="python")
    breeds_df = breeds_df.drop(breeds_df.columns[1], axis=1)
    return breeds_df.values.flatten()


def create_model():
    # use preexisting weights from xception model
    print("Getting imagenet weights...")
    xception_model = xception.Xception(weights='imagenet', include_top=False,
                                       input_shape=(image_size, image_size, 3), pooling=pooling)

    for layer in xception_model.layers:
        layer.trainable = False

    # add an additional fully connected layer
    print("Adding final layer...")
    last_layer = xception_model.output
    dense_layer = Dense(number_breeds, activation='softmax')(last_layer)

    # compile model
    print("Compiling...")
    model = Model(inputs=xception_model.input, outputs=dense_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # callbacks = [EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
    return model


def get_image(path, image_height, image_width):
    image = cv2.imread(path)
    return cv2.resize(image, (image_height, image_width))


def format_refs(refs):
    refs = [str(re.sub('[^A-Za-z0-9\-_, ]+', '', ref)) for ref in refs]
    refs = [str(x).strip().split(",") for x in refs]
    refs = [[x[0], int(x[1])] for x in refs]
    return np.array(refs)


def get_label(filename):
    pass


def get_label(breed, refs):
    labels = np.where(refs[:, 0] == breed)
    return labels[0]



def save_model_and_labels(model, lb):
    # save model
    print("Saving model...")
    model.save(model_path)

    # save the labels
    print("Saving labels...")
    f = open(labels_path, "wb")
    f.write(pickle.dumps(lb))
    f.close()


# old way to read images, now I am using batches
def read_data():
    print("LOADING IMAGES...")

    X_img = []
    y_img = []

    # with open("breed_reference.csv") as f:
    #     refs = f.readlines()
    # refs = format_refs(refs)
    # breeds_df = pd.read_csv("breed_reference.csv", sep=",", quotechar="\'", header=None, engine="python")
    # breeds_df = breeds_df.drop(breeds_df.columns[1], axis=1)
    # breeds = breeds_df.values

    for root, dirs, files in tqdm(os.walk('data/images')):
        root_split = root.split("-")
        breed = "-".join(root_split[1:])

        for index, f in enumerate(files):
            if index < images_per_breed:
                image = get_image(root + "/" + f, image_size, image_size)
                # image_prep = xception.preprocess_input(image.copy())
                # X_img.append(image_prep)
                X_img.append(image)
                y_img.append(breed)

    # convert images to arrays, specify data types
    X = np.array(X_img, np.float32) / 255  # preprocess so that pixel values is in range [0,1]
    y_img = list(np.array(y_img).flatten())
    y = np.array(y_img)

    # binarize the labels
    lb = LabelBinarizer()
    y = lb.fit_transform(y)

    # split data randomly
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3)
    print(len(X_train), len(y_train), len(X_valid), len(y_valid))
    return X_train, X_valid, y_train, y_valid, lb


if __name__ == "__main__":
    main()