from keras.models import load_model
import numpy as np
import argparse
import pickle
import cv2
from keras.applications import xception


image_size = 299


def main():
    args = parse_args()

    # load the image
    image, output = get_image(args["image"], image_size, image_size)
    image_4d = [image]

    # preprocess image
    image_4d = np.array(image_4d, np.float32) / 255

    # load model and label binarizer
    model = load_model(args["model"])
    lb = pickle.loads(open(args["labels"], "rb").read())

    # classify the input image
    print("[INFO] classifying image...")
    probability = model.predict(image_4d)[0]
    print(probability)
    index = np.argmax(probability)
    label = lb.classes_[index]

    # build the label and draw the label on the image
    label = "{}: {:.2f}%".format(label, probability[index] * 100)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # show the output image
    print("[INFO] {}".format(label))
    cv2.imshow("Output", output)
    cv2.waitKey(0)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to trained model")
    ap.add_argument("-l", "--labels", required=True,
                    help="path to label binarizer")
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    args = vars(ap.parse_args())
    return args


def get_image(path, image_height, image_width):
    image_raw = cv2.imread(path)
    image = cv2.resize(image_raw, (image_height, image_width))
    image_copy = image.copy()
    # image_prep = xception.preprocess_input(image.copy())
    return image, image_copy


if __name__ == "__main__":
    main()