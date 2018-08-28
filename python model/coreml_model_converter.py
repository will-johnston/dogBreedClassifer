from keras.models import load_model
import coremltools
import argparse
import json

# parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to model in h5 format")
ap.add_argument("-l", "--labels", required=True, help="path to labels text file")
args = vars(ap.parse_args())

# load model and labels
model = load_model(args["model"])
labels_dict = json.load(open(args["labels"]))
labels = list(labels_dict.keys())

# convert to coreml model
print("Converting model...")
coreml_model = coremltools.converters.keras.convert(
    model,
    input_names="image",
    image_input_names="image",
    image_scale=1/255.0,
    class_labels=labels,
    is_bgr=True)

# save the model
print("Saving model...")
output = "dogBreedClassifier_2.mlmodel"
coreml_model.save(output)

