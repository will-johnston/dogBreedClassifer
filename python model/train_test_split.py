import os
import numpy as np
import shutil


'''This script is used to split the image data within the subdirectories, for use with batch loading/training'''

validation_proportion = 0.2
train_dir = "data/train_data"
val_dir = "data/val_data"

# this should have subdirectories with a portion of the images in both dirs
for root, dirs, files in os.walk('data/images'):
    if len(dirs) == 0:
        # we are in one of the subdirectories
        root_split = root.split("/")
        train_subdir = os.path.join(train_dir, root_split[-1])
        val_subdir = os.path.join(val_dir, root_split[-1])

        os.mkdir(train_subdir)
        os.mkdir(val_subdir)

        # split files proportionally into two new subdirectories
        images = np.array(files)
        np.random.shuffle(images)
        cutoff = int(len(images) * validation_proportion)
        validation_data = images[:cutoff]
        training_data = images[cutoff:]
        print(len(training_data), len(validation_data))

        for image_name in training_data:
            path = os.path.join(root, image_name)
            shutil.copy(path, train_subdir)

        for image_name in validation_data:
            path = os.path.join(root, image_name)
            shutil.copy(path, val_subdir)
