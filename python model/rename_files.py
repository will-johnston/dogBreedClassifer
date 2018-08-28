import os

'''Used to convert image names in Standford data set where breed is in filename instead of wordnet id'''

for root, dirs, files in os.walk('data/images'):
    root_split = root.split("-")
    breed = "-".join(root_split[1:])
    print(breed)
    for index, f in enumerate(files):
        file_split = f.split("_")
        filename = breed + "_" + file_split[1]
        src = root + "/" + f
        dst = root + "/" + filename
        os.rename(src, dst)