import os

for root, dirs, files in os.walk('data/images'):
    root_split = root.split("-")
    breed = "-".join(root_split[1:])
    breed_path = os.path.join('data/images', breed)

    print(root, breed_path)
    os.rename(root, breed_path)

