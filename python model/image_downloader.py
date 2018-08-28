import sys
import os
import re
import urllib.request as urllib


'''Add the values here to properly download from Image-Net'''
USERNAME = ''
ACCESS_KEY = ''


def main():
    id_list = get_ids()
    urls = get_urls(id_list)
    download_images(urls)


def get_ids():
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        ids = f.readlines()
    ids = [x.strip() for x in ids]
    ids = [re.sub('[^A-Za-z0-9]+', '', x) for x in ids]
    return ids


def get_urls(ids):
    url_lists = [get_image_url(x) for x in ids]
    urls = [url for sublist in url_lists for url in sublist]
    return urls


def download_images(urls):
    dir_name = sys.argv[2]
    dir_path = os.path.join('images', dir_name)
    print("downloading images to: {}.".format(dir_path))
    if not os.path.isdir(dir_path):
        print("invalid directory, make sure directory has parent directory 'images'")
        sys.exit(0)
    if len(sys.argv) == 5:
        max_images = int(sys.argv[4])
    else:
        max_images = -1
    index = 0
    fail_count = 0

    for url in urls:
        if index == max_images:
            print("Reached max image count")
            break
        try:
            image = urllib.URLopener()
            image.retrieve(url, "{0}/{1}{2}.jpg".format(dir_path, dir_name, index))
            index += 1
        except Exception:
            fail_count += 1
    print("Failed to load {0} out of {1} images".format(fail_count, fail_count + index))


def get_image_url(wnid):

        url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=' + str(wnid)
        f = urllib.urlopen(url)
        image_urls = str(f.read()).split('\\r\\n')

        # last line doesn't contain an image url
        return image_urls[:-1]


if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 5:
        print("Usage: python image_downloader.py <wordnet_ids> <relative path from 'images'> -m <max_images>")
        exit(0)
    main()