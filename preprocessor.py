import cv2
from skimage.io import imread_collection
from PIL import Image

def resize(path):

    collections = imread_collection(path)
    for file in collections.files:
        img = cv2.imread(file)
        arr = cv2.resize(img, (360, 240))
        Image.fromarray(arr).save(file)





if __name__ == '__main__':
    path = './dataset/motorbike/*.jpg'
    resize(path)