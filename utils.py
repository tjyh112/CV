import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
#
# def random_int_list(start, stop, length):
#
#     assert(stop-start+1 >= length)
#
#     random_list = []
#     for i in range(length):
#         v = random.randint(start, stop)
#         while v in random_list:
#             v = random.randint(start, stop)
#         random_list.append(v)
#     return np.array(random_list)


def make_one_hot(data, class_count):

    return (np.arange(class_count) == data[:, None]).astype(np.integer)

def imshow(img):
    img = (img / 2 + 0.5) * 255  # unnormalize
    npimg = img.numpy().astype(int)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def imsave(img, path, prediction):
    img = (img / 2 + 0.5) * 255 # unnormalize
    npimg = img.numpy().astype(int)
    npimg = np.transpose(npimg, (1, 2, 0))
    result = Image.fromarray(np.uint8(npimg))
    if not os.path.exists(path+'/predictions'):
        os.mkdir(path+'/predictions')


    i = 0
    file_path = path + '/predictions/' + prediction + str(i) + '.jpg'
    while os.path.exists(file_path):
        i += 1
        file_path = path + '/predictions/' + prediction + str(i) + '.jpg'
    print('saving image')
    result.save(file_path)

def get_folders(path):

    result = []
    for entry in os.scandir(path):
        result.append(path+'/'+entry.name)
    return result

if __name__ == '__main__':
    print(get_folders('./dataset/train'))