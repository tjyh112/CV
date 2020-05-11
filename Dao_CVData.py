import torchvision.transforms as transforms
import torchvision

import torch
from CVData import *


class Dao_CVData():

    def __init__(self, path, train, batch_size, shuffle, num_worker):
        super()
        self.path = path
        self.train = train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_worker = num_worker
        self.transform = transforms.Compose([transforms.ToTensor(),# put value range into [0,1]
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # v = (v - mean=0.5) /std=0.5
        # self.transform = transforms.ToTensor()
        # self.transform = transforms.Compose(
        #     [
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomGrayscale(),
        #         transforms.ToTensor()])
        self.dataloader = None

    def load(self):

        self.dataset = CVData(path=self.path, transform=self.transform, train=self.train)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                                      num_workers=self.num_worker)

if __name__ == '__main__':

    train_Dao = Dao_CVData(path='./dataset', train=True, batch_size=4, shuffle=True, num_worker=0)
    train_Dao.load()
    print(train_Dao.dataloader.dataset.data.size())
    print('loaded train data')

    # dataiter = iter(train_Dao.dataloader)
    # images, labels = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
