
from CVData import *
from AlexNet import *
import torch.optim as optim
from Dao_CVData import *


class Trainer:
    def __init__(self, learning_rate, batch_size, momentum, num_round, path):
        super()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_round = num_round
        self.batch_size = batch_size
        self.trainDao = Dao_CVData(path=path, train=True, batch_size=self.batch_size, shuffle=True, num_worker=0)

    def fit(self):
        net = AlexNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.trainDao.load()
        trainloader = self.trainDao.dataloader
        for epoch in range(self.num_round):
            if epoch == 1:
                print('second round')

            total_loss = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                #change data type to do forward
                inputs = inputs.float()
                labels = labels.long()


                #We then set the gradients to zero, so that we are ready for the next loop. Otherwise, our gradients
                # would record a running tally of all the operations that had happened (i.e. loss.backward()
                # adds the gradients to whatever is already stored, rather than replacing them).
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                # print statistics
                if i % 20 == 1 and i != 1 :
                    # print(net.conv1.weight)
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, total_loss/20))
                    total_loss = 0
                # print(i, loss.item())
        print('Finished Training')
        return net
if __name__ == '__main__':
    trainer = Trainer(learning_rate=0.00001, batch_size=4, momentum=0.9, num_round=2, path='./dataset/train')
    model = trainer.fit()
    model_PATH = './net.pth'
    torch.save(model.state_dict(), model_PATH)


