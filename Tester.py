from Dao_CVData import *
from Net import *
from utils import *

class Tester:
    def __init__(self, path, model, classes):
        super()
        self.model = model
        self.path = path
        self.classes = classes
        testDao = Dao_CVData(path=path, train=False, batch_size=4, shuffle=True, num_worker=0)
        testDao.load()
        self.testloader = testDao.dataloader



    def show_n_batch_prediction(self, n):
        for i, data in enumerate(self.testloader):
            if n == 0:
                break
            images, labels = data

            images = images.float()
            labels = labels.long()

            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
            print('ground truth:', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
            print('prediction:', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
            imshow(torchvision.utils.make_grid(images))
            n -= 1

    def show_n_batch_type_specified_prediction(self, n, t):
        for i, data in enumerate(self.testloader):
            images, labels = data

            images = images.float()
            labels = labels.long()


            outputs = self.model(images)
            if n == 0:
                break
            for j, label in enumerate(labels):
                if classes[label] != t:
                    continue
                output = list(outputs[j])
                predict = output.index(max(output))
                print('ground truth:', ' '.join('%5s,  ' % t))
                print('prediction:', ' '.join('%5s  ' % classes[predict]))
                image = images[j]
                imshow(torchvision.utils.make_grid(image))
                n -= 1

    def accuracy(self):

        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:

                images, labels = data

                images = images.float()
                labels = labels.long()

                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()


                # save img
                for i, image in enumerate(images):
                    imsave(image, self.path, classes[predicted[i]])
        print('Accuracy of the network on the  test images: %d %%' % (
                100 * correct / total))

    def single_accuracy(self):
        class_correct = list(0. for i in range(6))
        class_total = list(0. for i in range(6))
        with torch.no_grad():
            for data in self.testloader:
                images, labels = data

                images = images.float()
                labels = labels.long()

                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(1):
                    # print(labels)
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(6):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))

if __name__ == '__main__':
    # trainer = Trainer(learning_rate=0.001, momentum=0.9, num_round=2, path='./data')
    # model = trainer.fit()
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # import os
    # if os.path.exists("./dataset/test/predictions"):
    #     os.remove("./dataset/test/predictions/*.jpg")
    #     os.rmdir(("./dataset/test/predictions/"))
    classes = ['Car', 'Motor', 'Background', 'Face', 'Airplane', 'Leaf']
    model = Net()
    model_PATH = './net.pth'
    model.load_state_dict(torch.load(model_PATH))
    tester = Tester(path='./dataset/test', model=model, classes=classes)
        # tester.show_n_batch_prediction(5)
        # tester.show_n_batch_type_specified_prediction(10, 'Background')
    tester.accuracy()
    tester.single_accuracy()
