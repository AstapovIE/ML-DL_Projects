from torchvision.datasets import CIFAR10

if __name__ == "__main__":
    train_dataset = CIFAR10("CIFAR10/train", download=True)
    test_dataset = CIFAR10("CIFAR10/test", download=True)

#
# def prepare_data():
#     train_dataset = CIFAR10("CIFAR10/train", download=True)
#     test_dataset = CIFAR10("CIFAR10/test", download=True)
#     return train_dataset, test_dataset