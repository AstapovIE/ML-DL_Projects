import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models

# TODO add scheduler

class AlexNet():
    def __init__(self, n_classes, use_gpu, bigger):
        # num_features -- это размерность вектора фич, поступающего на вход FC-слою
        in_features = 9216
        model = models.alexnet(weights='AlexNet_Weights.DEFAULT')

        if bigger:
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, n_classes)  # Финальный слой без активации (она будет внутри loss)
            )
        else:
            model.classifier = nn.Linear(in_features, n_classes)


        # model.classifier = nn.Linear(num_features, n_classes)
        if use_gpu:
            model = model.cuda()
        # Обучаем только классификатор
        self.optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)
        self.model = model
        self.name = 'AlexNet'

class ResNet18():
    def __init__(self, n_classes, use_gpu):
        # num_features -- это размерность вектора фич, поступающего на вход FC-слою
        model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        print(model.fc.in_features)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
        if use_gpu:
            model = model.cuda()
        # Обучаем только классификатор
        self.optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)
        self.model = model
        self.name = 'ResNet50'

class Efficient_b2():
    def __init__(self, n_classes, use_gpu, bigger=False):
        model = models.efficientnet_b2(weights='EfficientNet_B2_Weights.DEFAULT')
        for param in model.features.parameters():
            param.requires_grad = False

        in_features = model.classifier[1].in_features

        if bigger:
            model.classifier = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, n_classes)  # Финальный слой без активации (она будет внутри loss)
            )
        else:
            model.classifier = nn.Linear(model.classifier[1].in_features, n_classes)
        if use_gpu:
            model = model.cuda()

        # Обучаем только классификатор
        self.optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)
        self.model = model
        self.name = 'EfficientNetB2'

class Inception_v3():
    def __init__(self, n_classes, use_gpu, bigger=False):
        model = models.inception_v3(weights='DEFAULT')
        for param in model.parameters():
            param.requires_grad = False

        in_features = model.fc.in_features
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, n_classes)

        if bigger:
            model.fc = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, n_classes)  # Финальный слой без активации (она будет внутри loss)
            )
        else:
            model.fc = nn.Linear(in_features, n_classes)
        if use_gpu:
            model = model.cuda()

        # Обучаем только классификатор
        self.optimizer = torch.optim.Adam(params=[
            {"params": model.AuxLogits.fc.parameters(), "lr": 1e-4},
            {"params": model.fc.parameters(), "lr": 1e-4}
        ])
        model.aux_logits = True
        self.model = model
        self.name = 'Inception_v3'


# Очень простая сеть
class SimpleCnn(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.out = nn.Linear(96 * 5 * 5, n_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)
        logits = self.out(x)
        return logits