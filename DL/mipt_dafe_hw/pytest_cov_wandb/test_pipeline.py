import pytest
import torch
import os
import json
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
import torch.nn as nn

from train import compute_accuracy
# from prepare_data import prepare_data
from hparams import config

@pytest.mark.parametrize("preds,targets,result",[
    (torch.tensor([1,2,3]),torch.tensor([1,2,3]), 1.0),
    (torch.tensor([1,2,3]),torch.tensor([0,0,0]), 0.0),
    (torch.tensor([1,2,3]),torch.tensor([1,2,0]), 2/3),
    ])
def test_accuracy_parametrized(preds, targets, result):
    assert torch.allclose(compute_accuracy(preds, targets), torch.tensor([result]), rtol=0, atol=1e-5)

@pytest.fixture
def train_dataset():
    # note: реализуйте и протестируйте подготовку данных (скачивание и препроцессинг)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    train_dataset = CIFAR10(root='CIFAR10/train',
                            train=True,
                            transform=transform,
                            download=False,
                            )
    yield train_dataset

@pytest.fixture
def test_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    test_dataset = CIFAR10(root='CIFAR10/test',
                           train=False,
                           transform=transform,
                           download=False,
                           )

    yield test_dataset

def test_prepare_data_train(train_dataset):
    assert isinstance(train_dataset, CIFAR10)
    assert len(train_dataset) > 0
    assert isinstance(train_dataset[0], tuple)

def test_prepare_data_test(test_dataset):
    assert isinstance(test_dataset, CIFAR10)
    assert len(test_dataset) > 0
    assert isinstance(test_dataset[0], tuple)

# ЕСЛИ НАДО ЧТОБЫ prepare.py БЫЛ В summary pytest-cov, то раскомментить это и в самом .py метод
# def test_prepare_data():
#     train_dataset, test_dataset = prepare_data()
#     assert isinstance(train_dataset, CIFAR10)
#     assert len(train_dataset) > 0
#     assert isinstance(train_dataset[0], tuple)
#     assert isinstance(test_dataset, CIFAR10)
#     assert len(test_dataset) > 0
#     assert isinstance(test_dataset[0], tuple)


@pytest.fixture(params=[{"batch_size": 32}, {"batch_size": 64}])
def hyperparameters(request):
    params = {
        "batch_size": request.param["batch_size"],
        "epochs": 2,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "zero_init_residual": False
    }
    yield params


def test_loss_decrease_on_first_batch(hyperparameters, train_dataset):
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet18(pretrained=False, num_classes=10, zero_init_residual=hyperparameters["zero_init_residual"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters["learning_rate"], weight_decay=hyperparameters["weight_decay"])

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        first_loss = loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        second_loss = loss.item()

        assert second_loss < first_loss
        break

@pytest.mark.parametrize(["device"], [["cpu"], ["cuda"]])
def test_train_on_one_batch(train_dataset, device):
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    device = torch.device(device)

    model = resnet18(pretrained=False, num_classes=10, zero_init_residual=config["zero_init_residual"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        preds = torch.argmax(outputs, 1)
        accuracy = compute_accuracy(preds, labels)

        assert loss.item() > 0, "Ошибка: потери должны быть положительными"
        assert 0 <= accuracy.item() <= 1, "Ошибка: точность должна быть в диапазоне [0, 1]"
        break  # Достаточно одного батча для проверки

def test_training():
    os.system("python train.py")  # Запускаем полный процесс обучения

    assert os.path.exists("model.pt"), "Ошибка: файл model.pt не был сохранён"
    assert os.path.exists("run_id.txt"), "Ошибка: run_id.txt не был сохранён"

    os.system("python compute_metrics.py")  # Запускаем валидацию модели

    assert os.path.exists("final_metrics.json"), "Ошибка: файл с финальными метриками отсутствует"

    with open("final_metrics.json", "r") as f:
        metrics = json.load(f)
        assert "accuracy" in metrics, "Ошибка: метрика accuracy отсутствует в файле метрик"
        assert 0 <= metrics["accuracy"] <= 1, "Ошибка: значение точности некорректно"
