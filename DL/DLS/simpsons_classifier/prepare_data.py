from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from torchvision import transforms
from torchvision.transforms import v2

# разные режимы датасета
DATA_MODES = ['train', 'val', 'test']
# все изображения будут масштабированы к размеру 224x224 px
RESCALE_SIZE = 224

class SimpsonsDataset(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """
    def __init__(self, files, mode, transform, rescale_size):
        super().__init__()
        # список файлов для загрузки
        self.files = sorted(files)
        # режим работы
        self.mode = mode
        self.transform = transform
        self.rescale_size = rescale_size

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        self.len_ = len(self.files)

        self.label_encoder = LabelEncoder()

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as le_dump_file:
                  pickle.dump(self.label_encoder, le_dump_file)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index):
        x = self.load_sample(self.files[index])
        x = self._prepare_sample(x)
        x = np.array(x / 255, dtype='float32')
        x = self.transform(x)
        if self.mode == 'test':
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y

    def _prepare_sample(self, image):
        image = image.resize((self.rescale_size, self.rescale_size))
        return np.array(image)

# def imshow(inp, title=None, plt_ax=plt, default=False):
#     """Imshow для тензоров"""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt_ax.imshow(inp)
#     if title is not None:
#         plt_ax.set_title(title)
#     plt_ax.grid(False)
#     plt.show()

def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform

def data_transforms_v2():
    data_transforms = {
        'train': v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(RESCALE_SIZE, RESCALE_SIZE)),
            v2.RandomHorizontalFlip(p=0.3),
            v2.RandomPerspective(distortion_scale=0.2),
            v2.RandomRotation(degrees=(-30, 30)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(RESCALE_SIZE, RESCALE_SIZE)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def data_transforms_v3():
    data_transforms = {
        'train': v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(299, 299)),
            v2.RandomHorizontalFlip(p=0.3),
            v2.RandomPerspective(distortion_scale=0.2),
            v2.RandomRotation(degrees=(-30, 30)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': v2.Compose([
            v2.ToImage(),
            v2.Resize(size=(299, 299)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def get_train_and_val_dataset(is_inception=True):
    TRAIN_DIR = Path('../../../../MLData/simpsons/train/simpsons_dataset')
    train_val_files = sorted(list(TRAIN_DIR.rglob('*.jpg')))

    train_val_labels = [path.parent.name for path in train_val_files]
    n_classes = len(np.unique(train_val_labels))

    train_files, val_files = train_test_split(train_val_files, test_size=0.10, stratify=train_val_labels)

    if is_inception:
        val_dataset = SimpsonsDataset_v2(val_files, mode='val', transform=data_transforms_v3()['val'])
        train_dataset = SimpsonsDataset_v2(train_files, mode='train', transform=data_transforms_v3()['train'])
    else:
        val_dataset = SimpsonsDataset_v2(val_files, mode='val', transform=data_transforms_v2()['val'])
        train_dataset = SimpsonsDataset_v2(train_files, mode='train', transform=data_transforms_v2()['train'])
    return train_dataset, val_dataset, n_classes


# def get_test_dataset():
#     TEST_DIR = Path('../../../../MLData/simpsons/testset/testset')
#     test_files = sorted(list(TEST_DIR.rglob('*.jpg')))
#
#     test_dataset = SimpsonsDataset_v2(test_files, mode='test', transform=data_transforms_v2()['train'], rescale_size=RESCALE_SIZE)
#     return test_dataset

def get_test_dataset(is_inception=False):
    TEST_DIR = Path('../../../../MLData/simpsons/testset/testset')
    test_files = sorted(list(TEST_DIR.rglob('*.jpg')))

    if is_inception:
        return SimpsonsDataset_v2(test_files, mode='test', transform=data_transforms_v3()['val'])
    return SimpsonsDataset_v2(test_files, mode='test', transform=data_transforms_v2()['val'])



class SimpsonsDataset_v2(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """
    def __init__(self, files, mode, transform):
        super().__init__()
        # список файлов для загрузки
        self.files = sorted(files)
        # режим работы
        self.mode = mode
        self.transform = transform

        if self.mode not in DATA_MODES:
            print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
            raise NameError

        self.len_ = len(self.files)

        self.label_encoder = LabelEncoder()

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

            with open('label_encoder.pkl', 'wb') as le_dump_file:
                  pickle.dump(self.label_encoder, le_dump_file)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index):
        x = self.load_sample(self.files[index])
        x = self.transform(x)
        if self.mode == 'test':
            return x
        else:
            label = self.labels[index]
            label_id = self.label_encoder.transform([label])
            y = label_id.item()
            return x, y