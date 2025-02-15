from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from torchvision import transforms

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
        # для преобразования изображений в тензоры PyTorch и нормализации входа
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
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

def get_train_and_val_dataset():
    TRAIN_DIR = Path('../../../../MLData/simpsons/train/simpsons_dataset')
    train_val_files = sorted(list(TRAIN_DIR.rglob('*.jpg')))

    train_val_labels = [path.parent.name for path in train_val_files]
    n_classes = len(np.unique(train_val_labels))

    train_files, val_files = train_test_split(train_val_files, test_size=0.25, stratify=train_val_labels)

    val_dataset = SimpsonsDataset(val_files, mode='val', transform=get_transform(), rescale_size=RESCALE_SIZE)
    train_dataset = SimpsonsDataset(train_files, mode='train', transform=get_transform(), rescale_size=RESCALE_SIZE)
    return train_dataset, val_dataset, n_classes


def get_test_dataset():
    TEST_DIR = Path('../../../../MLData/simpsons/testset/testset')
    test_files = sorted(list(TEST_DIR.rglob('*.jpg')))

    test_dataset = SimpsonsDataset(test_files, mode='test', transform=get_transform(), rescale_size=RESCALE_SIZE)
    return test_dataset




# train_val_files = sorted(list(TRAIN_DIR.rglob('*.jpg')))


# train_val_labels = [path.parent.name for path in train_val_files]
# train_files, val_files = train_test_split(train_val_files, test_size=0.15, stratify=train_val_labels)

# transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])




# for a in val_dataset:
#     print(a[0].shape)
#     break
#
# fig, ax = plt.subplots(nrows=3, ncols=3,figsize=(8, 8),sharey=True, sharex=True)
# for fig_x in ax.flatten():
#     random_characters = int(np.random.uniform(0,1000))
#     im_val, label = val_dataset[random_characters]
#     img_label = " ".join(map(lambda x: x.capitalize(),val_dataset.label_encoder.inverse_transform([label])[0].split('_')))
#     imshow(im_val.data.cpu(),title=img_label,plt_ax=fig_x)


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