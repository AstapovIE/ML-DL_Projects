from train import train
from prepare_data import get_train_and_val_dataset, get_test_dataset
from TL_models import SimpleCnn, AlexNet, Efficient_b2
from compute_metrics import predict_test_to_csv

import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
use_gpu = True if torch.cuda.is_available() else False


train_dataset, val_dataset, n_classes = get_train_and_val_dataset()
test_dataset = get_test_dataset()

# ----------------------------------------------------SIMPLE_CNN--------------------------------------------------------

# simple_cnn = SimpleCnn(n_classes).to(DEVICE)
# opt = torch.optim.Adam(simple_cnn.parameters())
# criterion = nn.CrossEntropyLoss()
#
#
# history, simple_cnn_trained = train(train_dataset, val_dataset, model=simple_cnn, optimizer=opt, criterion=criterion, epochs=2, batch_size=128)
#
# torch.save(simple_cnn_trained.state_dict(), 'simple_cnn.pth')
# simple_cnn_trained.load_state_dict(torch.load('simple_cnn.pth'))


# ----------------------------------------------------ALEX_NET--------------------------------------------------------

# alexnet = AlexNet(n_classes, use_gpu)
# criterion = nn.CrossEntropyLoss()
# num_epochs = 6
#
# history, alexnet_trained = train(train_dataset, val_dataset, model=alexnet.model, optimizer=alexnet.optimizer, criterion=criterion, epochs=num_epochs, batch_size=128)
#
# torch.save(alexnet_trained.state_dict(), f'alexnet_trained{num_epochs}.pth')
# alexnet_trained.load_state_dict(torch.load(f'alexnet_trained{num_epochs}.pth'))


# ----------------------------------------------------ALEX_NET--------------------------------------------------------

effb2 = AlexNet(n_classes, use_gpu)
criterion = nn.CrossEntropyLoss()
num_epochs = 6

# history, effb2_trained = train(train_dataset, val_dataset, model=effb2.model, optimizer=effb2.optimizer, criterion=criterion, epochs=num_epochs, batch_size=128)

# torch.save(effb2_trained.state_dict(), f'{effb2.name}_trained{num_epochs}.pth')
# effb2_trained.load_state_dict(torch.load(f'{effb2.name}_trained{num_epochs}.pth'))
# predict_test_to_csv(effb2_trained, test_dataset)

effb2.model.load_state_dict(torch.load(f'effb2_trained{num_epochs}.pth'))
predict_test_to_csv(effb2.model, test_dataset, f'{effb2.name}_{num_epochs}')
