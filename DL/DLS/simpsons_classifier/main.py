from train import train
from prepare_data import get_train_and_val_dataset, get_test_dataset
from TL_models import SimpleCnn, AlexNet, Efficient_b2, ResNet18, Inception_v3
from compute_metrics import predict_test_to_csv, calc_f1_on_val

import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
use_gpu = True if torch.cuda.is_available() else False
print(DEVICE)


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
# for bigger in [True, False]:
#     alexnet = AlexNet(n_classes, use_gpu, bigger)
#     criterion = nn.CrossEntropyLoss()
#     num_epochs = 30
#
#     history, alexnet_trained = train(train_dataset, val_dataset, model=alexnet.model, optimizer=alexnet.optimizer, criterion=criterion, epochs=num_epochs, batch_size=64, is_inception=False)
#
#     torch.save(alexnet_trained.state_dict(), f'{"big_" if bigger else ''}{alexnet.name}_trained{num_epochs}.pth')
#     alexnet_trained.load_state_dict(torch.load(f'{"big_" if bigger else ''}{alexnet.name}_trained{num_epochs}.pth'))
#     predict_test_to_csv(alexnet_trained, test_dataset, f'{"big_" if bigger else ''}{alexnet.name}_{num_epochs}_val010')
#
#     print(calc_f1_on_val(alexnet_trained, val_dataset), bigger)

#
#
# ###########        F1       ############
# # alex = AlexNet(n_classes, use_gpu)
# # alex.model.load_state_dict(torch.load(f'{alex.name}_trained{3}.pth'))
# print(calc_f1_on_val(alexnet_trained, val_dataset, 200))


# ----------------------------------------------------Efficient_b2--------------------------------------------------------
# for bigger in [True, False]:
#     effb2 = Efficient_b2(n_classes, use_gpu, bigger)
#     criterion = nn.CrossEntropyLoss()
#     num_epochs = 1
#
#     history, effb2_trained = train(train_dataset, val_dataset, model=effb2.model, optimizer=effb2.optimizer, criterion=criterion, epochs=num_epochs, batch_size=64, is_inception=False)
#
#     torch.save(effb2_trained.state_dict(), f'{"big_" if bigger else ''}{effb2.name}_trained{num_epochs}.pth')
#     effb2_trained.load_state_dict(torch.load(f'{"big_" if bigger else ''}{effb2.name}_trained{num_epochs}.pth'))
#     predict_test_to_csv(effb2_trained, test_dataset, f'{"big_" if bigger else ''}{effb2.name}_{num_epochs}_val010')
#
#     print(calc_f1_on_val(effb2_trained, val_dataset), bigger)


# ----------------------------------------------------ResNet18--------------------------------------------------------

# resnet = ResNet18(n_classes, use_gpu)
# criterion = nn.CrossEntropyLoss()
# num_epochs = 1
#
# history, resnet_trained = train(train_dataset, val_dataset, model=resnet.model, optimizer=resnet.optimizer, criterion=criterion, epochs=num_epochs, batch_size=64)
#
# torch.save(resnet_trained.state_dict(), f'{resnet.name}_trained{num_epochs}.pth')
# resnet_trained.load_state_dict(torch.load(f'{resnet.name}_trained{num_epochs}.pth'))
# predict_test_to_csv(resnet_trained, test_dataset, f'{resnet.name}_{num_epochs}_val005')

# ----------------------------------------------------Inception_v3--------------------------------------------------------

# train_dataset, val_dataset, n_classes = get_train_and_val_dataset(is_inception=True)
# test_dataset = get_test_dataset(is_inception=True)
#
# for bigger in [True, False]:
#     incv3 = Inception_v3(n_classes, use_gpu, bigger)
#     criterion = nn.CrossEntropyLoss()
#     num_epochs = 15
#
#     history, incv3_trained = train(train_dataset, val_dataset, model=incv3.model, optimizer=incv3.optimizer, criterion=criterion, epochs=num_epochs, batch_size=64, is_inception=True)
#
#     torch.save(incv3_trained.state_dict(), f'{"big_" if bigger else ''}{incv3.name}_trained{num_epochs}.pth')
#     incv3_trained.load_state_dict(torch.load(f'{"big_" if bigger else ''}{incv3.name}_trained{num_epochs}.pth'))
#     predict_test_to_csv(incv3_trained, test_dataset, f'{"big_" if bigger else ''}{incv3.name}_{num_epochs}_val010')
#
#     print(calc_f1_on_val(incv3_trained, val_dataset), bigger)