import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def predict_one_sample(model, inputs, device=DEVICE):
    """Предсказание, для одной картинки"""
    with torch.no_grad():
        inputs = inputs.to(device)
        model.eval()
        logit = model(inputs).cpu()
        probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
    return probs

def calc_f1_on_val(model, val_dataset):
    # random_characters = int(np.random.uniform(0,1000))
    # ex_img, true_label = val_dataset[random_characters]
    # probs_im = predict_one_sample(model_Alex, ex_img.unsqueeze(0))

    # idxs = list(map(int, np.random.uniform(0,1000, 200)))
    imgs = [val_dataset[id][0].unsqueeze(0) for id in range(len(val_dataset))]

    y_pred = np.argmax(predict(model, imgs), -1)
    actual_labels = [val_dataset[id][1] for id in range(len(val_dataset))]

    return f1_score(actual_labels, y_pred, average='micro')


def predict(model, test_loader):
    with torch.no_grad():
        logits = []

        for inputs in test_loader:
            inputs = inputs.to(DEVICE)
            model.eval()
            outputs = model(inputs).cpu()
            logits.append(outputs)

    probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    return probs

def predict_test_to_csv(model, test_dataset, comment=''):
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128)
    probs = predict(model, test_loader)

    label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))
    preds = label_encoder.inverse_transform(np.argmax(probs, axis=1))
    test_filenames = [path.name for path in test_dataset.files]

    # saving to csv
    my_submit = pd.read_csv("../../../../MLData/simpsons/sample_submission.csv")
    my_submit = pd.DataFrame({'Id': test_filenames, 'Expected': preds})
    my_submit.to_csv(f'predictions/{comment}.csv', index=False)
