import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import pickle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def predict(model, test_loader):
    print(len(test_loader))
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
