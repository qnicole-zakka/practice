'''
PyTorch Hello-world Level Problem
- How to train a model in PyTorch
- How to evaluate model performance
= How to save and load models
'''

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


class MyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    

def get_dataset():
    X = torch.rand((1000, 16))
    y = 2 * X
    return X, y


def train():
    model = MyNN()
    model.to('cuda')
    model.train()
    X, y = get_dataset()
    X.to('cuda')
    y.to('cuda')
    EPOCHS = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    criterion = torch.nn.MSELoss(reduction='mean')

    for e in range(EPOCHS):
        optimizer.zero_grad()
        y_pred = model(X)
        y_pred = y_pred.reshape(1000) 
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'v0.h5')


def test():
    model = MyNN()
    model.load_state_dict(torch.load('v0.h5'))
    model.eval()
    X, y = get_dataset()

    with torch.no_grad():
        y_pred = model(X)
        print(accuracy_score(y_pred=y_pred, y_true=y))


train()
test()
        



