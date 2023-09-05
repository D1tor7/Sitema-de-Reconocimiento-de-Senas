import numpy as np
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


train_features = np.empty((0,42),float)

B = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\B.txt', delimiter=' ', dtype='float')
C = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\C.txt', delimiter=' ', dtype='float')
H = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\H.txt', delimiter=' ', dtype='float')
none = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\none.txt', delimiter=' ', dtype='float')



for i in range(0, len(B)):
    train_features = np.append(train_features, [B[i,:]], axis=0)
for i in range(0, len(C)):
    train_features = np.append(train_features, [C[i,:]], axis=0)
for i in range(0, len(H)):
    train_features = np.append(train_features, [H[i,:]], axis=0)
for i in range(0, len(none)):
    train_features = np.append(train_features, [none[i,:]], axis=0)







print(train_features.shape)
train_features = torch.tensor(train_features, dtype=torch.float32)

targets = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=float)
targets = torch.tensor(targets, dtype=torch.int64) 
print(train_features.shape)
print(targets.shape)


class PyTorchMLP(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
                
            # 1° capa
            torch.nn.Linear(num_features, 25),
            torch.nn.ReLU(),

            # 2° capa
            torch.nn.Linear(25, 15),
            torch.nn.ReLU(),

            # capa de salida
            torch.nn.Linear(15, num_classes),
        )

    def forward(self, x):
        logits = self.all_layers(x)
        return logits


#Entrenamiento
torch.manual_seed(1)
model = PyTorchMLP(num_features=42, num_classes=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.05) # Stochastic gradient descent


num_epochs = 1161
i = 0
for n in range(num_epochs):    
        model = model.train()
        logits = model(train_features)
        loss = nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


#compute accuracy
model = model.eval()
correct = 0.0
total_examples = 0
    
with torch.inference_mode(): # basically the same as torch.no_grad
        logits = model(train_features)
        
predictions = torch.argmax(logits, dim=1)

compare = targets == predictions
correct += torch.sum(compare)
total_examples += len(compare)
train_acc = correct / total_examples

print(f"Train Acc {train_acc*100:.2f}%")

print(model)

torch.save(model.state_dict(), "E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\modelospickle\\modelo05BCHnone.pickle")
