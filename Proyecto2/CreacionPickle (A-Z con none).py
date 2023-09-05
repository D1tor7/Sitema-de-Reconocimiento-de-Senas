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
A = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\A.txt', delimiter=' ', dtype='float')
B = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\B.txt', delimiter=' ', dtype='float')
C = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\C.txt', delimiter=' ', dtype='float')
D = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\D.txt', delimiter=' ', dtype='float')
E = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\E.txt', delimiter=' ', dtype='float')
F = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\F.txt', delimiter=' ', dtype='float')
G = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\G.txt', delimiter=' ', dtype='float')
H = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\H.txt', delimiter=' ', dtype='float')
I = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\I.txt', delimiter=' ', dtype='float')
K = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\K.txt', delimiter=' ', dtype='float')
L = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\L.txt', delimiter=' ', dtype='float')
M = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\M.txt', delimiter=' ', dtype='float')
N = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\N.txt', delimiter=' ', dtype='float')
none = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\none.txt', delimiter=' ', dtype='float')
O = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\O.txt', delimiter=' ', dtype='float')
P = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\P.txt', delimiter=' ', dtype='float')
Q = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\Q.txt', delimiter=' ', dtype='float')
R = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\R.txt', delimiter=' ', dtype='float')
T = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\T.txt', delimiter=' ', dtype='float')
U = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\U.txt', delimiter=' ', dtype='float')
V = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\V.txt', delimiter=' ', dtype='float')
W = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\W.txt', delimiter=' ', dtype='float')
X = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\X.txt', delimiter=' ', dtype='float')
Y = np.loadtxt('E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\LETRAS\\Y.txt', delimiter=' ', dtype='float')

for i in range(0, len(A)):
    train_features = np.append(train_features, [A[i,:]], axis=0)
for i in range(0, len(B)):
    train_features = np.append(train_features, [B[i,:]], axis=0)
for i in range(0, len(C)):
    train_features = np.append(train_features, [C[i,:]], axis=0)
for i in range(0, len(D)):
    train_features = np.append(train_features, [D[i,:]], axis=0)
for i in range(0, len(E)):
    train_features = np.append(train_features, [E[i,:]], axis=0)
for i in range(0, len(F)):
    train_features = np.append(train_features, [F[i,:]], axis=0)
for i in range(0, len(G)):
    train_features = np.append(train_features, [G[i,:]], axis=0)
for i in range(0, len(H)):
    train_features = np.append(train_features, [H[i,:]], axis=0)
for i in range(0, len(I)):
    train_features = np.append(train_features, [I[i,:]], axis=0)
for i in range(0, len(K)):
    train_features = np.append(train_features, [K[i, :]], axis=0)
for i in range(0, len(L)):
    train_features = np.append(train_features, [L[i, :]], axis=0)
for i in range(0, len(M)):
    train_features = np.append(train_features, [M[i, :]], axis=0)
for i in range(0, len(N)):
    train_features = np.append(train_features, [N[i, :]], axis=0)
for i in range(0, len(none)):
    train_features = np.append(train_features, [none[i, :]], axis=0)
for i in range(0, len(O)):
    train_features = np.append(train_features, [O[i, :]], axis=0)
for i in range(0, len(P)):
    train_features = np.append(train_features, [P[i, :]], axis=0)
for i in range(0, len(Q)):
    train_features = np.append(train_features, [Q[i, :]], axis=0)
for i in range(0, len(R)):
    train_features = np.append(train_features, [R[i, :]], axis=0)
for i in range(0, len(T)):
    train_features = np.append(train_features, [T[i, :]], axis=0)
for i in range(0, len(U)):
    train_features = np.append(train_features, [U[i, :]], axis=0)
for i in range(0, len(V)):
    train_features = np.append(train_features, [V[i, :]], axis=0)
for i in range(0, len(W)):
    train_features = np.append(train_features, [W[i, :]], axis=0)
for i in range(0, len(X)):
    train_features = np.append(train_features, [X[i, :]], axis=0)
for i in range(0, len(Y)):
    train_features = np.append(train_features, [Y[i, :]], axis=0)








print(train_features.shape)
train_features = torch.tensor(train_features, dtype=torch.float32)

targets = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23], dtype=float)
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
model = PyTorchMLP(num_features=42, num_classes=24)
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

torch.save(model.state_dict(), "E:\\User\\Escritorio\\Sistemas_Conversacionales\\Semana16\\modelospickle\\modelo04withnone.pickle")
