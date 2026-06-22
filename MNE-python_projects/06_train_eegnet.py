import torch
import torch.nn as nn
import torch.optim as optim
from eegnet_model import EEGNet
from eegnet_model import X
from eegnet_model import y

model = EEGNet()
loss_fn = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=0.005)

for epoch in range(50):
    model.train()
    optimiser.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimiser.step()
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y).sum().item()
    accuracy = correct / y.size(0)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/50] | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}")

