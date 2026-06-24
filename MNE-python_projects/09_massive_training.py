import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from eegnet_model import EEGNet

# torch.manual_seed(42)
# np.random.seed(42)
X_raw = np.load("X_massive.npy") * 1e6  # The Microvolt Scale Fix
y_raw = np.load("y_massive.npy") - 2

X = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(y_raw, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

model = EEGNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)

epochs = 50
for epoch in range(epochs):
    model.train()
    
    # We now iterate through the chunks of 16 trials
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

model.eval() 
with torch.no_grad(): 
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    
    correct = (predicted == y_test).sum().item()
    test_accuracy = correct / y_test.size(0)

print(f"Real-World Test Accuracy: {test_accuracy * 100:.2f}%")