import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def generate_kinematic_spooling_data(batch_size=32, time_steps=500):
    t = np.linspace(0, 2 * np.pi, time_steps)
    motor_1 = 15.0 * np.sin(t) 
    motor_2 = 15.0 * np.sin(t + (2 * np.pi / 3))
    motor_3 = 15.0 * np.sin(t + (4 * np.pi / 3))
    
    y_single = torch.tensor(np.column_stack((motor_1, motor_2, motor_3)), dtype=torch.float32)
    Y_target = y_single.unsqueeze(0).repeat(batch_size, 1, 1)
    
    X_input = torch.randn(batch_size, time_steps, 64) * 0.1 
    X_input[:, :, 0] += y_single[:, 0] 
    X_input[:, :, 1] += y_single[:, 1] 
    
    return X_input, Y_target

X_seq, Y_target = generate_kinematic_spooling_data()
#print("Input shape:", X_seq.shape)
#print("Target shape:", Y_target.shape)
rnn_layer = nn.RNN(64, 128, batch_first=True)
kine_head = nn.Linear(in_features=128, out_features=3)
# output, _ = rnn_layer(X_seq)
# prediction = kine_head(output)
#print("\n", prediction.shape)

loss_fn = nn.MSELoss()
optimiser = optim.Adam(list(rnn_layer.parameters())+list(kine_head.parameters()), lr=0.1)
for epoch in range(100):
    optimiser.zero_grad()
    output, _ = rnn_layer(X_seq)
    prediction = kine_head(output)
    #prediction using mse loss
    loss = loss_fn(prediction, Y_target)
    loss.backward()
    optimiser.step()
    if epoch%20 == 0 or epoch == 99:
        print(f"at epoch {epoch} the loss is {loss.item()}")

