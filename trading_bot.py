# I was too lazy to start work on the actual nn today but i coded the basic actions in as well as a test case
# i don't think i have enough data to get a good accuracy/profit however this is just a test with open source data

import pandas as pd
import numpy as np
import openpyxl as pyxl
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load in my data
trn_data = pd.read_csv("/Users/ianrichthammer/Library/Containers/com.microsoft.Excel/Data/Downloads/mega.csv")
tst_data = pd.read_excel("/Users/ianrichthammer/Library/Containers/com.microsoft.Excel/Data/Downloads/VOO_Test.xlsx")

# train them
x_train = pd.get_dummies(trn_data.drop('Next Open', axis=1))
x_test = pd.get_dummies(tst_data.drop('Next Open', axis=1))

x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)

# check check check
x_train.fillna(x_train.mean(), inplace=True)
x_test.fillna(x_test.mean(), inplace=True)

# data prep
y_train = trn_data['Next Open']
y_test = tst_data['Next Open']

# standardize data
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)
y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

# pytorch ez
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# nn work
class RegressionNN(nn.Module):
    def __init__(self):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(x_train.shape[1], 50)  # Input layer with number of features, hidden layer with 50 neurons
        self.fc2 = nn.Linear(50, 1)  # Output layer with 1 neuron (float output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation function to hidden layer
        x = self.fc2(x)  # Output layer (linear activation by default)
        return x


# loss & op
model = RegressionNN()
criterion = nn.MSELoss() 
optimizer = optim.SGD(model.parameters(), lr=0.01)

# train
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x_train_tensor).squeeze()  # Forward pass
    loss = criterion(output, y_train_tensor)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    y_pred = model(x_test_tensor).squeeze()
    y_pred = scaler_y.inverse_transform(y_pred.numpy().reshape(-1, 1)).flatten()
    y_test = scaler_y.inverse_transform(y_test_tensor.numpy().reshape(-1, 1)).flatten()
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error on test data: {mse:.4f}')

    # Plotting results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True Values vs. Predictions')
    plt.show()
