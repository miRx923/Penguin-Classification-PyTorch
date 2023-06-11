# Importing the libraries

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


device = ("cuda" if torch.cuda.is_available() else "cpu")
    

# Hyperparameters

num_epochs = 15000
learning_rate = 0.0001

data = pd.read_csv("penguins.csv")

# Classes count in dataset
sns.countplot(x = 'species', data=data)
plt.show()

print(f"Dataset:\n\n{data}")

# set plot style
sns.set(style="ticks")
sns.set_palette("husl")

# create plots over all dataset; for subset use iloc indexing
sns.pairplot(data, hue="species")

# display plots using matplotlib
plt.show()

# Data preprocessing

data = pd.read_csv("penguins.csv")

data_cleaned = data.dropna()
data_cleaned = data_cleaned.fillna(data_cleaned.mean(numeric_only=True))

print(f"Fixed data:\n\n{data_cleaned}")

# split data into input (X - select the first four columns) and output (y - select last column)

data_encoded = pd.get_dummies(data_cleaned, columns=["species", "island", "sex"])

X = data_encoded.drop(columns=["species_Adelie", "species_Chinstrap", "species_Gentoo"], axis=1)
Y = data_encoded[["species_Adelie", "species_Chinstrap", "species_Gentoo"]]

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Convert input and output data to tensors

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)  # Use dtype=torch.long for CrossEntropyLoss
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

model = nn.Sequential(
    nn.Linear(9, 36),
    nn.ReLU(),
    nn.Linear(36, 64),
    nn.ReLU(),
    nn.Linear(64, 3)
)

print(model)

# Define the optimizer and loss function

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Initialize empty lists to store the loss and accuracy values
train_losses = []
train_accs = []
test_losses = []
test_accs = []

for epoch in range(num_epochs + 1):
    # Training
    preds = model(X_train_tensor)
    y_train_np = y_train.values  # Convert y_train to NumPy array
    y_train_tensor_float = torch.from_numpy(y_train_np).float()  # Convert NumPy array to tensor
    train_loss = criterion(preds, y_train_tensor_float)
    train_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    train_accuracy = (y_train_tensor_float.argmax(-1) == preds.argmax(-1)).sum().item() / len(y_train)

    # Testing
    with torch.no_grad():  # No need to compute gradients during evaluation
        preds = model(X_test_tensor)
        y_test_np = y_test.values  # Convert y_test to NumPy array
        y_test_tensor_float = torch.from_numpy(y_test_np).float()  # Convert NumPy array to tensor
        test_loss = criterion(preds, y_test_tensor_float)
        test_accuracy = (y_test_tensor_float.argmax(-1) == preds.argmax(-1)).sum().item() / len(y_test)

    # Store loss and accuracy values
    train_losses.append(train_loss.item())
    train_accs.append(train_accuracy)
    test_losses.append(test_loss.item())
    test_accs.append(test_accuracy)

    if epoch % 1000 == 0:
        print(f"Epoch: {epoch} Train Loss: {train_loss.item():.4f} Train Acc: {train_accuracy:.4f} "
              f"Test Loss: {test_loss.item():.4f} Test Acc: {test_accuracy:.4f}")
        
y_pred = model(X_test_tensor)  # Assuming X_test_tensor is the test data tensor

y_test_class = y_test_tensor.argmax(dim=1)  # Convert y_test_tensor to class indices
y_pred_class = y_pred.argmax(dim=1)  # Convert y_pred to class indices

# Convert tensors to numpy arrays
y_test_class = y_test_class.detach().numpy()
y_pred_class = y_pred_class.detach().numpy()


print(f"Confusion matrix:\n\n{confusion_matrix(y_test_class, y_pred_class)}")

print(classification_report(y_test_class, y_pred_class))

# Loss plot
plt.plot(train_losses, label="Training loss")
plt.plot(test_losses, label="Testing loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
# Set the y-axis range
plt.ylim([0, 1])
plt.show()

# Accuracy plot
plt.plot(train_accs, label="Training accuracy")
plt.plot(test_accs, label="Testing accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


#torch.save(model.state_dict(), 'penguinModel.pth')

# Load the model with this code if needed:
#model.load_state_dict(torch.load('penguinModel.pth'))