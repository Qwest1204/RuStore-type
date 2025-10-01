from model.main_model import Classifier
from data.dataset import RuStoreDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

if __name__ == '__main__':
    train = pd.read_csv("data/train.tsv", sep='\t').dropna()

    dataset = RuStoreDataset(train)
    batch_size = 1  # Adjust as needed
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    input_dim = 2304  # As per the model
    num_classes = 394  # As per the model

    model = Classifier(input_dim=input_dim, num_classes=num_classes)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adjust learning rate as needed
    # Optional: Scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Training parameters
    num_epochs = 10  # Adjust as needed

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)  # Inputs shape: (bs, 3, 768) or similar, but model reshapes
            labels = labels.to(device)  # Labels: (bs,)

            optimizer.zero_grad()

            # Get logits (without softmax, for loss)
            logits = model.get_logits(inputs)

            # Compute loss
            loss = criterion(logits, labels)

            # Backpropagation
            loss.backward()

            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(correct)
            print(running_loss)

        # Optional: Step the scheduler
        # scheduler.step()

        # Epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / len(train_loader)

        print(f"epoch_loss: {epoch_loss}, epoch_acc: {epoch_acc}")
        torch.save(model.state_dict(), f"models/{epoch}.pth")