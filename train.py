import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


def mlp_train(model, data, train_loader, learning_rate=0.001, max_epochs=1000):
    loss_function = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(max_epochs):
        batches = iter(train_loader)
        batch_train_losses = np.empty(len(batches))
        batch_train_accuracy = np.empty(len(batches))
        for i, batch in enumerate(batches):
            batch_x, batch_y = batch
            logits = model.forward(batch_x)
            loss = loss_function.forward(logits, batch_y)
            preds = torch.max(torch.softmax(logits, 1), dim=1)[1]
            accuracy = torch.mean((preds == batch_y).float())
            batch_train_losses[i] = loss
            batch_train_accuracy[i] = accuracy
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_losses.append(batch_train_losses.mean())
        train_accuracies.append(batch_train_accuracy.mean())

        with torch.no_grad():
            test_logits = model.forward(data["x_test"])
            test_loss = loss_function.forward(test_logits, data["y_test"])
            test_preds = torch.max(torch.softmax(test_logits, 1), dim=1)[1]
            accuracy = torch.mean((test_preds == data["y_test"]).float())
            test_losses.append(test_loss.item())
            test_accuracies.append(accuracy.item())

    return model, train_losses, train_accuracies, test_losses, test_accuracies
