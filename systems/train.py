import torch
from torch import nn, optim
import numpy as np

from models.GRUNet import GRUNet
from utils.preprocessor import Preprocessor


BATCH_SIZE = 128
SEQ_LEN = 7
PRED = 1


def train(epochs=100):
    columns = ['open', 'close', 'low', 'high', 'volume']
    targets = ['close']
    pp = Preprocessor()
    train_loader, val_loader, test_loader = pp.preprocess(columns, targets, SEQ_LEN, PRED, BATCH_SIZE)

    model = GRUNet(input_dim=len(columns), hidden_dim=200, output_dim=PRED, gru_layers=10)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    train_preds = []
    val_preds = []

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        print('----------')
        print(f'Epoch: {epoch+1}/{epochs}')

        train_loss = 0
        val_loss = 0

        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            preds = model(inputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            train_preds.extend(preds.detach().numpy())
            train_loss += loss.item()
        batch_train_loss = train_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                preds = model(inputs)
                loss = criterion(preds, labels)

                val_preds.extend(preds.numpy())
                val_loss += loss.item()
        batch_val_loss = val_loss / len(val_loader)

        train_losses.append(batch_train_loss)
        val_losses.append(batch_val_loss)
        print(f'Train loss: {batch_train_loss:.4}, Validation loss: {batch_val_loss:.4}')
    train_preds = np.array(train_preds)
    val_preds = np.array(val_preds)

"""

# test
test_preds = []
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        preds = model(inputs)
        test_preds.extend(preds.numpy())
test_preds = np.array(test_preds)

# plot training
import matplotlib.pyplot as plt

plt.figure()
plt.title('Train Loss - Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(range(1, epochs + 1), train_losses, color='blue', linestyle='-', label='Train Loss')
plt.plot(range(1, epochs + 1), val_losses, color='red', linestyle='--', label='Validation Loss')
plt.legend()
plt.show()


# plot predictions
ft = 0  # 0 = open, 1 = close, 2 = highest, 3 = lowest

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)

# plt.plot(np.arange(y_train.shape[0]), y_train[:ft], color='blue', label='train target')
#
# plt.plot(np.arange(y_train.shape[0], y_train.shape[0] + y_val.shape[0]), y_val[:ft], color='gray', label='valid target')
#
# plt.plot(np.arange(y_train.shape[0] + y_val.shape[0], y_train.shape[0] + y_test.shape[0] + y_test.shape[0]),
#          y_test[:ft], color='black', label='test target')
#
# plt.plot(np.arange(len(train_preds)), train_preds[:ft], color='red', label='train prediction')
#
# plt.plot(np.arange(len(train_preds), len(train_preds) + len(val_preds)),
#          val_preds[:ft], color='orange', label='valid prediction')
#
# plt.plot(np.arange(len(train_preds) + len(val_preds),
#                    len(train_preds) + len(val_preds) + len(test_preds)),
#          test_preds[:ft], color='green', label='test prediction')
#
# plt.title('past and future stock prices')
# plt.xlabel('time [days]')
# plt.ylabel('normalized price')
# plt.legend(loc='best')

plt.subplot(1, 2, 2)
# plt.plot(np.arange(y_train.shape[0], y_train.shape[0] + y_test.shape[0]),
#          y_test[:, ft], color='black', label='test target')
#
# plt.plot(np.arange(y_train.shape[0], y_train.shape[0] + y_test.shape[0]),
#          test_preds[:, ft], color='green', label='test prediction')

plt.title('future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best')

plt.show()

# corr_price_development_train =\
#     np.sum(np.equal(np.sign(y_train[:1]-y_train[:0]), np.sign(y_train_pred[:1]-y_train_pred[:, 0]))
#            .astype(int)) / y_train.shape[0]
# corr_price_development_valid = \
#     np.sum(np.equal(np.sign(y_val[:1]-y_val[:0]), np.sign(y_val_pred[:1]-y_val_pred[:, 0]))
#            .astype(int)) / y_val.shape[0]
# corr_price_development_test =\
#     np.sum(np.equal(np.sign(y_test[:1]-y_test[:0]), np.sign(y_test_pred[:1]-y_test_pred[:, 0]))
#            .astype(int)) / y_test.shape[0]
#
# print('correct sign prediction for close - open price for train/valid/test: %.2f/%.2f/%.2f' %
#       (corr_price_development_train, corr_price_development_valid, corr_price_development_test))
"""
