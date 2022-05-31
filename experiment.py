import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import warnings

from models.RTransformer import RT
from features.make_dataset import DatasetMaker

# warnings.filterwarnings("ignore")  # Suppress the RunTimeWarning on unicode

parser = argparse.ArgumentParser()
# parser.add_argument('--cuda', action='store_false')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--clip', type=float, default=0.15)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--ksize', type=int, default=6)
parser.add_argument('--n_level', type=int, default=3)
parser.add_argument('--log_interval', type=int, default=1000, metavar='N')
parser.add_argument('--lr', type=float, default=5e-05)
parser.add_argument('--optim', type=str, default='Adam')
parser.add_argument('--rnn_type', type=str, default='GRU')
parser.add_argument('--d_model', type=int, default=1)
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--h', type=int, default=1)
parser.add_argument('--seed', type=int, default=1111)

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device("cpu")

base_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(base_path, 'data/')
s_dir = os.path.join(base_path, 'output/')

print(args)

# ---------------
BATCH_SIZE = 256
SEQ_LEN = 7
pred_len = 1
columns = ['close', 'open']
targets = ['close']
dm = DatasetMaker()
train_loader, val_loader, test_loader = dm.make_dataset(columns, targets, SEQ_LEN, pred_len, BATCH_SIZE)
# ------------

input_size = len(columns)

dropout = args.dropout

model = RT(input_size, len(columns), len(targets), h=args.h, rnn_type=args.rnn_type, ksize=args.ksize,
           n_level=args.n_level, n=args.n, dropout=dropout, pred_len=pred_len)
model.to(device)

model_name = "d_{}_h_{}_type_{}_k_{}_level_{}_n_{}_lr_{}_drop_{}".format(args.d_model, args.h, args.rnn_type,
                                                                         args.ksize, args.n_level, args.n, args.lr,
                                                                         args.dropout)

message_filename = s_dir + 'r_' + model_name + '.txt'
model_filename = s_dir + 'm_' + model_name + '.pt'
with open(message_filename, 'w') as out:
    out.write('start\n')

criterion = nn.CrossEntropyLoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def save(model, save_filename):
    with open(save_filename, "wb") as f:
        torch.save(model, f)
    print('Saved as %s' % save_filename)


def output_s(message, save_filename):
    print(message)
    with open(save_filename, 'a') as out:
        out.write(message + '\n')


def evaluate(data_loader, name='Eval'):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(data_loader):
            output = model(x)
            output = output.reshape(output.size(0), y.size(1), y.size(2)).double()
            loss = criterion(output, y.double())
            total_loss += loss.item()
            count += output.size(0)
        eval_loss = total_loss / count
        message = name + " loss: {:.5f}".format(eval_loss)
        output_s(message, message_filename)
        return eval_loss


def train(epoch):
    model.train()
    total_loss = 0
    count = 0
    for idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(inputs)
        output = output.reshape(output.size(0), labels.size(1), labels.size(2))
        loss = criterion(output, labels)
        total_loss += loss.item()
        count += output.size(0)

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        loss.backward()
        optimizer.step()

        if idx > 0 and idx % args.log_interval == 0:
            cur_loss = total_loss / count
            message = "Epoch {:2d} | lr {:.5f} | loss {:.5f}".format(epoch, lr, cur_loss)
            output_s(message, message_filename)
            total_loss = 0.0
            count = 0


if __name__ == "__main__":
    best_vloss = 1e8
    vloss_list = []
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        vloss = evaluate(val_loader, name='Validation')
        tloss = evaluate(test_loader, name='Test')
        if vloss < best_vloss or epoch == 1:
            save(model, model_filename)
            best_vloss = vloss
        if epoch > 10 and vloss > max(vloss_list[-3:]):
            lr /= 10
            output_s('lr = {}'.format(lr), message_filename)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        vloss_list.append(vloss)

    message = '-' * 89
    output_s(message, message_filename)
    model = torch.load(open(model_filename, "rb"))
    tloss = evaluate(X_test)

