import torch
import wandb
import torch.nn as nn
import torch.optim as optim

from utils.setup import save_model, load_model, get_filepaths, output_log


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()


def evaluate(model, data_loader, log_filename, name='Eval'):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            output = output.reshape(labels.size())
            loss = criterion(output, labels.double())
            total_loss += loss.item()
            count += output.size(0)
        eval_loss = total_loss / count
        message = name + " loss: {:.5f}".format(eval_loss)
        output_log(message, log_filename)
        return eval_loss


def run_epoch(epoch, model, optimizer, train_loader, args):
    log_filename, _ = get_filepaths(args)
    model.train()
    total_loss = 0
    count = 0
    for idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        output = output.reshape(labels.size())
        loss = criterion(output, labels)
        total_loss += loss.item()
        count += output.size(0)

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        loss.backward()
        optimizer.step()

        if idx > 0 and idx % args.log_interval == 0:
            cur_loss = total_loss / count
            message = "Epoch {:2d} | lr {:.5f} | loss {:.5f}".format(epoch, args.lr, cur_loss)
            output_log(message, log_filename)
            total_loss = 0.0
            count = 0


def train(model, data_loaders, args):
    lr = args.lr
    model.to(device)
    log_filename, model_filename = get_filepaths(args)
    train_loader, val_loader, test_loader = data_loaders
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

    best_vloss = 1e8
    vloss_list = []
    for epoch in range(1, args.epochs + 1):
        run_epoch(epoch, model, optimizer, train_loader, args)

        vloss = evaluate(model, val_loader, log_filename, name='Validation')
        if vloss < best_vloss or epoch == 1:
            save_model(model, model_filename)
            best_vloss = vloss
        if epoch > 10 and vloss > max(vloss_list[-3:]):
            lr /= 10
            output_log('lr = {}'.format(lr), log_filename)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        vloss_list.append(vloss)

    test(test_loader, args)


def test(test_loader, args):
    log_filename, model_filename = get_filepaths(args)
    output_log('-' * 89, log_filename)
    model = load_model(model_filename)
    tloss = evaluate(model, test_loader, log_filename, name='Test')

    wandb.log({"test_loss": tloss})
    wandb.watch(model)
