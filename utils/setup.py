import os
import wandb
import torch
import pathlib
import argparse
import numpy as np

from models.GRUNet import GRUNet
from models.RTransformer import RT


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, default='c')
    parser.add_argument('--targets', type=str, default='c')
    parser.add_argument('--dropout', type=float, default=0.35)
    parser.add_argument('--clip', type=float, default=0.35)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--ksize', type=int, default=9)
    parser.add_argument('--n_level', type=int, default=3)
    parser.add_argument('--log_interval', type=int, default=200, metavar='N')
    parser.add_argument('--lr', type=float, default=5e-01)
    parser.add_argument('--model', type=str, default='RT')
    parser.add_argument('--rnn_type', type=str, default='LSTM')
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--h', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--proj_len', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=14)
    parser.add_argument('--seed', type=int, default=1111)

    args = parser.parse_args()
    log_filename, _ = get_filepaths(args)
    start_log(log_filename)
    set_seeds(args)
    print(args)

    args_dict = vars(args)
    wandb.init(project="zzsn-nyse", entity="piotrfratczak", config=args_dict)
    config = wandb.config
    args = dotdict(config)

    return args


def set_seeds(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def parse_features(args):
    features = []
    if 'c' in args:
        features.append('close')
    if 'o' in args:
        features.append('open')
    if 'h' in args:
        features.append('high')
    if 'l' in args:
        features.append('low')
    if not args:
        raise ValueError('Features list is empty.')
    return features


def get_features(args):
    features = parse_features(args.features)
    targets = parse_features(args.targets)
    return features, targets


def get_model(args):
    features, targets = get_features(args)
    if args.model in ['rt', 'RT', 'RTransformer', 'R-Transformer']:
        model = RT(input_size=len(features),
                   output_size=len(targets),
                   d_model=args.d_model,
                   h=args.h,
                   rnn_type=args.rnn_type,
                   ksize=args.ksize,
                   n_level=args.n_level,
                   n=args.n,
                   proj_len=args.proj_len,
                   dropout=args.dropout)
    else:
        model = GRUNet(input_size=len(features),
                       output_size=len(targets),
                       hidden_size=args.h,
                       gru_layers=args.n,
                       proj_len=args.proj_len,
                       dropout=args.dropout)
    return model


def get_filepaths(args):
    base_path = os.path.dirname(pathlib.Path(__file__).parent)
    out_dir = os.path.join(base_path, 'output/')

    if args.model in ['rt', 'RT', 'RTransformer', 'R-Transformer']:
        model_name = "RT_d_{}_h_{}_type_{}_k_{}_level_{}_n_{}"\
            .format(args.d_model, args.h, args.rnn_type, args.ksize, args.n_level, args.n)
    else:
        model_name = 'GRU_h_{}_layers_{}'.format(args.h, args.n)

    model_name += '_feat_{}_target_{}_proj_{}_seq_{}_drop_{}'\
        .format(args.features, args.targets, args.proj_len, args.seq_len, args.dropout)

    log_filename = out_dir + 'log_' + model_name + '.txt'
    model_filename = out_dir + 'model_' + model_name + '.pt'

    return log_filename, model_filename


def save_model(model, save_filename):
    with open(save_filename, "wb") as f:
        torch.save(model, f)
    print('Saved as %s' % save_filename)


def load_model(model_filename):
    model = torch.load(open(model_filename, "rb"))
    return model


def output_log(message, filename):
    print(message)
    with open(filename, 'a') as out:
        out.write(message + '\n')


def start_log(filename):
    with open(filename, 'w') as out:
        out.write('start\n')
