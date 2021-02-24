import argparse

from torch.utils.data import DataLoader
from data import PathAttenDataset, TextVocab, UniTextVocab
from trainer import Trainer
from model import Model
import torch
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def train():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--dataset", type=str, help="train dataset", default='csn_python', choices=['csn_python'])
    parser.add_argument("--on_memory", type=boolean_string, default=True, help="Loading on memory: true or false")
    # dataset size
    parser.add_argument("--max_code_length", type=int, default=512,
                        help="avg is 128, max is 1200, <256=0.925  <512=0.985")
    parser.add_argument("--sub_token_length", type=int, default=5, help="")
    parser.add_argument("--max_path_length", type=int, default=8, help="now part=0.01")
    parser.add_argument("--max_path_num", type=int, default=512, help="now part=0.01  the num of directed unique edge")
    parser.add_argument("--max_target_len", type=int, default=6,
                        help="valid token + 1(special token)   function num: avg is 2.5, max is 7")
    # vocab
    parser.add_argument("--s_vocab_portion", type=float, default=0.999, help="")
    parser.add_argument("--t_vocab_portion", type=float, default=1, help="")
    parser.add_argument("--vocab_threshold", type=int, default=100, help="if use uni vocab, and use vocab threshold")
    parser.add_argument("--uni_vocab", type=boolean_string, default=False, help="")

    # trainer
    parser.add_argument("--with_cuda", type=boolean_string, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of adam")
    parser.add_argument("--log_freq", type=int, default=5000, help="printing loss every n iter: setting n")
    parser.add_argument("--clip", type=float, default=0, help="0 is no clip")
    parser.add_argument("--batch_size", type=int, default=64, help="number of batch_size")  # 4,16,8 on two gpus
    parser.add_argument("--val_batch_size", type=int, default=64, help="number of batch_size of valid")
    parser.add_argument("--infer_batch_size", type=int, default=32, help="number of batch_size of infer")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=16, help="dataloader worker size")
    parser.add_argument("--save", type=boolean_string, default=True, help="whether to save model checkpoint")
    parser.add_argument("--weight_decay", type=float, default=3e-5, help="")
    parser.add_argument("--accu_batch_size", type=int, default=64, help="number of batch_size")
    parser.add_argument("--label_smoothing", type=float, default=0.2, help="number of batch_size")
    # model
    parser.add_argument("--dropout", type=float, default=0.2, help="")
    # token embedding
    parser.add_argument("--pretrain", type=boolean_string, default=True, help="")
    parser.add_argument("--embedding_file", type=str, default='/dat01/jinzhi/pengh/glove.42B.300d.txt', help="")
    # 'D:/glove.42B.300d.txt'
    # 'var/data/pengh/glove.42B.300d.txt'
    # '/dat01/jinzhi/pengh/glove.42B.300d.txt'

    # path embedding
    parser.add_argument("--path_embedding_size", type=int, default=32, help="hidden size of transformer model")
    parser.add_argument("--path_embedding_num", type=int, default=164,
                        help="node type num, and also be used for padding idx!!")  # TODO set node num
    # transformer
    parser.add_argument("--activation", type=str, default='GELU', help="", choices=['GELU', 'RELU'])
    parser.add_argument("--hidden", type=int, default=512, help="hidden size of transformer model")
    parser.add_argument("--ff_fold", type=int, default=4, help="ff_hidden = ff_fold*hidden")
    parser.add_argument("--layers", type=int, default=6, help="number of layers")
    parser.add_argument("--decoder_layers", type=int, default=6, help="number of decoder layers")
    parser.add_argument("--attn_heads", type=int, default=8, help="number of attention heads")

    # advance
    parser.add_argument("--relation", type=boolean_string, default=False, help="")
    parser.add_argument("--tiny_data", type=int, default=0, help="only a little data ")
    parser.add_argument("--seed", type=boolean_string, default=False, help="fix seed")
    parser.add_argument("--data_debug", type=boolean_string, default=False, help="fix seed")
    parser.add_argument("--train", type=boolean_string, default=True, help="")
    parser.add_argument("--load_checkpoint", type=boolean_string, default=False, help="")
    args = parser.parse_args()
    if args.seed:
        setup_seed(20)
    print('Experiment on {} dataset'.format(args.dataset))
    if args.uni_vocab:
        s_vocab = UniTextVocab(args)
        t_vocab = s_vocab
    else:
        s_vocab = TextVocab(args, 'source')
        t_vocab = TextVocab(args, 'target')

    print("Loading Train Dataset")
    if args.data_debug:
        train_dataset = PathAttenDataset(args, s_vocab, t_vocab, type='valid')
    else:
        train_dataset = PathAttenDataset(args, s_vocab, t_vocab, type='train')

    print("Loading Valid Dataset")
    valid_dataset = PathAttenDataset(args, s_vocab, t_vocab, type='valid')

    if args.on_memory:
        num_workers = args.num_workers
    else:
        num_workers = 0

    print("Creating Dataloader")
    if args.train:
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=num_workers, shuffle=True)
    else:
        train_data_loader = None
    test_data_loader = DataLoader(valid_dataset, batch_size=args.val_batch_size, num_workers=num_workers)
    infer_data_loader = DataLoader(valid_dataset, batch_size=args.infer_batch_size, num_workers=num_workers)
    print("Building Model")
    model = Model(args, s_vocab, t_vocab)

    print("Creating Trainer")
    trainer = Trainer(args=args, model=model, train_data=train_data_loader, test_data=test_data_loader,
                      infer_data=infer_data_loader, t_vocab=t_vocab)
    if args.load_checkpoint:
        checkpoint_path = 'checkpoint/Naive_2021-02-18-23-38-45_14.pth'
        trainer.load(checkpoint_path)
    print("Training Start")
    for epoch in range(args.epochs):
        if args.train:
            trainer.train(epoch)
            trainer.test(epoch)
        trainer.predict(epoch)
    trainer.writer.close()


if __name__ == '__main__':
    train()
