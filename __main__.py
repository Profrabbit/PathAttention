import argparse

from torch.utils.data import DataLoader
from data import PathAttenDataset, TextVocab
from trainer import Trainer
from model import Model


def train():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--dataset", type=str, help="train dataset", default='csn_python', choices=['csn_python'])
    parser.add_argument("--on_memory", type=bool, default=False, help="Loading on memory: true or false")
    # dataset size
    parser.add_argument("--max_code_length", type=int, default=512,
                        help="avg is 128, max is 1200, <256=0.925  <512=0.985")
    parser.add_argument("--sub_token_length", type=int, default=5, help="")
    parser.add_argument("--max_path_length", type=int, default=8, help="now part=0.01")
    parser.add_argument("--max_path_num", type=int, default=512, help="now part=0.01  the num of directed unique edge")
    parser.add_argument("--max_target_len", type=int, default=8,
                        help="valid token + 1(special token)   function num: avg is 2.5, max is 7")
    # vocab
    parser.add_argument("--s_vocab_portion", type=float, default=1.0, help="")
    parser.add_argument("--t_vocab_portion", type=float, default=1.0, help="")
    # Source Vocab Size = 17448 Target Vocab Size = 6290

    # trainer
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate of adam")
    parser.add_argument("--log_freq", type=int, default=500, help="printing loss every n iter: setting n")
    parser.add_argument("--clip", type=int, default=1, help="")
    parser.add_argument("--batch_size", type=int, default=1, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="dataloader worker size")
    # model
    parser.add_argument("--dropout", type=float, default=0.1, help="")
    # token embedding
    parser.add_argument("--pretrain", type=bool, default=False, help="")
    parser.add_argument("--embedding_file", type=str, default='./......', help="")  # TODO glove path
    # path embedding
    parser.add_argument("--path_embedding_size", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("--path_embedding_num", type=int, default=128,
                        help="node type num, and also be used for padding idx!!")  # TODO set node num
    # transformer
    parser.add_argument("--activation", type=str, default='RELU', help="", choices=['GELU', 'RELU'])
    parser.add_argument("--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("--layers", type=int, default=8, help="number of layers")
    parser.add_argument("--attn_heads", type=int, default=8, help="number of attention heads")

    # advance
    parser.add_argument("--relation", type=bool, default=True, help="")
    parser.add_argument("--tiny_data", type=int, default=100, help="only a little batch ")

    args = parser.parse_args()
    print('Experiment on {} dataset'.format(args.dataset))
    s_vocab = TextVocab(args, 'source')
    t_vocab = TextVocab(args, 'target')

    print("Loading Train Dataset")
    train_dataset = PathAttenDataset(args, s_vocab, t_vocab, type='valid')

    print("Loading Valid Dataset")
    valid_dataset = PathAttenDataset(args, s_vocab, t_vocab, type='valid')

    print("Creating Dataloader")
    if args.on_memory:
        num_workers = args.num_workers
    else:
        num_workers = 0

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=num_workers)
    test_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=num_workers)

    print("Building Model")
    model = Model(args, s_vocab, t_vocab)

    print("Creating Trainer")
    trainer = Trainer(args=args, model=model, train_data=train_data_loader, test_data=test_data_loader, t_vocab=t_vocab)

    print("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.test(epoch)
        trainer.predict(epoch)


if __name__ == '__main__':
    train()
