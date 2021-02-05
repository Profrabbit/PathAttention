import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from rouge import FilesRouge


class Trainer:
    def __init__(self, args, model, train_data, test_data, t_vocab):
        self.args = args
        cuda_condition = torch.cuda.is_available() and self.args.with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.model = model.to(self.device)
        self.train_data = train_data
        self.test_data = test_data
        self.optim = Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.NLLLoss(ignore_index=0)
        self.clip = self.args.clip
        self.tensorboard_writer = SummaryWriter(
            'run/{}_{}'.format('relation' if args.relation else 'Naive',
                               datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
        self.iter = -1

        self.log_freq = args.log_freq
        self.t_vocab = t_vocab
        self.dataset_dir = os.path.join('./data', self.args.dataset, 'data')

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"
        data_iter = tqdm(enumerate(data_loader),
                         desc="EP_%s:%d" % (str_code, epoch),
                         total=len(data_loader),
                         bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        for i, data in data_iter:
            if self.args.tiny_data > 0:
                if i >= self.args.tiny_data:
                    break

            if train:
                self.model.train()
            else:
                self.model.eval()
            data = {key: value.to(self.device) for key, value in data.items()}
            out = self.model(data)
            loss = self.criterion(out.view(out.shape[0] * out.shape[1], -1), data['f_target'].view(-1))
            if train:
                self.optim.zero_grad()
                if self.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                loss.backward()
                self.optim.step()
            avg_loss += loss.item()
            post_fix = {
                'str': str_code,
                "epoch": epoch,
                "iter": i,
                "Iter loss": loss.item() / self.args.batch_size,
            }
            if train:
                self.iter += 1
                self.tensorboard_writer.add_scalar('Loss', post_fix['Iter loss'], self.iter)
            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))

    def predict(self, epoch):
        true_positive, false_positive, false_negative = 0, 0, 0

        def filter_special(lis_):
            return list(filter(lambda x: x not in self.t_vocab.special_index, lis_))

        def statistics(predict, original, ref_file, pred_file):
            # TODO 测量不能用数字来比 而应该用原始的文本进行比较 因为ref有可能变成unk了 这个事情就有点麻烦了
            nonlocal true_positive, false_positive, false_negative
            for p, o in zip(predict, original):
                p, o = filter_special(p.tolist()), filter_special(o.tolist())
                ref_file.write(' '.join([self.t_vocab.re_find(sub_token) for sub_token in o]) + '\n')
                pred_file.write(' '.join([self.t_vocab.re_find(sub_token) for sub_token in p]) + '\n')
                p, o = sorted(p), sorted(o)
                if p == o:
                    true_positive += len(o)
                    continue
                for sub_token in p:
                    if sub_token in o:
                        true_positive += 1
                    else:
                        false_positive += 1
                for sub_token in o:
                    if sub_token not in p:
                        false_negative += 1

        def calculate_results(true_positive, false_positive, false_negative):
            if true_positive + false_positive > 0:
                precision = true_positive / (true_positive + false_positive)
            else:
                precision = 0
            if true_positive + false_negative > 0:
                recall = true_positive / (true_positive + false_negative)
            else:
                recall = 0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0
            return precision, recall, f1

        data_iter = tqdm(enumerate(self.test_data),
                         desc="EP_%s:%d" % ('infer', epoch),
                         total=len(self.test_data),
                         bar_format="{l_bar}{r_bar}")

        ref_file_name = os.path.join(self.dataset_dir, 'ref.txt')
        predicted_file_name = os.path.join(self.dataset_dir, 'pred.txt')
        with open(ref_file_name, 'w') as ref_file, open(predicted_file_name, 'w') as pred_file:
            for i, data in data_iter:
                if self.args.tiny_data > 0:
                    if i >= self.args.tiny_data:
                        break
                self.model.eval()
                data = {key: value.to(self.device) for key, value in data.items()}
                memory, memory_key_padding_mask = self.model.encode(data)
                f_source = torch.ones(self.args.batch_size, 1, dtype=torch.long).fill_(self.t_vocab.sos_index).to(
                    memory.device)
                for _ in range(self.args.max_target_len):
                    out = self.model.decode(memory, f_source, memory_key_padding_mask)  # 这个地方之后可能需要进行概率移动
                    out[:, :, self.t_vocab.unk_index] = float('-inf')
                    idx = torch.argmax(out, dim=-1)[:, -1].view(-1, 1)
                    f_source = torch.cat((f_source, idx), dim=-1)
                statistics(f_source, data['f_source'], ref_file, pred_file)
        precision, recall, f1 = calculate_results(true_positive, false_positive, false_negative)
        files_rouge = FilesRouge()
        # rouge = files_rouge.get_scores(predicted_file_name, ref_file_name, avg=True)
        print(
            "precision={:.6f}, recall={:.6f}, f1={:.6f}".format(precision, recall, f1))
