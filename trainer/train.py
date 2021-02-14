import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from rouge import FilesRouge


class Trainer:
    def __init__(self, args, model, train_data, test_data, infer_data, t_vocab):
        self.args = args
        cuda_condition = torch.cuda.is_available() and self.args.with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        if cuda_condition and torch.cuda.device_count() > 1:
            self.wrap = True
            model = nn.DataParallel(model)
        else:
            self.wrap = False
        self.model = model.to(self.device)
        self.train_data = train_data
        self.test_data = test_data
        self.infer_data = infer_data
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.clip = self.args.clip
        self.writer_path = '{}_{}'.format('relation' if args.relation else 'Naive',
                                          datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        print(self.writer_path)
        self.tensorboard_writer = SummaryWriter(os.path.join('run', self.writer_path))
        self.writer = open(os.path.join('run', self.writer_path, 'experiment.txt'), 'w')
        print(self.args, file=self.writer, flush=True)
        self.iter = -1
        self.log_freq = args.log_freq
        self.t_vocab = t_vocab
        self.dataset_dir = os.path.join('./data', self.args.dataset, 'data')
        self.best_epoch, self.best_loss = 0, float('inf')
        self.accu_steps = self.args.accu_batch_size // self.args.batch_size
        self.criterion = nn.NLLLoss(ignore_index=0)
        if self.args.relation:
            print(
                "Total Parameters: {}*1e6".format(sum([p.nelement() for _, p in self.model.named_parameters()]) // 1e6),
                file=self.writer, flush=True)
        else:
            model_parameters = []
            for name, param in self.model.named_parameters():
                if 'path' in name:
                    continue
                else:
                    model_parameters.append(param)
            print("Total Parameters: {}*1e6".format(sum([p.nelement() for p in model_parameters]) // 1e6),
                  file=self.writer, flush=True)

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def label_smoothing_loss(self, logits, targets, eps=0, reduction='mean'):
        if eps == 0:
            return self.criterion(logits, targets)
        K = logits.shape[-1]
        one_hot_target = F.one_hot(targets, num_classes=K)
        l_targets = (one_hot_target * (1 - eps) + eps / K).detach()
        loss = -(logits * l_targets).sum(-1).masked_fill(targets == 0, 0.0)
        if reduction == 'mean':
            return loss.sum() / torch.count_nonzero(targets)
        elif reduction == 'sum':
            return loss.sum()
        return loss

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"
        data_iter = tqdm(enumerate(data_loader),
                         desc="EP_%s:%d" % (str_code, epoch),
                         total=len(data_loader),
                         bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        if train:
            self.optim.zero_grad()
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}
            if train:
                self.model.train()
                out = self.model(data)
                loss = self.label_smoothing_loss(out.view(out.shape[0] * out.shape[1], -1),
                                                 data['f_target'].view(-1),
                                                 eps=self.args.label_smoothing)  # avg at every step
                accu_loss = loss / self.accu_steps
                accu_loss.backward()
                if self.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                if (i + 1) % self.accu_steps == 0:
                    self.optim.step()
                    self.optim.zero_grad()
            else:
                self.model.eval()
                with torch.no_grad():
                    out = self.model(data)
                    loss = self.criterion(out.view(out.shape[0] * out.shape[1], -1),
                                          data['f_target'].view(-1))  # avg at every step
            avg_loss += loss.item()
            post_fix = {
                'str': str_code,
                "epoch": epoch,
                "iter": i,
                "Iter loss": loss.item(),
            }
            if train:
                self.iter += 1
                self.tensorboard_writer.add_scalar('Loss', post_fix['Iter loss'], self.iter)
            # if i % self.log_freq == 0:
            #     data_iter.write(str(post_fix))
        avg_loss = avg_loss / len(data_iter)
        if not train:
            if avg_loss <= self.best_loss:
                self.best_loss = avg_loss
                self.best_epoch = epoch
            print("Best Valid At EP%d_%s, best_loss=" % (self.best_epoch, str_code), self.best_loss, file=self.writer,
                  flush=True)
        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss, file=self.writer, flush=True)
        print('-------------------------------------', file=self.writer, flush=True)

        if self.args.save:
            save_dir = './checkpoint'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(self.model.state_dict(),
                       os.path.join(save_dir, "{}_{}.pth".format(self.writer_path, epoch)))

    def predict(self, epoch):
        true_positive, false_positive, false_negative = 0, 0, 0

        def filter_special(lis_):
            if self.t_vocab.eos_index in lis_:
                lis = lis_[:lis_.index(self.t_vocab.eos_index)]
            else:
                lis = lis_
            return list(filter(lambda x: x not in self.t_vocab.special_index, lis))

        def statistics(predict, original, ref_file, pred_file):
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

        data_iter = tqdm(enumerate(self.infer_data),
                         desc="EP_%s:%d" % ('infer', epoch),
                         total=len(self.infer_data),
                         bar_format="{l_bar}{r_bar}")

        ref_file_name = os.path.join(self.dataset_dir, 'ref.txt')
        predicted_file_name = os.path.join(self.dataset_dir, 'pred.txt')
        with open(ref_file_name, 'w') as ref_file, open(predicted_file_name, 'w') as pred_file:
            for i, data in data_iter:
                self.model.eval()
                data = {key: value.to(self.device) for key, value in data.items()}
                with torch.no_grad():
                    memory, memory_key_padding_mask = self.model.module.encode(
                        data) if self.wrap else self.model.encode(data)
                    f_source = torch.ones(memory.shape[0], 1, dtype=torch.long).fill_(
                        self.t_vocab.sos_index).to(
                        self.device)
                    for _ in range(self.args.max_target_len):
                        out = self.model.module.decode(memory, f_source,
                                                       memory_key_padding_mask) if self.wrap else self.model.decode(
                            memory, f_source, memory_key_padding_mask)
                        # out[:, :, self.t_vocab.unk_index] = float('-inf')
                        idx = torch.argmax(out, dim=-1)[:, -1].view(-1, 1)
                        f_source = torch.cat((f_source, idx), dim=-1)
                statistics(f_source, data['f_source'], ref_file, pred_file)
        precision, recall, f1 = calculate_results(true_positive, false_positive, false_negative)
        files_rouge = FilesRouge()
        # rouge = files_rouge.get_scores(predicted_file_name, ref_file_name, avg=True)
        print(
            "precision={:.6f}, recall={:.6f}, f1={:.6f}".format(precision, recall, f1), file=self.writer, flush=True)
        print('-------------------------------------', file=self.writer, flush=True)
