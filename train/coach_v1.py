"""
这个脚本是用来训练面部轮廓的
"""
import sys
sys.path.append(".")
sys.path.append("..")
from models.mouth_net import MouthNet
from torch.utils.tensorboard import SummaryWriter
from datasets.dataset import *
from configs.config_v1 import config as cfg
from torch.utils.data import DataLoader
import os
import torch.optim as optim
import torch.nn as nn
os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu_ids']

class Coach:
    def __init__(self):
        self.global_test_loss = float('Inf')

        # 得到配置文件
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 创建主要的网络
        self.net = MouthNet(class_num=cfg['class_num']).to(self.device)
        self.net.train()

        if cfg['ckpt'] != "":
            ckpt = torch.load(cfg['ckpt'])
            # 使用不严格的weight加载方式, 并且舍弃shape mismatch的
            pretrain_state_dict = ckpt
            net_state_dict = self.net.state_dict()
            for key in net_state_dict:
                if key in pretrain_state_dict.keys():
                    if net_state_dict[key].shape != pretrain_state_dict[key].shape:
                        pretrain_state_dict.pop(key)
            self.net.load_state_dict(pretrain_state_dict, strict=False)

        # 使用多卡训练
        if torch.cuda.device_count() > 1:
            print("Let's use ", torch.cuda.device_count(), "GPUs.")
            self.net = nn.DataParallel(self.net)

        # 创建训练日志
        if not os.path.exists("experiment/logs"):
            os.makedirs("experiment/logs")
        self.logger = SummaryWriter(log_dir='./experiment/logs')

        # 创建数据集
        trainsets = AudioDataset(cfg['train_target_root'], cfg['train_data_root'])
        self.trainloader = torch.utils.data.DataLoader(trainsets, batch_size=self.cfg['train_batch_size'], shuffle=True,
                                                       num_workers=cfg['num_workers'])

        testsets = AudioDataset(cfg['val_target_root'], cfg['val_data_root'])
        self.testloader = torch.utils.data.DataLoader(testsets, batch_size=self.cfg['train_batch_size'], shuffle=True,
                                                       num_workers=cfg['num_workers'])

        # 创建优化器(记得加上正则化的参数)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.cfg['lr'], betas=(0.9, 0.999), eps=1e-08,
                                        weight_decay=0.0001)

        # 创建损失函数
        self.MSELoss = torch.nn.MSELoss(reduction='sum')

    def MAELoss(self, pred, target):
        """
        2022.05.17增加了一个MAELoss, 这个loss对异常值比较敏感
        """
        loss = torch.sum(torch.abs(pred - target), dim=-1)
        loss = torch.mean(loss)
        return loss

    def criterion(self, pred, target):
        loss = self.MSELoss(pred, target)
        return loss

    def update_optimizer_lr(self, optimizer, lr):
        """
        为了动态更新learning rate， 加快训练速度
        :param optimizer: torch.optim type
        :param lr: learning rate
        :return:
        """
        for group in optimizer.param_groups:
            group['lr'] = lr

    def train(self):
        iter_num = 0
        mean_loss = 0
        for i in range(self.cfg['epoch']):
            for idx, (datas, targets) in enumerate(self.trainloader):
                iter_num += 1

                datas, targets = datas.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(datas)

                # 计算损失
                loss = self.criterion(outputs, targets)
                mean_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                # 打印loss
                if iter_num % self.cfg['print_loss'] == 0:
                    mean_loss = mean_loss / self.cfg['print_loss']
                    # mean_loss = np.array(mean_loss.detach().cpu())
                    print("lr = {} total iteration {} epoch {}, iteration {}, loss = {}".format(str(round(self.optimizer.param_groups[0]['lr'], 6)),
                                                                                                str(iter_num), str(i), str(idx),
                                                                                                str(round(mean_loss, 6))))
                    self.logger.add_scalar('{}/{}'.format('train', 'loss'), mean_loss, int(iter_num))
                    mean_loss = 0
                # test
                if iter_num % self.cfg['val_interval'] == 0:
                    self.net.eval()
                    self.eval(i, idx)
                    self.net.train()

                # lr decay
                # 2022.05.17调整lr下降的幅度, 之前是0.01, 现在是0.9 or 0.5
                # (可能是因为lr太大导致后期的训练波动, 使得eval loss比train loss大)
                if (iter_num - self.cfg['warmup_steps']) % self.cfg['lr_update_interval'] == 0:
                    lr = self.optimizer.param_groups[0]['lr'] * 0.9
                    self.update_optimizer_lr(self.optimizer, lr)

                elif iter_num < self.cfg['warmup_steps']:
                    lr = self.optimizer.param_groups[0]['lr'] * (iter_num / self.cfg['warmup_steps'])
                    self.update_optimizer_lr(self.optimizer, lr)

    def eval(self, epoch, iteration):

        test_loss = 0
        test_num = 0
        for idx, (datas, targets) in enumerate(self.testloader):
            test_num += 1

            datas, targets = datas.to(self.device), targets.to(self.device)

            with torch.no_grad():
                outputs = self.net(datas)

            # 计算损失
            loss = self.criterion(outputs, targets)
            test_loss += loss.item()

            if test_num > 20:
                break

        test_loss = test_loss / test_num
        if test_loss < self.global_test_loss:
            self.global_test_loss = test_loss

            if not os.path.exists("experiment/checkpoints"):
                os.makedirs("experiment/checkpoints")

            torch.save(self.net.state_dict(),
                           os.path.join("experiment/checkpoints", 'best_model_loss_{}.pth'.format(str(test_loss))))

        self.logger.add_scalar('{}/{}'.format('test', 'loss'), test_loss, epoch)
        print("lr = {} epoch {}, iteration {}, eval loss = {}".format(str(round(self.optimizer.param_groups[0]['lr'], 6)), str(epoch), str(iteration), str(round(test_loss, 6))))

if __name__ == '__main__':
    coach = Coach()
    coach.train()