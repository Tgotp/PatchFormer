from tqdm import tqdm
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping,adjust_learning_rate
from models import Pathformer
from data_provider.data_factory import data_provider

import torch.nn as nn

from torch.optim import lr_scheduler 
from torch import optim
import numpy as np
import torch
import time
import os    

class Exp_Main(Exp_Basic):
    def __init__(self,args):

        super(Exp_Main,self).__init__(args)


    def _build_model(self):
        model_dict = {
            'PathFormer': Pathformer ,
        }

        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model,device_ids=self.args.device_ids)
        return model
    
    def _get_data(self,flag):
        data_loader = (self.args,flag)
        return data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.BCELoss()
        return criterion
    
    def vali(self, vali_loader, criterion):
        total_loss = []
        correct = 0
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                correct += (outputs.argmax(1) == batch_y.argmax(1)).type(torch.float).sum().item()
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        correct /= len(vali_loader.dataset)
        return correct, total_loss

    def train(self, setting):
        
        train_loader = data_provider(self.args,flag='train')
        # vali_loader , signal_length_vali  = data_provider(flag='test')
        test_loader  = data_provider(self.args,flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)


        for epoch in range(self.args.train_epochs):
            print(f"Epoch #{epoch+1}:")
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            # test_acc = torchmetrics.Accuracy
            size = len(train_loader.dataset)
            i,correct = 0,0
            for batch_x,batch_y in tqdm(train_loader):
                iter_count += 1
                i += 1
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # print('batch_x:\n',batch_x)

                model_optim.zero_grad()
                
                outputs = self.model(batch_x)

                # print('outputs:\n',outputs)
                # print('batch_y:\n',batch_y)

                loss = criterion(outputs, batch_y)
                correct += (outputs.argmax(1) == batch_y.argmax(1)).type(torch.float).sum().item()
                
                train_loss.append(loss.item())

                if iter_count % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # vali_acc, vali_loss = self.vali(vali_loader, criterion)
            test_acc, test_loss = self.vali(test_loader, criterion)
            vali_loss = test_loss

            correct /= size

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f} | Train Accuracy: {4:.7f} Vali Accuracy: {5:.7f}".format(
                epoch + 1, train_steps, train_loss, test_loss, correct,test_acc))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        

        return self.model

    
