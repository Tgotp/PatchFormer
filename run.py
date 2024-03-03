import argparse
import torch
import random
import numpy as np
from exp.exp_main import Exp_Main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    
    parser.add_argument('--random_seed',type=int,default=2024,help='random_seed')
    parser.add_argument('--is_training',type=int,required=True,default=1,help='status')
    parser.add_argument('--model_id',type=str,required=True,default='test',help='model id')
    parser.add_argument('--model',type=str,required=True,default='PathFormer',
                            help='model name,options = [PathFormer]')
    
    parser.add_argument('--data_path',type=str,required=True,default='./data/Graph_BladeIcing/Icing_128_20/',help='dataset path')
    parser.add_argument('--dropout',type=float,default=0.05,help='dropout')
    parser.add_argument('--train_epochs',type=int,default=100,help='train epochs')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')


    parser.add_argument('--loss',type=str,default='mse', help='loss function')
    parser.add_argument('--learning_rate',type=float,default=0.0001, help='optimizer learning rate')
    parser.add_argument('--batch_size',type=int,default=64,help='batch size of train input data')

    parser.add_argument('--use_gpu',type=bool,default=True,help='use gpu')
    parser.add_argument('--use_multi_gpu',action='store_true',help='use multiple gpus',default=False)
    parser.add_argument('--gpu',type=int,default=0,help='gpu')
    parser.add_argument('--devices',type=str,default='0,1,2,3',help='device ids of multile gpus')
    
    args = parser.parse_args()

    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False 

    if args.use_gpu and args.use_multi_gpu :
        args.dvices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    print('Args in experiment:')
    print(args)

    if args.is_training:

        Exp = Exp_Main
        setting = '{}_{}'.format
        exp = Exp(args)
        exp.train(setting)
    

