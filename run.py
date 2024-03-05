import argparse
import torch
import random
import numpy as np
from exp.exp_main import Exp_Main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Series Forecasting')
    
    parser.add_argument('--random_seed',type=int,default=2024,help='random_seed')
    # data loader
    parser.add_argument('--data_path',type=str,required=True,default='./data/Graph_BladeIcing/Icing_128_20/',help='dataset path')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--is_training',type=int,required=True,default=1,help='status')
    parser.add_argument('--class_num',type=int,default=2,help='class num')
    parser.add_argument('--model_id',type=str,required=True,default='test',help='model id')
    parser.add_argument('--model',type=str,required=True,default='PathFormer',
                            help='model name,options = [PathFormer]')
    # wavelet
    parser.add_argument('--waveletLevel',type=int,default=4,help='wavelet Level')
    parser.add_argument('--waveletFilter', type=str, default='haar', help='wavelet Filter')

    # PatchTST
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='None', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')


    # Formers 
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=2, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--dropout',type=float,default=0.05,help='dropout')

    # optimization
    parser.add_argument('--train_epochs',type=int,default=100,help='train epochs')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser.add_argument('--loss',type=str,default='mse', help='loss function')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--learning_rate',type=float,default=0.0001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--batch_size',type=int,default=64,help='batch size of train input data')

    # GPU
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
        setting = '{}_{}'.format(
                args.model_id,
                args.model)
        exp = Exp(args)
        exp.train(setting)
    

