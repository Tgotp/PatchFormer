import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from utils.tools import wavelet_FCNN_preprocessing_set
from layers.PatchTST_backbone import PatchTST_backbone

class Model(nn.Module):
    def __init__(self,configs,signal_length,device, max_seq_len:Optional[int]=1024, d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs):
        super().__init__()
    
        # load parameters
        self.device = device
        self.signal_length = signal_length

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        kernel_size = configs.kernel_size

        left = 0
        length = 0        
        self.hete_models = []

        for i in range(len(signal_length)):
            length = self.signal_length[i]
            right = int(left + length)
            
            left += length

            context_window = signal_length[i]
            target_window = context_window
            print('context_window',signal_length[i])

            self.hete_models.append(PatchTST_backbone(c_in=context_window, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs).to(device))

        self.fs = nn.functional.softmax
        self.linear = nn.Linear(240*26,2)
    
    def forward(self, x):
        
        left = 0
        length = 0
        total_length = x.shape[1]
        waveletOutput = []
        for i in range(len(self.signal_length)):
            length = self.signal_length[i]
            right = int(left + length)
            x_crop = x[:, left:right, :]
            left += length

            x_crop = x_crop.permute((0, 2, 1))
            # print('x_crop',x_crop.shape)
            y = self.hete_models[i](x_crop)
            y = y.permute((0, 2, 1))
            waveletOutput.append(y)
        
        x = torch.cat(waveletOutput,axis = 1)
        x = torch.flatten(x,1,2)
        # print('linearshape',x.shape)
        x = self.linear(x)
        x = self.fs(x,dim=0)
        return x