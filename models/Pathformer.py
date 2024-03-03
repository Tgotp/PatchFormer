import torch
import torch.nn as nn
from utils.tools import wavelet_FCNN_preprocessing_set

class Model(nn.Module):
    def __init__(self,configs):
        super().__init__()
    
        # load parameters

        # self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
        #                           max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
        #                           n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
        #                           dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
        #                           attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
        #                           pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
        #                           pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
        #                           subtract_last=subtract_last, verbose=verbose, **kwargs)
        self.model = nn.Linear(configs.enc_in,2)
    
    def forward(self, x):
        x = self.model(x)
        return x