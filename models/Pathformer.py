import torch
import torch.nn as nn
from utils.tools import wavelet_FCNN_preprocessing_set

class Model(nn.Module):
    def __init__(self,configs):
        super().__init__()

        # self.model = PatchTST_backbone()

    def forward(self, x):
        # x = self.model(x)
        return x