<<<<<<< HEAD
from exp.exp_basic import Exp_Basic

class Exp_Main(Exp_Basic):
    def __init__(self,args):
        super(Exp_Main,self).__init__(args)
=======
from exp.exp_basic import Exp_Basic
from models import Pathformer
import torch.nn as nn
import torch

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
    
    
    
>>>>>>> 934e7a6d4a561713eaf8a4a487a0719dbbbf7427
