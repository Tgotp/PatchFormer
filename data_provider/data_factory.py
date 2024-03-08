from torch.utils import data
from utils.tools import to_categorical, wavelet_FCNN_preprocessing_set,Multi_Scale_Router
import torch
import numpy as np

def data_provider(args,flag):
    
    data_x = torch.from_numpy(np.load(args.data_path + 'X_' + flag + '.npy'))
    data_y = torch.from_numpy(np.load(args.data_path + 'y_' + flag + '.npy'))
    
    data_x = Multi_Scale_Router(data_x,args.K)
    print('fft x.shape',data_x.shape)

    class_num = 2
    data_y = to_categorical(data_y, class_num)  # 独热编码

    # print('x.shape',data_x.shape)

    dataset = data.TensorDataset(data_x,data_y)

    data_loader = torch.utils.data.DataLoader(dataset,batch_size = args.batch_size,shuffle = True)

    return data_loader