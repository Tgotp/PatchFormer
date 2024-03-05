from torch.utils import data
from utils.tools import to_categorical, wavelet_FCNN_preprocessing_set
import torch
import numpy as np

def data_provider(args,flag):
    
    data_x = torch.from_numpy(np.load(args.data_path + 'X_' + flag + '.npy'))
    data_y = torch.from_numpy(np.load(args.data_path + 'y_' + flag + '.npy'))
    
    class_num = 2
    data_y = to_categorical(data_y, class_num)  # 独热编码

    print('x.shape',data_x.shape)

    data_x, signal_length = wavelet_FCNN_preprocessing_set(data_x, waveletLevel=args.waveletLevel,
                                                            waveletFilter=args.waveletFilter)
    data_x = torch.from_numpy(data_x)
    print('wavex.shape',data_x.shape)
    print('signal_length',len(signal_length))

    dataset = data.TensorDataset(data_x,data_y)

    data_loader = torch.utils.data.DataLoader(dataset,batch_size = args.batch_size,shuffle = True)

    return data_loader, signal_length