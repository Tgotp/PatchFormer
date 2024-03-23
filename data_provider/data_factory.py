from torch.utils import data
from utils.tools import to_categorical,custom_smote
import torch
import numpy as np

def data_provider(args,flag,resample ='false'):
    
    data_x = np.load(args.data_path + args.data + 'X_' + flag + '.npy')
    data_y = np.load(args.data_path + args.data + 'y_' + flag + '.npy')
    
    if resample==True:
        sampling_strategy = 'auto'
        data_x, data_y = custom_smote(data_x, data_y, sampling_strategy=sampling_strategy, random_state=42)
        

    data_x = torch.from_numpy(data_x)
    data_y = torch.from_numpy(data_y)
    
    data_x = data_x.permute(0,2,1)
    class_num = 2
    data_y = to_categorical(data_y, class_num)  # 独热编码


    print('x.shape',data_x.shape)

    dataset = data.TensorDataset(data_x,data_y)

    data_loader = torch.utils.data.DataLoader(dataset,batch_size = args.batch_size,shuffle = True)

    return data_loader