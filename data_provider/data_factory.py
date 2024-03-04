from torch.utils import data
import torch
import numpy as np

def data_provider(args,flag):
    
    data_x = torch.from_numpy(np.load(args.data_path + 'X_' + flag + '.npy'))
    data_y = torch.from_numpy(np.load(args.data_path + 'y_' + flag + '.npy'))

    dataset = data.TensorDataset(data_x,data_y)

    data_loader = torch.utils.data.DataLoader(dataset,batch_size = args.batch_size,shuffle = True)

    return data_loader