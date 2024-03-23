import numpy as np
import torch
import pywt
import matplotlib.pyplot as plt
import time
from sklearn.utils import check_random_state

plt.switch_backend('agg')

def custom_smote(X, y, sampling_strategy='auto', random_state=None):
    random_state = check_random_state(random_state)

    # 计算需要生成的样本数量
    y = y.ravel()
    class_counts = np.bincount(y)
    if sampling_strategy == 'auto':
        target_count = np.max(class_counts)
    else:
        target_count = sampling_strategy * np.sum(class_counts)

    # 遍历少数类别样本，生成合成样本
    new_samples = []
    for class_label in np.unique(y):
        if class_counts[class_label] < target_count:
            # 找到该类别的样本
            class_samples = X[y == class_label]
            num_samples_to_generate = int(target_count) - class_counts[class_label]

            # 随机生成样本
            for i in range(num_samples_to_generate):
                # 随机选择一个样本
                random_sample = class_samples[random_state.randint(0, class_counts[class_label])]
                new_samples.append(random_sample)
                y = np.append(y, class_label)

    # 合并原样本和新生成的样本
    X_resampled = np.vstack((X, np.array(new_samples)))
    return X_resampled, y

def to_categorical(x,y):
    x = x.int().squeeze()
    return torch.eye(y)[x]

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

def Multi_Scale_Router(data,K): # [batch size, features, signal length]
    
    # 计算 DFT
    dft_result = torch.fft.fftn(data, dim=[-1])  # 在信号长度维度上进行 DFT

    # 计算每个频率分量的幅度
    amplitudes = torch.abs(dft_result)

    # 找到幅度最大的 K 个频率分量的索引
    top_indices = torch.argsort(amplitudes, dim=-1, descending=True)[..., :K]
    # print('fft indices',top_indices)
    # print('fft dft_result.shape',dft_result.shape)

    # 构造一个与原始信号相同形状的零张量，并将选定的频率分量放置在适当的位置
    dft_modified = torch.zeros_like(dft_result)
    dft_modified.scatter_(-1, top_indices, dft_result)

    # 使用逆离散傅里叶变换（IDFT）将修改后的频率域信号转换回时域
    idft_result = torch.fft.ifftn(dft_modified, dim=[-1]).real  # 在信号长度维度上进行 IDFT

    # 从原始信号中减去季节性模式，得到剩余部分
    residual = data - idft_result

    # print("Residual:", residual)

    return residual

def wavelet_FCNN_preprocessing_set(X, waveletLevel=3, waveletFilter='haar'):
    '''
    :param X: (sample_num, feature_num, sequence_length)
    :param waveletLevel:
    :param waveletFilter:
    :return: result (sample_num, extended_sequence_length, feature_num)
    '''
    N = X.shape[0]
    feature_dim = X.shape[1]
    length = X.shape[2]
    signal_length = []
    signal_length.append(length)

    stats = []

    extened_X = []
    extened_X.append(np.transpose(X, (0, 2, 1)))

    for i in range(N):# for each sample
        for j in range(feature_dim): # for each dim
            wavelet_list = pywt.wavedec(X[i][j], waveletFilter, level=waveletLevel)
            if i == 0 and j == 0:
                for l in range(waveletLevel):
                    current_length = len(wavelet_list[waveletLevel - l])
                    signal_length.append(current_length)
                    extened_X.append(np.zeros((N,current_length,feature_dim)))
            for l in range(waveletLevel):
                extened_X[l+1][i,:,j] = wavelet_list[waveletLevel-l]

    result = None
    first = True
    for mat in extened_X:
        mat_mean = mat.mean()
        mat_std = mat.std()
        mat = (mat-mat_mean)/(mat_std)
        stats.append((mat_mean,mat_std))
        if first:
            result = mat
            first = False
        else:
            result = np.concatenate((result, mat), axis=1)

    print(result.shape)
    return result, signal_length
