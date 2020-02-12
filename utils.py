import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def save_mask_prediction_example(mask, pred, iter):
	plt.imshow(pred[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_prediction.png")
	plt.imshow(mask[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_mask.png")


def show_curve(y1s, title='loss'):
    """
    plot curlve for Loss and Accuacy\\
    Args:\\
        ys: loss or acc list\\
        title: loss or accuracy
    """
    x = np.array(range(len(y1s)))
    y1 = np.array(y1s)
    plt.plot(x, y1, label='train')
    plt.axis()
    plt.title('{} curve'.format(title))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(title))
    plt.legend(loc='best')
    #plt.show()
    plt.savefig("picture/{}.png".format(title))
    plt.show()
    plt.close()
    print('Saved figure: picture/{}.png'.format(title))


def bayes_uncertain(image_np, label_np, results):
    """
    Keyword arguments:
    image_np -- 原图
    label_np -- 标签
    results -- 同一张图片的不同预测结果
    Return: 预测结果的均值和方差
    """
    results = np.array(results) # list->numpy
    shape = results.shape
    # mean_result
    # variance_result

    mean_result = np.zeros((128, 128))      # 均值
    variance_result = np.zeros((128, 128))  # 方差

    # 计算均值
    for i in range(shape[0]):
        mean_result += results[i]
    mean_result /= shape[0]

    # 计算方差
    for i in range(shape[0]):
        variance_result += np.square(mean_result-results[i])
    variance_result /= shape[0]
    
    # 显示保存图片
    fig, ax = plt.subplots(2,2, sharey=True, figsize=(14,12))

    ax[0][0].set_title("Original data")
    ax[0][1].set_title("Ground Truth")
    ax[1][0].set_title("mean predicted result")
    ax[1][1].set_title("variance")

    ax00 = ax[0][0].imshow(image_np, aspect="auto", cmap="gray")
    ax01 = ax[0][1].imshow(label_np, aspect="auto")
    ax10 = ax[1][0].imshow(mean_result, aspect="auto")
    ax11 = ax[1][1].imshow(variance_result, aspect="auto")

    fig.colorbar(ax00, ax=ax[0][0])
    fig.colorbar(ax01, ax=ax[0][1])
    fig.colorbar(ax10, ax=ax[1][0])
    fig.colorbar(ax11, ax=ax[1][1])
    
    # 保存
    plt.savefig('picture/mean_variance.jpg')

    
