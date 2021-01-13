from __future__ import print_function
import matplotlib.pyplot as plt

# 实现输入一个含噪图片，输出一个去噪图片的函数，


import os
import warnings
warnings.filterwarnings('ignore')

from include import *
from PIL import Image
import PIL

import numpy as np
import torch
import torch.optim
from torch.autograd import Variable

GPU = True
if GPU == True:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("num GPUs",torch.cuda.device_count())
else:
    dtype = torch.FloatTensor



def denoise(img_noisy_var, img_clean_var, k=128,numit = 1900,rn = 0.0,find_best=True,upsample_first = True):
    num_channels = [k]*4
    output_depth = img_noisy_var.shape[0]
    net = decodernw(output_depth,num_channels_up=num_channels,upsample_first=upsample_first).type(dtype)
    mse_n, mse_t, ni, net = fit( num_channels=num_channels,
                        reg_noise_std=rn,
                        num_iter=numit,
                        LR = 0.0001,
                        img_noisy_var=img_noisy_var,
                        net=net,
                        img_clean_var=img_clean_var,
                        find_best=find_best
                        )
    out_img_np = net( ni.type(dtype) ).data.cpu().numpy()[0]
    return out_img_np, mse_t

def myimgshow(plt,img):
    plt.imshow(np.clip(img.transpose(1, 2, 0),0,1), cmap='gray')

def plot_results(out_img_np,img_np,img_noisy_np):
    fig = plt.figure(figsize = (15,15)) # create a 5 x 5 figure

    ax1 = fig.add_subplot(131)
    myimgshow(ax1,img_np)
    ax1.set_title('Original image')
    ax1.axis('off')

    ax2 = fig.add_subplot(132)
    myimgshow(ax2,img_noisy_np)
    ax2.set_title( "Noisy observation, PSNR: %.2f" % psnr(img_np,img_noisy_np) )
    ax2.axis('off')

    ax3 = fig.add_subplot(133)
    myimgshow(ax3,out_img_np)
    ax3.set_title( "Deep-Decoder denoised image, SNR: %.2f" % psnr(img_np,out_img_np) )
    ax3.axis('off')

    plt.show()

# def do_denoise(clean_img, noise_img):
def do_denoise(noise_img):
    clean_img = np.load("./original_img.npy")
    # clean_img = np.load("./original_magnitudes.npy")
    # clean_img/noise_img: 三通道图像，(h,w,c)，ndarray形式
    # 假设传进来的是未归一化的原图
    if (len(clean_img.shape)==2):  # 灰度图在通道上置1
        clean_img = np.expand_dims(clean_img, axis=2)
        noise_img = np.expand_dims(noise_img, axis=2)

    print("clean img shape: {}".format(clean_img.shape))
    print("noisy img shape: {}".format(noise_img.shape))

    # 如果输入没有归一化，这里要做归一化
    max_value = np.max(clean_img)
    clean_img /= max_value
    noise_img /= max_value

    print(type(noise_img))
    print(noise_img.shape)
    # 调整通道顺序
    clean_img = clean_img.transpose(2, 0, 1)
    noise_img = noise_img.transpose(2, 0, 1)
    # 转为torch variable
    # print("after transpose:", noise_img.shape)
    clean_img_var = np_to_var(clean_img).type(dtype)
    noise_img_var = np_to_var(noise_img).type(dtype)
    # print("after np_to_var:", noise_img_var.shape)

    out_img_np, mse_t = denoise(noise_img_var, clean_img_var, k=128, numit =800, rn = 0.0)

    # plot_results(out_img_np, clean_img, noise_img)
    print( "Noisy observation, PSNR: %.2f" % psnr(clean_img,noise_img) )
    print( "Deep-Decoder denoised image, SNR: %.2f" % psnr(clean_img,out_img_np) )

    # 为了给到hio的输入，在这里要裁掉维度为1的通道
    # print(np.squeeze(out_img_np).shape)
    # 不可以忘记恢复归一化！传入和输出的都应该是未归一化的结果
    return np.squeeze(out_img_np) * max_value



def main():
    # img_path = "./original_magnitudes.npy"
    # noise_img_path = "./noise_magnitudes.npy"
    img_path = "./original_img.npy"
    noise_img_path = "./result.npy"
    clean_img = np.load(img_path)
    noise_img = np.load(noise_img_path)

    do_denoise(clean_img, noise_img)

if __name__ == "__main__":
    main()
