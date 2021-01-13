import random
import os
import warnings
warnings.filterwarnings('ignore')

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn
from torch.autograd import Variable
from mpl_toolkits.mplot3d import Axes3D
from fit import fit
from net import U_Net
from util import data_transform

GPU = True
if GPU == True:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("num GPUs",torch.cuda.device_count())
else:
    dtype = torch.FloatTensor

pcolor = {'red': (239/255, 131/255, 119/255),
          'blue': (65/255, 113/255, 156/255),
          'green': (173, 208, 92)}

def inference():
    unet = U_Net(in_ch=1, out_ch=1).cuda()
    # unet.load_state_dict(torch.load('best.pth'))
    unet.load_state_dict(torch.load('实验/关于网络结构/lena/slim-8-last.pth'))
    # 用于训练网络的原始图像及其傅里叶intensity
    image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    # image = (255 - image)*0.1
    image = cv2.resize(image, (256,256))
    # 支撑域
    # reduced_image = cv2.resize(image, (128, 128))
    # image = np.zeros((256, 256))
    # image[64:192, 64:192] = reduced_image

    phase_obj = image / 255 * 2 * np.pi *0.5 #* 0.5
    magnitudes = np.abs(np.fft.fft2(np.exp(1j*phase_obj)))
    magnitudes_t = Variable(data_transform(magnitudes).unsqueeze(0))
    retrieved_phase = unet(magnitudes_t.type(dtype)).data.cpu().squeeze(0).numpy()[0]

    # phase shift问题，减最小值处理
    phase_obj -= np.min(phase_obj)
    retrieved_phase -= np.min(retrieved_phase)

    plt.figure()
    plt.imshow(phase_obj/(2*np.pi), cmap='gray', vmin=0, vmax=0.5)
    plt.colorbar(ticks=[0, 0.5])
    plt.axis('off')
    plt.figure()
    plt.imshow(retrieved_phase/(2*np.pi), cmap='gray', vmin=0, vmax=0.5)
    plt.colorbar(ticks=[0, 0.5])
    plt.axis('off')
    plt.show()

    # 非训练图像
    # img_list = ['peppers_gray.tif', 'cameraman.png', 'house.tif']
    # for idx, img in enumerate(img_list):
    #     image_test = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    #     image_test = cv2.resize(image_test, (256,256))
    #     phase_obj_test = image_test / 255 * 2 * np.pi *0.5 #* 0.5
    #     magnitudes_test = np.abs(np.fft.fft2(np.exp(1j*phase_obj_test)))
    #     magnitudes_t_test = Variable(data_transform(magnitudes_test).unsqueeze(0))
    #     retrieved_phase_test = unet(magnitudes_t_test.type(dtype)).data.cpu().squeeze(0).numpy()[0]
    #     plt.subplot(2,4,idx*2+3)
    #     plt.imshow(phase_obj_test, cmap='gray')
    #     plt.subplot(2,4,idx*2+4)
    #     plt.imshow(retrieved_phase_test, cmap='gray')
    # plt.show()

def inference_exp():
    # -------加载system结果 ------------
    unet1 = U_Net(in_ch=1, out_ch=1).cuda()
    unet1.load_state_dict(torch.load('D:/00 论文相关/毕设/实验/恢复结果/0105-圆孔-2pi/best.pth'))
    # 用于训练网络的原始图像及其傅里叶intensity
    input = np.load('D:/00 论文相关/毕设/实验/恢复结果/0105-圆孔-2pi/f.npy')
    input = input / np.max(input)
    input = np.sqrt(input)
    input = np.fft.ifftshift(input)
    magnitudes_t = Variable(data_transform(input).unsqueeze(0))

    # phase shift问题，减最小值处理
    img_size = 1024
    full_size_aperture = 34.31
    tmp = np.arange(-full_size_aperture/2, full_size_aperture/2+full_size_aperture/(img_size-1), full_size_aperture/(img_size-1))
    print(tmp[-1]-tmp[0])
    x, y = np.meshgrid(tmp, tmp)
    diameter = 10
    defocus_term = (x**2+y**2) * 2 * np.pi / (632.8e-6 * 2 * 200**2) * 50# 单位mm
    defocus_term[np.sqrt(x**2+y**2)>diameter/2] = 0
    defocus_term = Variable(data_transform(defocus_term).unsqueeze(0)).type(dtype)
    retrieved_phase_sys, out_d1, out_d2, out_d3, _, _, _ = unet1(magnitudes_t.type(dtype), defocus_term)
    retrieved_phase_sys = retrieved_phase_sys.data.cpu().squeeze().numpy()
    print(retrieved_phase_sys.shape)

    min_0 = np.min(retrieved_phase_sys[np.sqrt(x**2+y**2)<diameter/2])
    retrieved_phase_sys[np.sqrt(x**2+y**2)<diameter/2] -= min_0 # 外面不减里面减

    # ----- 加载样品结果 ------------
    unet2 = U_Net(in_ch=1, out_ch=1).cuda()
    unet2.load_state_dict(torch.load('D:/00 论文相关/毕设/实验/恢复结果/0105-zju-2pi-2/best.pth'))
    # 用于训练网络的原始图像及其傅里叶intensity
    input = np.load('D:/00 论文相关/毕设/实验/恢复结果/0105-zju-2pi-2/f.npy')
    input = input / np.max(input)
    input = np.sqrt(input)
    input = np.fft.ifftshift(input)
    magnitudes_t = Variable(data_transform(input).unsqueeze(0))

    retrieved_phase, out_d1, out_d2, out_d3, _, _, _ = unet2(magnitudes_t.type(dtype), defocus_term)
    retrieved_phase = retrieved_phase.data.cpu().squeeze().numpy()
    print(retrieved_phase.shape)

    min_0 = np.min(retrieved_phase[np.sqrt(x**2+y**2)<diameter/2])
    retrieved_phase[np.sqrt(x**2+y**2)<diameter/2] -= min_0 # 外面不减里面减

    # --- 对结果进行低通滤波
    print('max and min of sys: ', np.max(retrieved_phase_sys), np.min(retrieved_phase_sys))
    print('max and min of zju: ', np.max(retrieved_phase), np.min(retrieved_phase))

    retrieved_phase_sys = cv2.medianBlur(retrieved_phase_sys, 3)
    # retrieved_phase = cv2.medianBlur(retrieved_phase, 3)
    # 双边滤波
    # retrieved_phase = cv2.bilateralFilter(retrieved_phase, 5,0.3, 1)

    # --------- 校正 -------------
    retrieved_phase_corr = retrieved_phase - retrieved_phase_sys
    retrieved_phase_corr = cv2.medianBlur(retrieved_phase_corr, 3)
    retrieved_phase_corr = cv2.bilateralFilter(retrieved_phase_corr, 5,0.3, 1)

    # 二维图
    plt.figure()
    plt.imshow(retrieved_phase_sys[300:-300, 300:-300]/(2*np.pi), cmap='gray')
    plt.axis('off')
    plt.title('result_sys')
    plt.colorbar()
    plt.figure()
    # plt.imshow(retrieved_phase/(2*np.pi), cmap='gray', vmin=0, vmax=0.5)
    plt.imshow(retrieved_phase[300:-300, 300:-300]/(2*np.pi), cmap='gray')
    # plt.colorbar(ticks=[0, 0.5])
    plt.axis('off')
    plt.title('result_tmp')
    plt.colorbar()

    plt.figure()
    # plt.imshow(retrieved_phase/(2*np.pi), cmap='gray', vmin=0, vmax=0.5)
    plt.imshow(retrieved_phase_corr[300:-300, 300:-300]/(2*np.pi), cmap='gray')
    # plt.colorbar(ticks=[0, 0.5])
    plt.plot(range(100, 300), [212]*200, color=pcolor['red'], linestyle='--', linewidth=4)
    plt.axis('off')
    plt.title('result_zju')
    plt.colorbar()
    # 一维图
    plt.figure()
    plt.plot(range(1024), retrieved_phase_corr[512, :]/(2*np.pi), color=pcolor['red'], linestyle='-', linewidth=2)  # 看下全貌
    # plt.plot(range(100, 300), retrieved_phase_corr[512, 400:-424]/(2*np.pi), color=pcolor['red'], linestyle='-', linewidth=2)
    plt.legend(['experimental results'], fontsize=20, loc='upper left')
    # plt.ylim([0,0.5])
    # plt.xlim([75,150])
    # plt.yticks([0,0.25,0.5])
    # plt.xticks([75, 112, 150], fontsize=20)
    # plt.yticks(fontsize=20)


    # 三维图
    fig = plt.figure()
    x, y = np.meshgrid(range(424), range(424))

    # 根据多项式拟合结果减去那个离焦面
    x1, y1 = x-212, y-212
    coff = [-2.224758e-04, -1.040499e-03, 2.35139]
    generate = (x1**2+y1**2)*coff[0] + (x1+y1)*coff[1] + coff[2]
    generate = generate / (2*np.pi)

    # final = tmp-generate
    # final = final - np.min(final)
    # plt.imshow(final[140:270, :], cmap='gray')
    # plt.colorbar()

    # ax = plt.axes(projection='3d')

    tmp = retrieved_phase_corr[300:-300, 300:-300]/(2*np.pi)
    # tmp = tmp - generate
    copy = tmp.copy()
    tmp[copy!=0] = tmp[copy!=0] - generate[copy!=0]

    test = tmp[142:271, 70:353]
    # test = tmp[142:271, 75:348]
    plt.figure()
    # plt.plot(test[test.shape[0]//2, :])
    plt.imshow(test, vmin=-0.2, vmax=0.5)
    # test[test==0] = None
    print('result pv: %f rad' % (2*np.pi*(np.max(test)-np.min(test))))
    print('result rms: %f rad' % (2*np.pi*(np.sqrt(np.sum(test**2)/(test.shape[0]*test.shape[1])))))
    plt.colorbar(ticks=[-0.2, 0,  0.5])
    plt.show()

    plt.imshow(tmp, cmap='gray')
    plt.axis('off')
    plt.title('result')
    plt.colorbar()

    tmp[copy==0] = None
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x, y, tmp[::-1, :], rstride=10, cstride=10, cmap='jet', edgecolor='none', vmin=0, vmax=0.5)
    ax.set_zticks([-0.2,0.15,0.5])
    ax.view_init(azim=-124, elev=83)
    ax.set_title('sample')
    # plt.contour3D(x, y, )

    # # 根据多项式拟合结果减去那个离焦面
    # x, y = x-212, y-212
    # coff = [-2.214758e-04, 3.140499e-03, 2.55139]
    # generate = (x**2+y**2)*coff[0] + (x+y)*coff[1] + coff[2]
    # # plt.figure()
    # # plt.imshow(generate)
    # generate = generate / (2*np.pi)
    #
    # plt.figure()
    # final = tmp-generate
    # # final = final - np.min(final)
    # plt.imshow(final[140:270, :], cmap='gray')
    # plt.colorbar()

    # 多项式拟合
    # plt.figure()
    # plt.plot(range(img_size), retrieved_phase_corr[:, img_size//2])
    # f1 = np.polyfit(range(-145, 150), retrieved_phase_corr[365:660, img_size//2], 4)
    # print(f1)
    # p1 = np.poly1d(f1)
    # fit_value = p1(range(-145, 150))
    # plt.plot(range(365, 660), fit_value)
    # f1 = np.polyfit(range(-119, 120), retrieved_phase[391:630, img_size//2], 2)
    # print(f1)
    # p1 = np.poly1d(f1)
    # fit_value = p1(range(-119, 120))
    # plt.plot(range(391, 630), fit_value)
    plt.show()

def correct():
    zju_result = np.load('result_zju.npy')
    system_result = np.load('result_system.npy')
    corrected_result = zju_result - system_result
    corrected_result -= np.min(corrected_result)
    plt.figure()
    # plt.imshow(corrected_result/(2*np.pi), cmap='gray', vmin=0, vmax=0.35)
    plt.imshow(corrected_result/(2*np.pi), cmap='gray')
    plt.axis('off')
    plt.title('retrieved_phase')
    plt.colorbar()
    plt.show()

def main():
    inference_exp()
    # correct()

main()
