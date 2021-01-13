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

from fit import fit, fit_exp
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


def retrieve_exp(I1, I2, I3, I4, defocus_term):
    unet = U_Net(in_ch=1, out_ch=1).cuda()
    mse_loss, mse_loss2, net_input, unet = fit_exp(net=unet,
                                    net_input=I1.type(dtype),
                                    ref_intensity1=I2.type(dtype),
                                    ref_intensity2=I3.type(dtype),
                                    ref_intensity3=I4.type(dtype),
                                    num_iter=10000,
                                    LR=0.1,
                                    lr_decay_epoch=0,
                                    add_noise=False,
                                    gt=None,
                                    cosLR=False,
                                    reducedLR=False,
                                    defocus_term=defocus_term)

    best_unet = U_Net(in_ch=1, out_ch=1).cuda()
    best_unet.load_state_dict(torch.load('best.pth'))
    output, out_d1, out_d2, out_d3, _, _, _ = best_unet(I1.type(dtype), defocus_term.type(dtype))
    output = output.data.cpu().squeeze(0).numpy()[0]
    out_d1 = out_d1.data.cpu().squeeze(0).numpy()[0]
    out_d2 = out_d2.data.cpu().squeeze(0).numpy()[0]
    out_d3 = out_d3.data.cpu().squeeze(0).numpy()[0]
    return mse_loss, mse_loss2, output, out_d1, out_d2, out_d3

def test():
    # ---- read and process intensity images ---- #
    # bkg1 = np.load('exp/w_sample/f.npy')
    # 目前有三处孔径设置，1为defocus处，2为net.py的cyl层，3为fit.py的calc_intensity处
    # 三处size，net的cyl，calc_intensity的cyl
    img_size = 1024

    # 无样品圆孔
    # j1 = np.load('./exp/1221/wo_sample/combined/%s/f_less_exposed.npy' % img_size)
    # df1 = np.load('./exp/1221/wo_sample/combined/%s/df4.npy' % img_size)
    # df2 = np.load('./exp/1221/wo_sample/combined/%s/df7.npy' % img_size)
    # df3 = np.load('./exp/1221/wo_sample/combined/%s/df10.npy' % img_size)

    # 无样品 圆孔去噪
    # j1 = np.load('./exp/1221/wo_sample/combined/denoised/%s/f.npy' % img_size)
    # df1 = np.load('./exp/1221/wo_sample/combined/denoised/%s/df4.npy' % img_size)
    # df2 = np.load('./exp/1221/wo_sample/combined/denoised/%s/df7.npy' % img_size)
    # df3 = np.load('./exp/1221/wo_sample/combined/denoised/%s/df10.npy' % img_size)

    # # 有样品圆孔（圆环板）
    # j1 = np.load('./exp/1221/w_sample/combined/%s/f.npy' % img_size)
    # df1 = np.load('./exp/1221/w_sample/combined/%s/df4.npy' % img_size)
    # df2 = np.load('./exp/1221/w_sample/combined/%s/df7.npy' % img_size)
    # df3 = np.load('./exp/1221/w_sample/combined/%s/df10.npy' % img_size)

    # 有样品圆孔去噪 孔径5mm
    # j1 = np.load('./exp/1221/w_sample/combined/denoised/%s/f.npy' % img_size)
    # df1 = np.load('./exp/1221/w_sample/combined/denoised/%s/df4.npy' % img_size)
    # df2 = np.load('./exp/1221/w_sample/combined/denoised/%s/df7.npy' % img_size)
    # df3 = np.load('./exp/1221/w_sample/combined/denoised/%s/df10.npy' % img_size)
    # print(np.min(j1), np.min(df1), np.min(df2), np.min(df3))

    # 无样品圆孔去噪 孔径10mm
    # j1 = np.load('./exp/1228/wo_sample/combined/denoised/%s/f.npy' % img_size)
    # df1 = np.load('./exp/1228/wo_sample/combined/denoised/%s/df4.npy' % img_size)
    # df2 = np.load('./exp/1228/wo_sample/combined/denoised/%s/df7.npy' % img_size)
    # df3 = np.load('./exp/1228/wo_sample/combined/denoised/%s/df10.npy' % img_size)

    # 有样品zju去噪 孔径10mm
    # j1 = np.load('./exp/1228/w_sample/combined/denoised/%s/f.npy' % img_size)
    # df1 = np.load('./exp/1228/w_sample/combined/denoised/%s/df4.npy' % img_size)
    # df2 = np.load('./exp/1228/w_sample/combined/denoised/%s/df7_2.npy' % img_size)
    # df3 = np.load('./exp/1228/w_sample/combined/denoised/%s/df10_2.npy' % img_size)

    # 有样品zju去噪 孔径10mm by 1230
    # j1 = np.load('./exp/1230/w_sample/combined/denoised/%s/f.npy' % img_size)
    # df1 = np.load('./exp/1230/w_sample/combined/denoised/%s/df4.npy' % img_size)
    # df2 = np.load('./exp/1230/w_sample/combined/denoised/%s/df7_2.npy' % img_size)
    # df3 = np.load('./exp/1230/w_sample/combined/denoised/%s/df10.npy' % img_size)

    # 有样品zju去噪 孔径10mm by 0103
    # j1 = np.load('./exp/0103/w_sample/combined/denoised/%s/f.npy' % img_size)
    # df1 = np.load('./exp/0103/w_sample/combined/denoised/%s/df4.npy' % img_size)
    # df2 = np.load('./exp/0103/w_sample/combined/denoised/%s/df7.npy' % img_size)
    # df3 = np.load('./exp/0103/w_sample/combined/denoised/%s/df10.npy' % img_size)

    # 有样品zju去噪 孔径10mm by 0105
    j1 = np.load('./exp/0105/w_sample/combined/denoised/%s/f_2.npy' % img_size)
    df1 = np.load('./exp/0105/w_sample/combined/denoised/%s/df4.npy' % img_size)
    df2 = np.load('./exp/0105/w_sample/combined/denoised/%s/df7.npy' % img_size)
    df3 = np.load('./exp/0105/w_sample/combined/denoised/%s/df10.npy' % img_size)

    # 无样品圆孔去噪 孔径10mm by 0105
    # j1 = np.load('./exp/0105/wo_sample/combined/denoised/%s/f.npy' % img_size)
    # df1 = np.load('./exp/0105/wo_sample/combined/denoised/%s/df4.npy' % img_size)
    # df2 = np.load('./exp/0105/wo_sample/combined/denoised/%s/df7.npy' % img_size)
    # df3 = np.load('./exp/0105/wo_sample/combined/denoised/%s/df10.npy' % img_size)

    # 试下仿真的圆孔恢复结果？能很快恢复到较好结果
    # j1 = np.load('./exp/1221/wo_sample/combined/%s/test_f.npy' % img_size)
    # df1 = np.load('./exp/1221/wo_sample/combined/%s/test_df4.npy' % img_size)
    # df2 = np.load('./exp/1221/wo_sample/combined/%s/test_df7.npy' % img_size)
    # df3 = np.load('./exp/1221/wo_sample/combined/%s/test_df10.npy' % img_size)

    # 中心有偏移的仿真圆孔
    # j1 = np.load('./exp/1221/wo_sample/combined/%s/shift_test_f.npy' % img_size)
    # df1 = np.load('./exp/1221/wo_sample/combined/%s/shift_test_df4.npy' % img_size)
    # df2 = np.load('./exp/1221/wo_sample/combined/%s/shift_test_df7.npy' % img_size)
    # df3 = np.load('./exp/1221/wo_sample/combined/%s/shift_test_df10.npy' % img_size)


    # normalization
    j1 = j1 / np.max(j1)
    df1 = df1 / np.max(df1)
    df2 = df2 / np.max(df2)
    df3 = df3 / np.max(df3)

    # convert intensities to amplitudes
    j1 = np.sqrt(j1)
    df1 = np.sqrt(df1)
    df2 = np.sqrt(df2)
    df3 = np.sqrt(df3)

    plt.subplot(221)
    plt.imshow(j1)
    plt.subplot(222)
    plt.imshow(df1)
    plt.subplot(223)
    plt.imshow(df2)
    plt.subplot(224)
    plt.imshow(df3)
    plt.show()

    # ?是否需要shift？（因为fit中的计算强度是未shift的）
    j1 = np.fft.ifftshift(j1)
    df1 = np.fft.ifftshift(df1)
    df2 = np.fft.ifftshift(df2)
    df3 = np.fft.ifftshift(df3)

    j1 = Variable(data_transform(j1).unsqueeze(0)).type(dtype)
    df1 = Variable(data_transform(df1).unsqueeze(0)).type(dtype)
    df2 = Variable(data_transform(df2).unsqueeze(0)).type(dtype)
    df3 = Variable(data_transform(df3).unsqueeze(0)).type(dtype)

    # ---- generate the defocus term according to the radius of aperture ---- #
    # 成像透镜半径 1cm？1.5cm？待测  相机像素大小3.69μm？待确认
    # full_size_aperture = 24000//3.69 # 目测透镜直径2.4cm
    # full_size_aperture = 24
    full_size_aperture = 34.31# 42.88   #单位mm
    # tmp = np.arange(-full_size_aperture/2+full_size_aperture/img_size, full_size_aperture/2+full_size_aperture/img_size, full_size_aperture/img_size)
    tmp = np.arange(-full_size_aperture/2, full_size_aperture/2+full_size_aperture/(img_size-1), full_size_aperture/(img_size-1))
    print(tmp.shape)
    x, y = np.meshgrid(tmp, tmp)
    diameter = 10
    # defocus_term = 2 * (x**2+y**2) - 1
    defocus_term = -(x**2+y**2) * 2 * np.pi / (632.8e-6 * 2 * 200**2) * 50# 单位mm
    # defocus_term = -(x**2+y**2) * 2 * np.pi / (632.991e-6 * 8 * 250**2 / full_size_aperture**2) * 4 / full_size_aperture**2# 单位mm
    defocus_term[np.sqrt(x**2+y**2)>diameter/2] = 0
    defocus_term = Variable(data_transform(defocus_term).unsqueeze(0)).type(dtype)
    # 注意，实际实验是圆形口径，而仿真是方形口径，改为圆形相当于增加约束，fit函数需修改
    # plt.figure()
    # plt.imshow(defocus_term)
    # plt.show()

    # ---- retrieve phase ---- #
    # check list:
    # out相位范围是否正确？
    # defocus param范围是否正确？
    mse_loss, mse_loss2, retrieved_phase, out_d1, out_d2, out_d3 = retrieve_exp(j1, df1, df2, df3, defocus_term)

    retrieved_phase -= np.min(retrieved_phase)
    out_d1 -= np.min(out_d1)
    out_d2 -= np.min(out_d2)
    out_d3 -= np.min(out_d3)
    # 减去全局最小值好像应该改为减去支持域内最小值？否则全局最小值固定就是0（外圈）
    min_0 = np.min(retrieved_phase[np.sqrt(x**2+y**2)<diameter/2]) # 好像应该以同一个为准
    # min_1 = np.min(out_d1[np.sqrt(x**2+y**2)<diameter/2])
    # min_2 = np.min(out_d2[np.sqrt(x**2+y**2)<diameter/2])
    # min_3 = np.min(out_d3[np.sqrt(x**2+y**2)<diameter/2])
    retrieved_phase[np.sqrt(x**2+y**2)<diameter/2] -= min_0 # 外面不减里面减
    out_d1[np.sqrt(x**2+y**2)<diameter/2] -= min_0
    out_d2[np.sqrt(x**2+y**2)<diameter/2] -= min_0
    out_d3[np.sqrt(x**2+y**2)<diameter/2] -= min_0

    # tmp = np.arange(-full_size_aperture/2+full_size_aperture/img_size, full_size_aperture/2+full_size_aperture/img_size, full_size_aperture/img_size)
    x, y = np.meshgrid(tmp, tmp)
    aperture = np.zeros((img_size,img_size))
    aperture[np.sqrt(x**2+y**2)<=diameter/2] = 1
    retrieved_intensity = np.square(np.abs(np.fft.fftshift(np.fft.fft2(aperture*np.exp(1j*retrieved_phase)))))
    retrieved_df1 = np.square(np.abs(np.fft.fftshift(np.fft.fft2(aperture*np.exp(1j*out_d1)))))
    retrieved_df2 = np.square(np.abs(np.fft.fftshift(np.fft.fft2(aperture*np.exp(1j*out_d2)))))
    retrieved_df3 = np.square(np.abs(np.fft.fftshift(np.fft.fft2(aperture*np.exp(1j*out_d3)))))


    plt.figure()
    plt.imshow(retrieved_phase, cmap='jet')
    plt.title('retrieved_phase')
    plt.axis('off')
    plt.colorbar()

    retrieved_intensity = retrieved_intensity / np.max(retrieved_intensity)
    retrieved_df1 = retrieved_df1 / np.max(retrieved_df1)
    retrieved_df2 = retrieved_df2 / np.max(retrieved_df2)
    retrieved_df3 = retrieved_df3 / np.max(retrieved_df3)
    plt.figure()
    plt.subplot(241)
    j1 = j1.cpu().squeeze()
    j1 = np.fft.fftshift(j1)
    j1 = np.square(j1)
    plt.imshow(j1, cmap='jet')
    plt.title('original_intensity')
    plt.axis('off')
    plt.colorbar()
    plt.subplot(245)
    plt.imshow(retrieved_intensity, cmap='jet')
    plt.title('retrieved_intensity')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(242)
    df1 = df1.cpu().squeeze()
    df1 = np.fft.fftshift(df1)
    df1 = np.square(df1)
    plt.imshow(df1, cmap='jet')
    plt.title('original_intensity_df1')
    plt.axis('off')
    plt.colorbar()
    plt.subplot(246)
    plt.imshow(retrieved_df1, cmap='jet')
    plt.title('retrieved_df1')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(243)
    df2 = df2.cpu().squeeze()
    df2 = np.fft.fftshift(df2)
    df2 = np.square(df2)
    plt.imshow(df2, cmap='jet')
    plt.title('original_intensity_df2')
    plt.axis('off')
    plt.colorbar()
    plt.subplot(247)
    plt.imshow(retrieved_df2, cmap='jet')
    plt.title('retrieved_df2')
    plt.axis('off')
    plt.colorbar()

    plt.subplot(244)
    df3 = df3.cpu().squeeze()
    df3 = np.fft.fftshift(df3)
    df3 = np.square(df3)
    plt.imshow(df3, cmap='jet')
    plt.title('original_intensity_df3')
    plt.axis('off')
    plt.colorbar()
    plt.subplot(248)
    plt.imshow(retrieved_df3, cmap='jet')
    plt.title('retrieved_df3')
    plt.axis('off')
    plt.colorbar()

    plt.figure()
    plt.plot(range(img_size), retrieved_intensity[:, img_size//2], color='b')
    plt.plot(range(img_size), j1[:, img_size//2], color='r')

    plt.figure()
    plt.plot(range(img_size), retrieved_df1[:, img_size//2], color='b')
    plt.plot(range(img_size), df1[:, img_size//2], color='r')

    plt.figure()
    plt.plot(range(img_size), retrieved_df2[:, img_size//2], color='b')
    plt.plot(range(img_size), df2[:, img_size//2], color='r')

    plt.figure()
    plt.plot(range(img_size), retrieved_df3[:, img_size//2], color='b')
    plt.plot(range(img_size), df3[:, img_size//2], color='r')

    plt.figure()
    plt.plot(range(len(mse_loss)), mse_loss)

    plt.figure()
    plt.plot(range(0, len(mse_loss), 10), mse_loss2)

    plt.show()


if __name__ == '__main__':
    test()
