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

from fit import fit, fit_fresnel
from net import U_Net, Slice_Net, Slim_Net
from util import data_transform, two_step_prop_fresnel, cyl

PHASE_RANGE = 1

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# setup_seed(20)

GPU = True
if GPU == True:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("num GPUs",torch.cuda.device_count())
else:
    dtype = torch.FloatTensor


def psnr(x_hat,x_true,maxv=1.):
    x_hat = x_hat.flatten()
    x_true = x_true.flatten()
    mse=np.mean(np.square(x_hat-x_true))
    print('mse in psnr: ', mse)
    psnr_ = 10.*np.log(maxv**2/mse)/np.log(10.)
    return psnr_

def retrieve_fresnel(measured_intensity, measured_intensity1, measured_intensity2, phase_obj, defocus1, defocus2):
    unet = U_Net(in_ch=1, out_ch=1).cuda()
    # unet = Slim_Net(in_ch=1, out_ch=1).cuda()
    mse_loss, mse_loss2, net_input, unet = fit_fresnel(net=unet,
                                    net_input=measured_intensity.type(dtype),
                                    ref_intensity1=measured_intensity1.type(dtype),
                                    ref_intensity2=measured_intensity2.type(dtype),
                                    num_iter=10000,
                                    LR=0.1,
                                    lr_decay_epoch=0,
                                    add_noise=False,
                                    gt=Variable(torch.tensor(phase_obj)),
                                    cosLR=False,
                                    defocus1=defocus1,
                                    defocus2=defocus2)


    # output = unet(net_input.type(dtype)).data.cpu().squeeze(0).numpy()[0]
    best_unet = U_Net(in_ch=1, out_ch=1).cuda()
    best_unet.load_state_dict(torch.load('best.pth'))
    output = best_unet(measured_intensity.type(dtype)).data.cpu().squeeze(0).numpy()[0]
    # output = unet(measured_intensity.type(dtype)).data.cpu().squeeze(0).numpy()[0]

    return mse_loss, mse_loss2, output

def retrieve(measured_intensity,
             measured_intensity1, measured_intensity2, measured_intensity3,
             phase_obj,
             phase_obj1, phase_obj2, phase_obj3,
             phase_modulate1, phase_modulate2, phase_modulate3,
             defocus_term):
    unet = U_Net(in_ch=1, out_ch=1).cuda()
    # unet = Slim_Net(in_ch=1, out_ch=1).cuda()
    mse_loss, mse_loss2, net_input, unet = fit(net=unet,
                                    net_input=measured_intensity.type(dtype),
                                    ref_intensity1=measured_intensity1.type(dtype),
                                    ref_intensity2=measured_intensity2.type(dtype),
                                    ref_intensity3=measured_intensity3.type(dtype),
                                    num_iter=10000,
                                    LR=0.1,
                                    lr_decay_epoch=0,
                                    add_noise=False,
                                    gt=Variable(torch.tensor(phase_obj)),
                                    cosLR=False,
                                    reducedLR=False,
                                    modulate1=Variable(torch.tensor(phase_modulate1)),
                                    modulate2=Variable(torch.tensor(phase_modulate2)),
                                    modulate3=Variable(torch.tensor(phase_modulate3)),
                                    defocus_term=defocus_term)
                                    # modulate1=None,
                                    # modulate2=Variable(torch.tensor(phase_modulate2)))

    # output = unet(net_input.type(dtype)).data.cpu().squeeze(0).numpy()[0]
    best_unet = U_Net(in_ch=1, out_ch=1).cuda()
    best_unet.load_state_dict(torch.load('best.pth'))
    output, _, _, _, _, _, _ = best_unet(measured_intensity.type(dtype), defocus_term.type(dtype))
    output = output.data.cpu().squeeze(0).numpy()[0]
    # output = unet(measured_intensity.type(dtype)).data.cpu().squeeze(0).numpy()[0]

    return mse_loss, mse_loss2, output

def run():
    # np.random.seed(233)
    # image = imageio.imread('cameraman.png', as_gray=True)
    # image = cv2.imread('mandril_gray.tif', cv2.IMREAD_GRAYSCALE)
    image = cv2.imread('zju.jpg', cv2.IMREAD_GRAYSCALE)
    image = (255 - image)*0.5   # 使用test.jpg的时候记得打开这个 否则边缘为pi 中间为0
    print('original image shape ', image.shape)
    image = cv2.resize(image, (256,256))
    # 支撑域？这个概念的理解还有点问题
    # reduced_image = cv2.resize(image, (128, 128))
    # image = np.zeros((256, 256))
    # image[64:192, 64:192] = reduced_image

    print('original image max ', np.max(image))
    # 如果是phase object，对灰度图缩放到0-2π之间
    phase_obj = image / 255 * 2 * np.pi *PHASE_RANGE #* 0.5
    print('max and min of phase_obj: ', np.max(phase_obj), np.min(phase_obj))
    # network input
    magnitudes = np.abs(np.fft.fft2(np.exp(1j*phase_obj)))  # 归一化振幅的相位调制
    print('original amplitude max: %f   min: %f' % (np.max(magnitudes), np.min(magnitudes)))
    # 增加一个离焦项，使用泽尼克来实现  # 可以是离焦调制，也完全可以是其他已知的调制？
    # deep phase decoder相当于是加了相位调制？
    # -----
    defocus_term = imageio.imread('modulate1.jpg', as_gray=True)
    defocus_term = cv2.resize(defocus_term, (256, 256))
    # defocus_term = defocus_term / 255 * 2 * np.pi    # 先在这一项里设出2pi，这样系数回归区间只需要在0-1
    defocus_term = defocus_term / 255  # 已经值net输出位置设置了乘2pi
    defocus_term = Variable(data_transform(defocus_term)).type(dtype)
    # -----
    modulate_phase_image1 = imageio.imread('modulate1.jpg', as_gray=True)
    modulate_phase_image1 = cv2.resize(modulate_phase_image1, (256,256))
    phase_modulate1 = modulate_phase_image1 / 255 * 2 * np.pi * 5.12          # 这个离焦参数 设为网络自动学习？
    phase_obj1 = phase_obj + phase_modulate1

    # 加两种不同的调制试试
    modulate_phase_image2 = cv2.imread('modulate1.jpg', cv2.IMREAD_GRAYSCALE)
    modulate_phase_image2 = cv2.resize(modulate_phase_image2, (256, 256))
    phase_modulate2 = modulate_phase_image2 / 255 * 2 * np.pi * 10.81
    phase_obj2 = phase_obj + phase_modulate2

    # 三图调制
    modulate_phase_image3 = cv2.imread('modulate1.jpg', cv2.IMREAD_GRAYSCALE)
    modulate_phase_image3 = cv2.resize(modulate_phase_image3, (256, 256))
    phase_modulate3 = modulate_phase_image3 / 255 * 2 * np.pi * 15.77
    phase_obj3 = phase_obj + phase_modulate3

    # magnitudes = np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j*phase_obj))))
    # 若此处shift，则计算loss时也要shift
    magnitudes1 = np.abs(np.fft.fft2(np.exp(1j*phase_obj1)))
    print('rmse between in-focus and defocus intensity 1: ', np.sqrt(np.mean((magnitudes-magnitudes1)**2)))
    magnitudes2 = np.abs(np.fft.fft2(np.exp(1j*phase_obj2)))
    print('rmse between in-focus and defocus intensity 2: ', np.sqrt(np.mean((magnitudes-magnitudes2)**2)))
    magnitudes3 = np.abs(np.fft.fft2(np.exp(1j*phase_obj3)))
    print('rmse between in-focus and defocus intensity 3: ', np.sqrt(np.mean((magnitudes-magnitudes3)**2)))

    # ---- 灰度级离散化 -----
    magnitudes = np.square(magnitudes)
    magnitudes1 = np.square(magnitudes1)
    magnitudes2 = np.square(magnitudes2)
    magnitudes3 = np.square(magnitudes3)
    plt.figure()
    plt.imshow(magnitudes2)
    plt.show()
    # 14位
    print("max of intensity: %d %d %d %d" % (np.max(magnitudes),np.max(magnitudes1),np.max(magnitudes2),np.max(magnitudes3)))
    # focus的中心能量最高，用它

    # 实际中，每个焦面的光强无法用统一值做归一化，因为曝光时间不一样！
    # magnitudes = np.floor(magnitudes / np.max(magnitudes) * np.power(2, 14)) # 这个floor肯定导致了一些零值
    # magnitudes1 = np.floor(magnitudes1 / np.max(magnitudes1) * np.power(2, 14))
    # magnitudes2 = np.floor(magnitudes2 / np.max(magnitudes2) * np.power(2, 14))
    # magnitudes3 = np.floor(magnitudes3 / np.max(magnitudes3) * np.power(2, 14))
    # 过曝
    bit_camera = 14
    exposure_time = 10
    magnitudes = np.mod(np.floor(magnitudes / np.max(magnitudes) * np.power(2, bit_camera) * exposure_time), np.power(2, bit_camera)) # 这个floor肯定导致了一些零值
    magnitudes1 = np.mod(np.floor(magnitudes1 / np.max(magnitudes1) * np.power(2, bit_camera) * exposure_time), np.power(2, bit_camera))
    magnitudes2 = np.mod(np.floor(magnitudes2 / np.max(magnitudes2) * np.power(2, bit_camera) * exposure_time), np.power(2, bit_camera))
    magnitudes3 = np.mod(np.floor(magnitudes3 / np.max(magnitudes3) * np.power(2, bit_camera) * exposure_time), np.power(2, bit_camera))
    # 实际中肯定不可能刚好归一化，而是先过曝
    # normalize_factor = np.max(magnitudes)
    # magnitudes = np.floor(magnitudes / normalize_factor * np.power(2, 14))
    # magnitudes1 = np.floor(magnitudes1 / normalize_factor * np.power(2, 14))
    # magnitudes2 = np.floor(magnitudes2 / normalize_factor * np.power(2, 14))
    # magnitudes3 = np.floor(magnitudes3 / normalize_factor * np.power(2, 14))

    # 实际中光强未知，干脆用归一化的intensity计算loss?
    magnitudes /= np.power(2, bit_camera)
    magnitudes1 /= np.power(2, bit_camera)
    magnitudes2 /= np.power(2, bit_camera)
    magnitudes3 /= np.power(2, bit_camera)

    # 16位

    # 还原
    magnitudes = np.sqrt(magnitudes)
    magnitudes1 = np.sqrt(magnitudes1)
    magnitudes2 = np.sqrt(magnitudes2)
    magnitudes3 = np.sqrt(magnitudes3)
    # ----------------------

    # 保存三个intensity
    # np.save('intensity1.npy', magnitudes**2)
    # np.save('intensity2.npy', magnitudes1**2)
    # np.save('intensity3.npy', magnitudes2**2)


    # plt.subplot(121)
    # plt.imshow(magnitudes1, cmap='gray')
    # plt.title('magnitudes1')
    # plt.subplot(122)
    # plt.imshow(magnitudes2, cmap='gray')
    # plt.title('magnitudes2')
    # plt.show()
    # plt.figure()
    # plt.plot(magnitudes[0, :], color='r')
    # plt.plot(magnitudes1[0, :], color='g')
    # plt.plot(magnitudes2[0, :], color='b')
    # plt.show()
    # magnitudes_t = torch.Tensor(magnitudes)
    magnitudes_t = Variable(data_transform(magnitudes).unsqueeze(0))        # 三个振幅 而非强度
    magnitudes_t1 = Variable(data_transform(magnitudes1).unsqueeze(0))
    magnitudes_t2 = Variable(data_transform(magnitudes2).unsqueeze(0))
    magnitudes_t3 = Variable(data_transform(magnitudes3).unsqueeze(0))


    # 添加正态分布的随机噪声？
    # noise = magnitudes_t.clone()
    # noise1 = magnitudes_t1.clone()
    # noise2 = magnitudes_t2.clone()
    # magnitudes_t += Variable(noise.normal_()*10)
    # # print('psnr of the original image and noises: ', psnr(magnitudes_t.data.numpy()/np.max(tmp), tmp/np.max(tmp)))
    # # plt.subplot(221)
    # # plt.imshow(tmp.squeeze())
    # # plt.subplot(222)
    # # plt.imshow(magnitudes_t.squeeze().numpy())
    # # plt.subplot(212)
    # # plt.plot(tmp.squeeze()[0,:], color='r')
    # # plt.plot(magnitudes_t.squeeze().numpy()[0,:], color='b')
    # # plt.show()
    # magnitudes_t1 += Variable(noise1.normal_()*10)
    # magnitudes_t2 += Variable(noise2.normal_()*10)

    # 多调制
    mse_loss, mse_loss2, retrieved_phase = retrieve(magnitudes_t, magnitudes_t1, magnitudes_t2, magnitudes_t3,
                                                    phase_obj, phase_obj1, phase_obj2, phase_obj3,
                                                    phase_modulate1, phase_modulate2, phase_modulate3,
                                                    defocus_term)
    # 单调制
    # mse_loss, mse_loss2, retrieved_phase = retrieve(magnitudes_t, magnitudes_t1, magnitudes_t2,
    #                                                 phase_obj, phase_obj1, None,
    #                                                 phase_modulate1, None)
    # 无调制
    # mse_loss, mse_loss2, retrieved_phase = retrieve(magnitudes_t, magnitudes_t1, magnitudes_t2,
    #                                                 phase_obj, None, None,
    #                                                 None, None)



    np.save('mse_loss_2.npy', mse_loss2)

    # retrieved_phase = np.mod(retrieved_phase, 2*np.pi)
    # retrieved_phase *= 2*np.pi*0.5
    # retrieved_phase *= 2*np.pi*PHASE_RANGE
    retrieved_phase *= 1

    # print(phase_obj[150:156, 150:156])
    # print(retrieved_phase[150:156, 150:156])
    # print(phase_obj[0:6, 0:6])
    # print(retrieved_phase[0:6, 0:6])

    # phase shift问题，减最小值处理
    phase_obj -= np.min(phase_obj)
    retrieved_phase -= np.min(retrieved_phase)

    print('phase rmse: ', np.sqrt(np.mean((phase_obj-retrieved_phase)**2)))

    plt.subplot(221)
    plt.imshow(phase_obj, cmap='gray')
    plt.title('phase_obj')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(retrieved_phase, cmap='gray')
    plt.title('retrieved_phase')
    plt.axis('off')
    plt.colorbar()
    plt.subplot(212)
    plt.plot(phase_obj[128, :], color='r')
    plt.plot(retrieved_phase[128, :], color='b')
    plt.show()

    real_part = torch.cos(torch.tensor(retrieved_phase).type(dtype)).unsqueeze(-1)
    image_part = torch.sin(torch.tensor(retrieved_phase).type(dtype)).unsqueeze(-1)
    complex_phase = torch.cat((real_part, image_part), dim=-1).squeeze()
    f_phase = torch.fft(complex_phase, signal_ndim=2)
    re = torch.index_select(f_phase, dim=2, index=torch.tensor(0).type(torch.cuda.LongTensor))
    im = torch.index_select(f_phase, dim=2, index=torch.tensor(1).type(torch.cuda.LongTensor))
    pred_intensity = torch.sqrt(re**2 + im**2).squeeze().data.cpu().squeeze().numpy()

    check_magnitudes = np.abs(np.fft.fft2(np.exp(1j*(retrieved_phase))))
    np.save('phase_obj', phase_obj)
    np.save('retrieved_phase', retrieved_phase)
    np.save('magnitudes', np.abs(np.fft.fft2(np.exp(1j*phase_obj))))
    np.save('check_magnitudes', np.abs(np.fft.fft2(np.exp(1j*(retrieved_phase)))))
    # print(magnitudes[150:156, 150:156])
    # print(pred_intensity[150:156, 150:156])
    # print(magnitudes[0:6, 0:6])
    # print(pred_intensity[0:6, 0:6])
    print('intensity rmse: ', np.sqrt(np.mean((magnitudes-pred_intensity)**2)))
    print('rmse between original magnitudes and fft of pred_phase:', np.sqrt(np.mean((magnitudes-check_magnitudes)**2)))
    print('magnitudes shape: ', magnitudes.shape)
    print('check_magnitudes shape: ', check_magnitudes.shape)

    # plt.subplot(131)
    # plt.imshow(magnitudes, cmap='gray')
    # plt.title('magnitudes')
    # plt.subplot(132)
    # plt.imshow(pred_intensity, cmap='gray')
    # plt.title('retrieved magnitudes')
    # plt.subplot(133)
    # plt.imshow(check_magnitudes, cmap='gray')
    # plt.title('direct fft of retrieved_phase')
    # plt.show()

def run_fresnel():
    # 参数设置
    wavelength = 632.8e-6
    N = 256
    width = 10
    k = 2*np.pi/wavelength
    f = 2
    # z = 1000

    tmp = np.arange(-width//2, width//2, width/N)
    x, y = np.meshgrid(tmp, tmp)
    R = np.sqrt(x**2 + y**2)
    R[R>width/2] = 0
    lens = np.exp(-1j * k * R**2 / (2*f))* cyl(x,y,width/2)

    # phase image 读取
    image = cv2.imread('peppers_gray.tif', cv2.IMREAD_GRAYSCALE)
    print('original image shape ', image.shape)
    image = cv2.resize(image, (256,256))
    print('original image max ', np.max(image))
    # 如果是phase object，对灰度图缩放到0-2π之间
    phase_obj = image / 255 * 2 * np.pi *0.5 #* 0.5
    # phase_obj = 0
    print(np.max(phase_obj), np.min(phase_obj))

    # network input
    u0 = lens * np.exp(1j*phase_obj)
    m_d2_d1 = 0.001
    uz = two_step_prop_fresnel(u0, wavelength, width/N, m_d2_d1*width/N, f)
    magnitudes = np.abs(uz)
    print('original amplitude max: %f   min: %f' % (np.max(magnitudes), np.min(magnitudes)))

    # 增加离焦项
    defocus1 = 0.0005
    defocus2 = 0.0001
    magnitudes1 = np.abs(two_step_prop_fresnel(u0, wavelength, width/N, m_d2_d1*width/N, f+defocus1))
    magnitudes2 = np.abs(two_step_prop_fresnel(u0, wavelength, width/N, m_d2_d1*width/N, f+defocus2))
    print('rmse between in-focus and defocus intensity, first: %f    second: %f'
          % (np.sqrt(np.mean((magnitudes-magnitudes1)**2)), np.sqrt(np.mean((magnitudes-magnitudes2)**2))))

    plt.figure()
    plt.plot(magnitudes[128, :])
    plt.show()

    plt.subplot(131)
    plt.imshow(magnitudes, cmap='gray')
    plt.title('magnitudes')
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(magnitudes1, cmap='gray')
    plt.title('magnitudes1')
    plt.subplot(133)
    plt.imshow(magnitudes2, cmap='gray')
    plt.title('magnitudes2')
    plt.show()
    # plt.figure()
    # plt.plot(magnitudes[0, :], color='r')
    # plt.plot(magnitudes1[0, :], color='g')
    # plt.plot(magnitudes2[0, :], color='b')
    # plt.show()
    # magnitudes_t = torch.Tensor(magnitudes)
    magnitudes_t = Variable(data_transform(magnitudes).unsqueeze(0))
    magnitudes_t1 = Variable(data_transform(magnitudes1).unsqueeze(0))
    magnitudes_t2 = Variable(data_transform(magnitudes2).unsqueeze(0))
    # 添加正态分布的随机噪声？
    # noise = magnitudes_t.clone()
    # noise1 = magnitudes_t1.clone()
    # noise2 = magnitudes_t2.clone()
    # magnitudes_t += Variable(noise.normal_()*10)
    # # print('psnr of the original image and noises: ', psnr(magnitudes_t.data.numpy()/np.max(tmp), tmp/np.max(tmp)))
    # # plt.subplot(221)
    # # plt.imshow(tmp.squeeze())
    # # plt.subplot(222)
    # # plt.imshow(magnitudes_t.squeeze().numpy())
    # # plt.subplot(212)
    # # plt.plot(tmp.squeeze()[0,:], color='r')
    # # plt.plot(magnitudes_t.squeeze().numpy()[0,:], color='b')
    # # plt.show()
    # magnitudes_t1 += Variable(noise1.normal_()*10)
    # magnitudes_t2 += Variable(noise2.normal_()*10)

    # 多调制
    mse_loss, mse_loss2, retrieved_phase = retrieve_fresnel(magnitudes_t, magnitudes_t1, magnitudes_t2,
                                                    phase_obj, defocus1, defocus2)
    # 单调制
    # mse_loss, mse_loss2, retrieved_phase = retrieve_fresnel(magnitudes_t, magnitudes_t1, magnitudes_t2,
    #                                                 phase_obj, defocus1, None)
    # 无调制
    # mse_loss, mse_loss2, retrieved_phase = retrieve_fresnel(magnitudes_t, magnitudes_t1, magnitudes_t2,
    #                                                 phase_obj, None, None)


    np.save('mse_loss_2.npy', mse_loss2)

    # retrieved_phase = np.mod(retrieved_phase, 2*np.pi)
    retrieved_phase *= 2*np.pi*0.5

    # print(phase_obj[150:156, 150:156])
    # print(retrieved_phase[150:156, 150:156])
    # print(phase_obj[0:6, 0:6])
    # print(retrieved_phase[0:6, 0:6])

    # phase shift问题，减最小值处理
    phase_obj -= np.min(phase_obj)
    retrieved_phase -= np.min(retrieved_phase)

    print('phase rmse: ', np.sqrt(np.mean((phase_obj-retrieved_phase)**2)))

    plt.subplot(221)
    plt.imshow(phase_obj, cmap='gray')
    plt.title('phase_obj')
    plt.axis('off')
    plt.subplot(222)
    plt.imshow(retrieved_phase, cmap='gray')
    plt.title('retrieved_phase')
    plt.axis('off')
    plt.colorbar()
    plt.subplot(212)
    plt.plot(phase_obj[128, :], color='r')
    plt.plot(retrieved_phase[128, :], color='b')
    plt.show()

    real_part = torch.cos(torch.tensor(retrieved_phase).type(dtype)).unsqueeze(-1)
    image_part = torch.sin(torch.tensor(retrieved_phase).type(dtype)).unsqueeze(-1)
    complex_phase = torch.cat((real_part, image_part), dim=-1).squeeze()
    f_phase = torch.fft(complex_phase, signal_ndim=2)
    re = torch.index_select(f_phase, dim=2, index=torch.tensor(0).type(torch.cuda.LongTensor))
    im = torch.index_select(f_phase, dim=2, index=torch.tensor(1).type(torch.cuda.LongTensor))
    pred_intensity = torch.sqrt(re**2 + im**2).squeeze().data.cpu().squeeze().numpy()


    check_magnitudes = np.abs(two_step_prop_fresnel(lens * np.exp(1j*retrieved_phase), wavelength, width/N, m_d2_d1*width/N, f))
    np.save('phase_obj', phase_obj)
    np.save('retrieved_phase', retrieved_phase)
    np.save('magnitudes', magnitudes)
    np.save('check_magnitudes', check_magnitudes)
    # print(magnitudes[150:156, 150:156])
    # print(pred_intensity[150:156, 150:156])
    # print(magnitudes[0:6, 0:6])
    # print(pred_intensity[0:6, 0:6])
    print('intensity rmse: ', np.sqrt(np.mean((magnitudes-pred_intensity)**2)))
    print('rmse between original magnitudes and fft of pred_phase:', np.sqrt(np.mean((magnitudes-check_magnitudes)**2)))
    print('magnitudes shape: ', magnitudes.shape)
    print('check_magnitudes shape: ', check_magnitudes.shape)

    # plt.subplot(131)
    # plt.imshow(magnitudes, cmap='gray')
    # plt.title('magnitudes')
    # plt.subplot(132)
    # plt.imshow(pred_intensity, cmap='gray')
    # plt.title('retrieved magnitudes')
    # plt.subplot(133)
    # plt.imshow(check_magnitudes, cmap='gray')
    # plt.title('direct fft of retrieved_phase')
    # plt.show()

if __name__ == '__main__':
    run()
    # run_fresnel()
