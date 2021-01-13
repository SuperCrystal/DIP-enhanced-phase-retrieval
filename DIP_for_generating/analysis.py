import cv2
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
plt.rcParams['font.size'] = 40
# plt.rcParams['font.weight'] = 'bold'

def calc_mse(a, b):
    return np.mean((a-b)**2)

def calc_rmse(a, b):
    return np.sqrt(np.mean((a-b)**2))

def cal_relative_error(a, b):
    return np.abs(a-b)/a

pcolor = {'red': (239/255, 131/255, 119/255),
          'blue': (65/255, 113/255, 156/255),
          'green': (173, 208, 92)}

def main():
    ### 图1
    # path = 'D:/00 论文相关/pr_decoder/figures/1/'
    # for id in range(3):
    #     plt.figure(id+1)
    #     inte = np.load(path+'intensity%d.npy' % (1+id))
    #     plt.imshow(np.log(np.fft.fftshift(inte)), cmap='gray', vmin=-7, vmax=22)
    #     print(np.min(np.log(np.fft.fftshift(inte))), np.max(np.log(np.fft.fftshift(inte))))
    #     # plt.plot([128]*256, range(256), color='r', linestyle='--', linewidth=5)
    #     # plt.imshow(np.fft.fftshift(magnitudes), cmap='gray')
    #     # plt.colorbar(ticks=[-7, 22])
    #     plt.axis('off')
    # plt.show()

    #### 图2 可视化 sample intensity and its cross-section
    image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256,256))

    phase_obj = image / 255 * 2 * np.pi *0.5 #* 0.5
    magnitudes = np.abs(np.fft.fft2(np.exp(1j*phase_obj)))
    plt.figure()
    plt.imshow(phase_obj/(2*np.pi), cmap='gray', vmin=0, vmax=0.5)
    # plt.colorbar(ticks=[np.min(phase_obj), np.max(phase_obj)])
    plt.colorbar(ticks=[0, 0.5])   # 图上注明单位2Π rad
    plt.axis('off')
    plt.show()
    plt.figure()
    plt.imshow(np.log(np.fft.fftshift(magnitudes**2)), cmap='gray', vmin=-7, vmax=22)
    print(np.min(np.log(np.fft.fftshift(magnitudes**2))), np.max(np.log(np.fft.fftshift(magnitudes**2))))
    plt.plot([128]*256, range(256), color=pcolor['red'], linestyle='--', linewidth=5)
    # plt.imshow(np.fft.fftshift(magnitudes), cmap='gray')
    plt.colorbar(ticks=[-7, 22])
    plt.axis('off')
    plt.show()
    plt.figure()
    plt.plot(np.arange(-0.5, 0.5, 1/256), np.log(np.fft.fftshift(magnitudes**2))[image.shape[0]//2, :], color=pcolor['blue'])
    plt.xlim([-0.5, 0.5])
    plt.xticks([-0.5, 0 , 0.5], fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    # 原光强图
    plt.figure()
    plt.imshow(np.fft.fftshift(magnitudes**2), cmap='gray')
    plt.show()
    plt.figure()
    plt.plot(np.arange(-0.5, 0.5, 1/256),np.fft.fftshift(magnitudes**2)[image.shape[0]//2, :], color=pcolor['blue'])
    plt.show()

    # path = './实验/消融实验/'
    # experiments = os.listdir(path)

    # 图3.1 可视化 原图，mse恢复图，wmse恢复图
    # sample_name = 'cameraman'
    # path = './实验/wmse/权重影响/'+sample_name
    # original_phase = np.load(path+'/1/phase_obj.npy')
    # mse_phase = np.load(path+'/1/retrieved_phase.npy')
    # wmse_phase = np.load(path+'/100/retrieved_phase.npy')
    # #
    # top = 200
    # # start, end = 98, 158
    # start, end = 63, 163
    # plt.figure()
    # plt.imshow(original_phase/(2*np.pi), cmap='gray', vmin=0, vmax=0.5)
    # plt.plot(range(start, end), [top]*(end-start), color=pcolor['red'], linestyle='--', linewidth=5)
    # plt.colorbar(ticks=[0, 0.5])
    # plt.axis('off')
    # plt.figure()
    # plt.imshow(mse_phase/(2*np.pi), cmap='gray', vmin=0, vmax=0.5)
    # plt.plot(range(start, end), [top]*(end-start), color=pcolor['red'], linestyle='--', linewidth=5)
    # plt.colorbar(ticks=[0, 0.5])
    # plt.axis('off')
    # plt.figure()
    # plt.imshow(wmse_phase/(2*np.pi), cmap='gray', vmin=0, vmax=0.5)
    # plt.plot(range(start, end), [top]*(end-start), color=pcolor['red'], linestyle='--', linewidth=5)
    # plt.colorbar(ticks=[0, 0.5])
    # plt.axis('off')
    # print('rmse of MSE loss: %f' % calc_rmse(original_phase, mse_phase))
    # print('rmse of WMSE loss: %f' % calc_rmse(original_phase, wmse_phase))
    # # plt.show()
    # #
    # # # 图3.2 三者的截面图
    # plt.figure()
    # plt.plot(range(start, end), (original_phase/(2*np.pi))[top, start:end], color='k', linestyle='--', linewidth=2)
    # plt.plot(range(start, end), (mse_phase/(2*np.pi))[top, start:end], color=pcolor['blue'], linestyle='-', linewidth=2)
    # plt.plot(range(start, end), (wmse_phase/(2*np.pi))[top, start:end], color=pcolor['red'], linestyle='-', linewidth=2)
    # plt.legend(['original phase', 'retrieved with MSE loss', 'retrieved with WMSE loss'], fontsize=20, loc='upper left')
    # plt.ylim([0,0.5])
    # plt.xlim([start, end])
    # plt.yticks([0,0.25,0.5], fontsize=20)
    # plt.xticks([start, start+(end-start)/2, end], fontsize=20)
    # plt.show()
    #
    # # 图3.3
    # sample_name = 'house'
    # path = './实验/wmse/权重影响/'+sample_name
    # original_phase = np.load(path+'/1/phase_obj.npy')
    # mse_phase = np.load(path+'/1/retrieved_phase.npy')
    # wmse_phase = np.load(path+'/100/retrieved_phase.npy')
    # plt.figure()
    # plt.imshow(original_phase/(2*np.pi), cmap='gray', vmin=0, vmax=0.5)
    # plt.plot(range(75, 150), [145]*75, color=pcolor['red'], linestyle='--', linewidth=5)
    # plt.colorbar(ticks=[0, 0.5])
    # plt.axis('off')
    # plt.figure()
    # plt.imshow(mse_phase/(2*np.pi), cmap='gray', vmin=0, vmax=0.5)
    # plt.plot(range(75, 150), [145]*75, color=pcolor['red'], linestyle='--', linewidth=5)
    # plt.colorbar(ticks=[0, 0.5])
    # plt.axis('off')
    # plt.figure()
    # plt.imshow(wmse_phase/(2*np.pi), cmap='gray', vmin=0, vmax=0.5)
    # plt.plot(range(75, 150), [145]*75, color=pcolor['red'], linestyle='--', linewidth=5)
    # plt.colorbar(ticks=[0, 0.5])
    # plt.axis('off')
    # print('rmse of MSE loss: %f' % calc_rmse(original_phase, mse_phase))
    # print('rmse of WMSE loss: %f' % calc_rmse(original_phase, wmse_phase))
    # # plt.show()
    #
    # # 图3.4 三者的截面图
    # plt.figure()
    # plt.plot(range(75, 150), (original_phase/(2*np.pi))[145, 75:150], color='k', linestyle='--', linewidth=2)
    # plt.plot(range(75, 150), (mse_phase/(2*np.pi))[145, 75:150], color=pcolor['blue'], linestyle='-', linewidth=2)
    # plt.plot(range(75, 150), (wmse_phase/(2*np.pi))[145, 75:150], color=pcolor['red'], linestyle='-', linewidth=2)
    # plt.legend(['original phase', 'retrieved with MSE loss', 'retrieved with WMSE loss'], fontsize=20, loc='upper left')
    # plt.ylim([0,0.5])
    # plt.xlim([75,150])
    # plt.yticks([0,0.25,0.5])
    # plt.xticks([75, 112, 150], fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.show()

    # # 图4.1，恢复的amplitude，截面图，截面误差百分数
    # sample_name = 'cameraman'
    # path = './实验/wmse/权重影响/'+sample_name
    # original_magnitudes = np.load(path+'/1/magnitudes.npy')
    # mse_magnitudes = np.load(path+'/1/check_magnitudes.npy')
    # wmse_magnitudes = np.load(path+'/100/check_magnitudes.npy')
    # plt.figure()
    # plt.imshow(np.log(np.fft.fftshift(original_magnitudes**2)), cmap='gray', vmin=0, vmax=22)
    # plt.plot(range(256), [128]*256, color=pcolor['red'], linestyle='--', linewidth=5)
    # plt.colorbar(ticks=[0, 22])
    # plt.axis('off')
    # # plt.figure()
    # # plt.imshow(np.log(np.fft.fftshift(mse_magnitudes**2)), cmap='gray', vmin=0, vmax=22)
    # # plt.plot(range(256), [128]*256, color='r', linestyle='--', linewidth=5)
    # # plt.colorbar(ticks=[0, 22])
    # # plt.axis('off')
    # # plt.figure()
    # # plt.imshow(np.log(np.fft.fftshift(wmse_magnitudes**2)), cmap='gray', vmin=0, vmax=22)
    # # plt.plot(range(256), [128]*256, color='r', linestyle='--', linewidth=5)
    # # plt.colorbar(ticks=[0, 22])
    # # plt.axis('off')
    # # 绝对值截面图
    # # plt.figure()
    # # plt.plot(np.log(np.fft.fftshift(original_magnitudes**2))[128, :], color='k', linestyle='--', linewidth=2)
    # # plt.plot(np.log(np.fft.fftshift(mse_magnitudes**2))[128, :], color=pcolor['blue'], linestyle='-', linewidth=2)
    # # plt.plot(np.log(np.fft.fftshift(wmse_magnitudes**2))[128, :], color=pcolor['red'], linestyle='-', linewidth=2)
    # # plt.legend(['original phase', 'retrieved with MSE loss', 'retrieved with WMSE loss'], fontsize=20)
    # # plt.ylim([0,22])
    # # plt.xticks(fontsize=20)
    # # plt.yticks(fontsize=20)
    # # 相对误差百分数截面图
    # plt.figure()
    # plt.plot(np.arange(-0.5, 0.5, 1/256), cal_relative_error(np.fft.fftshift(original_magnitudes**2), np.fft.fftshift(mse_magnitudes**2))[128, :], color=pcolor['blue'], linestyle='-', linewidth=3)
    # plt.plot(np.arange(-0.5, 0.5, 1/256), cal_relative_error(np.fft.fftshift(original_magnitudes**2), np.fft.fftshift(wmse_magnitudes**2))[128, :], color=pcolor['red'], linestyle='-', linewidth=3)
    # MSE_relative_error = cal_relative_error(np.fft.fftshift(original_magnitudes**2), np.fft.fftshift(mse_magnitudes**2))[128, :]
    # WMSE_relative_error = cal_relative_error(np.fft.fftshift(original_magnitudes**2), np.fft.fftshift(wmse_magnitudes**2))[128, :]
    # print(sample_name)
    # print('MSE results, error of each part (divided by central window of 20 pixel):')
    # print(np.mean(MSE_relative_error[0:118]), np.mean(MSE_relative_error[118:138]), np.mean(MSE_relative_error[138:]))
    # print('WMSE results, error of each part (divided by central window of 20 pixel):')
    # print(np.mean(WMSE_relative_error[0:118]), np.mean(WMSE_relative_error[118:138]), np.mean(WMSE_relative_error[138:]))
    # # plt.scatter(range(256), cal_relative_error(np.fft.fftshift(original_magnitudes**2), np.fft.fftshift(mse_magnitudes**2))[128, :], s=20, color='', edgecolors='b', marker='o', linewidths=2)
    # # plt.scatter(range(256), cal_relative_error(np.fft.fftshift(original_magnitudes**2), np.fft.fftshift(wmse_magnitudes**2))[128, :], s=20, color='', edgecolors='r', marker='o', linewidths=2)
    # plt.legend(['retrieved with MSE loss', 'retrieved with WMSE loss'], fontsize=20)
    # plt.ylim([0,2])
    # plt.xlim([-0.5, 0.5])
    # plt.xticks([-0.5, 0, 0.5], fontsize=20)
    # plt.yticks([0, 2], fontsize=20)
    # # plt.show()
    #
    # # 图4.2，恢复的amplitude，截面图，截面误差百分数
    # sample_name = 'house'
    # path = './实验/wmse/权重影响/'+sample_name
    # original_magnitudes = np.load(path+'/1/magnitudes.npy')
    # mse_magnitudes = np.load(path+'/1/check_magnitudes.npy')
    # wmse_magnitudes = np.load(path+'/100/check_magnitudes.npy')
    # plt.figure()
    # plt.imshow(np.log(np.fft.fftshift(original_magnitudes**2)), cmap='gray', vmin=0, vmax=22)
    # plt.plot(range(256), [128]*256, color=pcolor['red'], linestyle='--', linewidth=5)
    # plt.colorbar(ticks=[0, 22])
    # plt.axis('off')
    # # plt.figure()
    # # plt.imshow(np.log(np.fft.fftshift(mse_magnitudes**2)), cmap='gray', vmin=0, vmax=22)
    # # plt.plot(range(256), [128]*256, color='r', linestyle='--', linewidth=5)
    # # plt.colorbar(ticks=[0, 22])
    # # plt.axis('off')
    # # plt.figure()
    # # plt.imshow(np.log(np.fft.fftshift(wmse_magnitudes**2)), cmap='gray', vmin=0, vmax=22)
    # # plt.plot(range(256), [128]*256, color='r', linestyle='--', linewidth=5)
    # # plt.colorbar(ticks=[0, 22])
    # # plt.axis('off')
    # # 绝对值截面图
    # # plt.figure()
    # # plt.plot(np.log(np.fft.fftshift(original_magnitudes**2))[128, :], color='k', linestyle='--', linewidth=1)
    # # plt.plot(np.log(np.fft.fftshift(mse_magnitudes**2))[128, :], color=pcolor['blue'], linestyle='-', linewidth=1)
    # # plt.plot(np.log(np.fft.fftshift(wmse_magnitudes**2))[128, :], color=pcolor['red'], linestyle='-', linewidth=1)
    # # plt.legend(['original phase', 'retrieved with MSE loss', 'retrieved with WMSE loss'], fontsize=16)
    # # plt.ylim([0,22])
    # # plt.xticks(fontsize=16)
    # # plt.yticks(fontsize=16)
    # # 相对误差百分数截面图
    # plt.figure()
    # plt.plot(np.arange(-0.5, 0.5, 1/256), cal_relative_error(np.fft.fftshift(original_magnitudes**2), np.fft.fftshift(mse_magnitudes**2))[128, :], color=pcolor['blue'], linestyle='-', linewidth=3)
    # plt.plot(np.arange(-0.5, 0.5, 1/256), cal_relative_error(np.fft.fftshift(original_magnitudes**2), np.fft.fftshift(wmse_magnitudes**2))[128, :], color=pcolor['red'], linestyle='-', linewidth=3)
    # MSE_relative_error = cal_relative_error(np.fft.fftshift(original_magnitudes**2), np.fft.fftshift(mse_magnitudes**2))[128, :]
    # WMSE_relative_error = cal_relative_error(np.fft.fftshift(original_magnitudes**2), np.fft.fftshift(wmse_magnitudes**2))[128, :]
    # print(sample_name)
    # print('MSE results, error of each part (divided by central window of 20 pixel):')
    # print(np.mean(MSE_relative_error[0:118]), np.mean(MSE_relative_error[118:138]), np.mean(MSE_relative_error[138:]))
    # print('WMSE results, error of each part (divided by central window of 20 pixel):')
    # print(np.mean(WMSE_relative_error[0:118]), np.mean(WMSE_relative_error[118:138]), np.mean(WMSE_relative_error[138:]))
    # # plt.scatter(range(256), cal_relative_error(np.fft.fftshift(original_magnitudes**2), np.fft.fftshift(mse_magnitudes**2))[128, :], s=20, color='', edgecolors='b', marker='o', linewidths=1)
    # # plt.scatter(range(256), cal_relative_error(np.fft.fftshift(original_magnitudes**2), np.fft.fftshift(wmse_magnitudes**2))[128, :], s=20, color='', edgecolors='r', marker='o', linewidths=1)
    # plt.legend(['retrieved with MSE loss', 'retrieved with WMSE loss'], fontsize=20)
    # plt.ylim([0,1])
    # plt.xlim([-0.5, 0.5])
    # plt.xticks([-0.5, 0, 0.5], fontsize=20)
    # plt.yticks([0, 1], fontsize=20)
    # plt.show()

    # 图5 不同a''下的恢复mse结果
    # name_list = ['cameraman', 'house', 'lena', 'pepper']
    # root = './实验/wmse/权重影响/'
    # for name in name_list:
    #     path = root+name
    #     original = np.load(path+'/1/phase_obj.npy')
    #     result_1 = np.load(path+'/1/retrieved_phase.npy')
    #     result_10 = np.load(path+'/10/retrieved_phase.npy')
    #     result_100 = np.load(path+'/100/retrieved_phase.npy')
    #     result_1000 = np.load(path+'/1000/retrieved_phase.npy')
    #     print(name)
    #     print(calc_rmse(original, result_1),
    #           calc_rmse(original, result_10),
    #           calc_rmse(original, result_100),
    #           calc_rmse(original, result_1000))

    # 图6 网络cardinality的影响
    # path = 'D:/11projects/Deep-phase-decoder/asm_diff/实验/关于网络结构/mandril/wmse100/'
    # list = [1,2,4,8,16]
    # image = cv2.imread('mandril_gray.tif', cv2.IMREAD_GRAYSCALE)
    # image = cv2.resize(image, (256,256))
    # phase_obj = image / 255 * 2 * np.pi *0.5 #* 0.5
    # phase_obj -= np.min(phase_obj)
    # for num, id in enumerate(list):
    #     plt.figure(num)
    #     retrieved_phase = np.load(path+str(id)+'/retrieved_phase.npy')
    #     retrieved_phase -= np.min(retrieved_phase)
    #     print('max min of retrieved phase: ', np.max(retrieved_phase), np.min(retrieved_phase))
    #     plt.imshow(retrieved_phase/(2*np.pi), cmap='gray', vmin=0, vmax=0.5)
    #     # plt.colorbar(ticks=[np.min(phase_obj), np.max(phase_obj)])
    #     plt.colorbar(ticks=[0, 0.5])   # 图上注明单位2Π rad
    #     plt.axis('off')
    #     print('RMSE of d = %d:' % id, calc_rmse(retrieved_phase, phase_obj))
    # plt.show()


    # mse loss 对比
    # for i, exp in enumerate(experiments):
    #     plt.subplot(2,2,i+1)
    #     mse_1 = np.load(path+exp+'/two-modulate/mse_loss_2.npy')
    #     mse_2 = np.load(path+exp+'/two-modulate-cosLR-tmax200/mse_loss_2.npy')
    #     plt.plot(mse_1, color='b')
    #     plt.plot(mse_2, color='r')
    #     plt.title(exp)
    # plt.show()

    # # wmse实验 intensity不同强度处的误差对比
    # ind = 0
    # for i, exp in enumerate(experiments):
    #     plt.subplot(2,2,i+1)
    #     i_o = np.load(path+exp+'/two-modulate/magnitudes.npy')
    #     i_1 = np.load(path+exp+'/two-modulate/check_magnitudes.npy')
    #     i_2 = np.load(path+exp+'/two-modulate-wmse/check_magnitudes.npy')
    #     plt.plot((i_1[ind,:]-i_o[ind,:])/i_o[ind,:], color='b')
    #     plt.plot((i_2[ind,:]-i_o[ind,:])/i_o[ind,:], color='r')
    #     plt.legend(['retrieved with mse loss', 'retrieved with weighted mse loss'])
    #     plt.ylim((-1,5))
    #     plt.title(exp)
    # plt.suptitle('the percentage error of magnitude at axis x=%d' % ind)
    # plt.show()
    #
    # # magnitude绝对值
    # for i, exp in enumerate(experiments):
    #     plt.subplot(2,2,i+1)
    #     i_o = np.load(path+exp+'/two-modulate/magnitudes.npy')
    #     i_1 = np.load(path+exp+'/two-modulate/check_magnitudes.npy')
    #     i_2 = np.load(path+exp+'/two-modulate-wmse/check_magnitudes.npy')
    #     plt.plot(i_o[0,:], color='b')
    #     plt.plot(i_1[0,:], color='g')
    #     plt.plot(i_2[0,:], color='r')
    #     plt.legend(['original', 'retrieved with mse loss', 'retrieved with weighted mse loss'])
    #     # plt.ylim((-1,3))
    #     plt.title(exp)
    # plt.suptitle('the value of magnitude at axis x=0')
    # plt.show()
    #
    # for i, exp in enumerate(experiments):
    #     plt.subplot(2,2,i+1)
    #     i_o = np.load(path+exp+'/two-modulate/magnitudes.npy')
    #     i_1 = np.load(path+exp+'/two-modulate/check_magnitudes.npy')
    #     i_2 = np.load(path+exp+'/two-modulate-wmse/check_magnitudes.npy')
    #     plt.imshow(np.log((i_1-i_o)/i_o+1))
    #     # plt.imshow(np.log((i_2-i_o)/i_o+1))
    #     plt.title(exp)
    #     plt.colorbar()
    # plt.show()

    # 

main()
