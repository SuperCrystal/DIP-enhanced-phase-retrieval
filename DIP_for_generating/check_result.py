import matplotlib.pyplot as plt
import numpy as np
import os

def calc_intensity(phase):
    print((np.fft.fft2(np.exp(1j*phase))).dtype)
    magnitudes = np.abs(np.fft.fft2(np.exp(1j*phase)))
    return magnitudes

def rmse(a, b):
    return np.sqrt(np.mean((a-b)**2))

def main():
    # file = './50000-sigmoid-noise-lrdecay/'
    file = './单调制/'
    phase_obj = np.load(file+'phase_obj.npy')
    retrieved_phase = np.load(file+'retrieved_phase.npy')
    magnitudes = np.load(file+'magnitudes.npy')
    check_magnitudes = np.load(file+'check_magnitudes.npy')
    # print(phase_obj.shape, retrieved_phase.shape, magnitudes.shape, check_magnitudes.shape)
    # print(phase_obj.dtype, retrieved_phase.dtype)
    # print(np.max(phase_obj), np.max(retrieved_phase))
    # print(np.min(phase_obj), np.min(retrieved_phase))

    print(rmse(phase_obj, retrieved_phase))
    print(np.max(phase_obj), np.max(retrieved_phase))
    print(np.min(phase_obj), np.min(retrieved_phase))
    print(rmse(magnitudes, check_magnitudes))
    print(rmse(calc_intensity(phase_obj), calc_intensity(retrieved_phase)))
    print(rmse(calc_intensity(phase_obj), calc_intensity(phase_obj+0.5)))
    print(np.min(phase_obj), np.min(phase_obj+0.5))
    # plt.figure()
    # plt.plot(phase_obj[0, :])
    # plt.plot(retrieved_phase[0, :])
    # plt.show()

    # plt.figure()
    # # plt.imshow(calc_intensity(retrieved_phase), cmap='gray')
    # plt.plot(calc_intensity(phase_obj)[0, :])
    # plt.plot(calc_intensity(phase_obj+0.5)[0, :])
    # plt.show()

    mse_loss1 = np.load('./单调制/mse_loss_2.npy')
    mse_loss2 = np.load('./多调制/mse_loss_2.npy')
    plt.figure()
    plt.plot(mse_loss1, color='r')
    plt.plot(mse_loss2, color='b')
    plt.show()

    # plt.figure()
    # # plt.plot(calc_intensity(phase_obj)[0,:])
    # # plt.plot(calc_intensity(retrieved_phase)[0,:])
    # # 计算该轴上不同位置的相对误差大小
    # relate_error = np.abs(calc_intensity(retrieved_phase)[0,:] - calc_intensity(phase_obj)[0,:])/(calc_intensity(phase_obj)[0,:])
    # # print(relate_error)
    # plt.plot(relate_error,color='r')
    # file = './10000-modulate-lrdecay500-wmse100-0.1/'
    # phase_obj = np.load(file+'phase_obj.npy')
    # retrieved_phase = np.load(file+'retrieved_phase.npy')
    # relate_error = np.abs(calc_intensity(retrieved_phase)[0,:] - calc_intensity(phase_obj)[0,:])/(calc_intensity(phase_obj)[0,:])
    # plt.plot(relate_error,color='b')
    # plt.show()
    #
    # plt.subplot(221)
    # plt.imshow(phase_obj, cmap='gray')
    # plt.title('phase_obj')
    # plt.subplot(222)
    # # plt.imshow(retrieved_phase, cmap='gray')
    # plt.imshow(retrieved_phase, cmap='gray')
    # plt.title('retrieved_phase')
    # plt.subplot(223)
    # plt.imshow(calc_intensity(phase_obj), cmap='gray')
    # plt.title('magnitudes')
    # plt.subplot(224)
    # plt.imshow(calc_intensity(retrieved_phase), cmap='gray')
    # plt.title('check_magnitudes')
    # plt.show()

if __name__ == '__main__':
    main()
