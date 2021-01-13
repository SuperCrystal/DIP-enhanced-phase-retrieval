import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import matplotlib
import scipy

pcolor = {'red': (239/255, 131/255, 119/255),
          'blue': (65/255, 113/255, 156/255),
          'green': (173, 208, 92)}

path = 'D:/00 论文相关/毕设/实验/data0105/无样品/'
def raw2npy(dir):
    # path = 'D:/00 论文相关/毕设/实验/data1216/'
    rawfile = np.fromfile(path+dir, dtype='uint16')

    img = np.reshape(rawfile, (2704,3376))
    print('min and max of %s: %d %d' % (dir, np.min(img), np.max(img)))
    return img

def save_as_raw(dir):
    if isinstance(dir, str):
        img = np.load(path+dir)
        img.tofile(path+dir.split('.')[0]+'.raw')
    elif isinstance(dir, np.ndarray):
        pass
    else:
        print('error: invaid input type!')

def display_raw_img(dir):
    # path = 'D:/00 论文相关/毕设/实验/data1216/'
    rawfile = np.fromfile(path+dir, dtype='uint16')
    print(rawfile.shape)

    img = np.reshape(rawfile, (2704,3376))
    plt.figure()
    plt.imshow(img)

def crop_img(img,x,y,size):
    cropped = img[x-size//2+1:x+size//2+1, y-size//2+1:y+size//2+1]
    return cropped

def combine_img(img1, img2, times):
    combine = img1.copy().astype('uint32')
    rp = img2.copy()
    for m in range(img1.shape[0]):
        for n in range(img1.shape[1]):
            if combine[m][n] >= 65520:
                combine[m][n] = rp[m][n] * times
    return combine

def average_denoise(dir, mode):
    if mode == 'raw':
        files = os.listdir(dir)
        avg = np.zeros((2704,3376), dtype='uint32')
        for file in files:
            img = np.fromfile(os.path.join(dir, file), dtype='uint16')
            img = np.reshape(img, (2704,3376))
            avg += img
        avg = avg / len(files)
        np.save(dir+'.npy', avg)
        plt.figure()
        plt.imshow(avg)
        plt.show()
        return avg
    elif mode == 'bmp':
        files = os.listdir(dir)
        avg = np.zeros((2704,3376), dtype='uint32')
        for file in files:
            print(os.path.join(dir, file))
            # img = PIL.Image.open(os.path.join(dir, file))
            tmp_img = cv2.imread(os.path.join(dir, file))
            # img = PIL.Image.fromarray(img)
            # img = scipy.misc.imread(tmp_img)
            print(tmp_img)
            cv2.imshow('imt', tmp_img)
            plt.figure()
            plt.imshow(tmp_img)
            plt.show()
    else:
        print('error')

def process_zygo():
    file = 'D:/00 论文相关/毕设/实验/zygo/20201231/8/3.txt'
    size = 992
    img = np.zeros((size, size))
    with open(file, 'r') as f:
        lines = f.readlines()
        content = lines[14:-1]
        for line in content:
            li = line.split('\n')[0]
            print(li)
            li = li.split(' ')
            x = int(li[0])
            y = int(li[1])
            z = li[2]
            if z == 'No':
                z = 0
            img[x][y] = z
    img = img - np.min(img)
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    plt.show()
    np.save('3_result', img)

# -----
# process_zygo()
# ----- 批量转为raw文件 ------
# path = 'D:/11projects/Deep-phase-decoder/auto_defocus/exp/1230/w_sample/combined/denoised/1024/recovery/'
# files = os.listdir(path)
# for file in files:
#     save_as_raw(file)

# -- 读取每个位置每个曝光下的10幅图像，取平均转换为npy文件
# path = 'D:/00 论文相关/毕设/实验/data0105-raw/有样品/'
# files = os.listdir(path)
# for file in files:
#     dir = os.path.join(path, file)
#     print(dir)
#     average_denoise(dir, 'raw')
# 读取背景光并平均
# path = 'D:/00 论文相关/毕设/实验/data0105-raw/背景光/'
# average_denoise(path, 'raw')
# 先扣除背景光，后续再拼图
# avg_bkg = np.load('D:/00 论文相关/毕设/实验/data0105-raw/背景光/avg_bkg.npy')
# path = path = 'D:/00 论文相关/毕设/实验/data0105-raw/无样品/'
# files = os.listdir(path)
# for file in files:
#     if file.split('.')[-1] != 'npy':
#         continue
#     original = np.load(path+file)
#     new = original - avg_bkg
#     new[new<0] = 0
#     new = new.astype('uint32')
#     np.save(path+'去背景/'+file, new)

# 找center
# name = os.listdir(path)
# for n in name:
#     if n.split('.')[-1] != 'npy':
#         continue
#     img = np.load(path+n)
#     print(n)
#     plt.figure()
#     plt.imshow(img)
#     plt.show()
# 无样品
# path = './no_sample/'
# files = os.listdir(path)
# for file in files:
#     dir = os.path.join(path, file)
#     average_denoise(dir, 'bmp')


# center = [[1281,1362], [1277,1378], [1273,1390], [1270,1402]]

# ----- crop to 256x256 ------- #
# name = os.listdir(path)
# for n in name:
#     if n.split('.')[-1] != 'raw':
#         continue
#     print(n)
#     display_raw_img(n)
#     plt.show()

# -----无样品
# 1221
# center = [[1310,1363], [1314,1381], [1317,1395], [1320,1412]]
# # name = ['f40.raw', 'f1000.raw', 'f2000.raw', 'f3000.raw']
# name = ['f40.raw', 'f1000.raw', 'f2000.raw', 'f3000.raw']
# # name = ['df-10-150.raw', 'df-10-200.raw', 'df-10-300.raw', 'df-10-500.raw', 'df-10-1000.raw', 'df-10-2000.raw']
# 1228
# center = [[1322,1362], [1318,1377], [1314,1389], [1310,1400]]
# name = ['f-40.raw', 'f-100.raw', 'f-400.raw', 'f-600.raw', 'f-1000.raw', 'f-2000.raw']
# name = ['df4-40.raw', 'df4-100.raw', 'df4-200.raw', 'df4-500.raw', 'df4-1000.raw', 'df4-2000.raw']
# name = ['df7-40.raw', 'df7-100.raw', 'df7-200.raw', 'df7-400.raw', 'df7-800.raw', 'df7-1000.raw']
# name = ['df10-40.raw', 'df10-100.raw', 'df10-200.raw', 'df10-500.raw', 'df10-800.raw', 'df10-1000.raw']
# size = 1024
# for n in name:
#     img = raw2npy(n)
#     img = crop_img(img, center[3][1], center[3][0], size)
#     np.savez(path+n.split('.')[0]+'_%s' % size, img)
#     plt.figure()
#     plt.imshow(img)
#     plt.show()
# 0105 
# center = [[2299,1636], [2305,1619], [2309,1605], [2313,1592]]
# # name = ['f-40.npy', 'f-100.npy', 'f-200.npy']
# # name = ['df4-120.npy', 'df4-150.npy', 'df4-200.npy']
# # name = ['df7-150.npy', 'df7-300.npy', 'df7-350.npy']
# name = ['df10-350.npy', 'df10-470.npy', 'df10-600.npy']
# size = 1024
# for n in name:
#     img = np.load(path+n)
#     print(img.dtype)
#     img = crop_img(img, center[3][1], center[3][0], size)
#     np.savez(path+n.split('.')[0]+'_%s' % size, img)
#     plt.figure()
#     plt.imshow(img)
#     plt.show()

# ----- 有样品
size = 1024
# 1221
# center = [[1307,1365], [1312, 1385], [1315, 1402], [1318, 1419]]
# # name = ['f40.raw', 'f100.raw', 'f400.raw', 'f1000.raw', 'f2000.raw']
# # name = ['df-4-40.raw', 'df-4-200.raw', 'df-4-500.raw', 'df-4-1000.raw', 'df-4-2000.raw']
# # name = ['df-7-40.raw', 'df-7-100.raw', 'df-7-200.raw', 'df-7-1000.raw', 'df-7-2000.raw']
# name = ['df-10-100.raw', 'df-10-200.raw', 'df-10-400.raw', 'df-10-500.raw', 'df-10-2000.raw']
# 1228
# center = [[1321,1362], [1318, 1376], [1314, 1388], [1309, 1400]]
# # name = ['f-40.raw', 'f-100.raw', 'f-200.raw', 'f-500.raw', 'f-1000.raw', 'f-2000.raw']
# # name = ['df4-40.raw', 'df4-100.raw', 'df4-200.raw', 'df4-500.raw', 'df4-1000.raw', 'df4-2000.raw']
# # name = ['df7-40.raw', 'df7-100.raw', 'df7-200.raw', 'df7-500.raw', 'df7-1000.raw', 'df7-2000.raw']
# name = ['df10-100.raw', 'df10-200.raw', 'df10-300.raw', 'df10-500.raw', 'df10-1000.raw', 'df10-2000.raw']
# for n in name:
#     img = raw2npy(n)
#     img = crop_img(img, center[3][1], center[3][0], size)
#     np.savez(path+n.split('.')[0]+'_%s' % size, img)
#     plt.figure()
#     plt.imshow(img)
#     plt.show()
# 1230
# center = [[1281,1362], [1277,1378], [1273,1390], [1270,1402]]
# # name = ['f-40.npy', 'f-100.npy', 'f-200.npy', 'f-400.npy']
# # name = ['df4-40.npy', 'df4-100.npy', 'df4-200.npy', 'df4-300.npy']
# # name = ['df7-100.npy', 'df7-200.npy', 'df7-350.npy', 'df7-550.npy']
# name = ['df10-500.npy', 'df10-700.npy', 'df10-1000.npy']
# for n in name:
#     img = np.load(path+n)
#     img = crop_img(img, center[3][1], center[3][0], size)
#     np.savez(path+n.split('.')[0]+'_%s' % size, img)
#     plt.figure()
#     plt.imshow(img)
#     plt.show()
# 0103
# center = [[2356,1618], [2363,1600], [2368,1586], [2372,1573]]
# name = ['f-500.npy', 'f-1000.npy', 'f-1500.npy']
# # name = ['df4-500.npy', 'df4-1000.npy', 'df4-1500.npy']
# # name = ['df7-1000.npy', 'df7-1500.npy', 'df7-2000.npy']
# # name = ['df10-2000.npy', 'df10-2500.npy', 'df10-3000.npy']
# for n in name:
#     img = np.load(path+n)
#     print(img.dtype)
#     img = crop_img(img, center[0][1], center[0][0], size)
#     np.savez(path+n.split('.')[0]+'_%s' % size, img)
#     plt.figure()
#     plt.imshow(img)
    # plt.show()
# 0105
# center = [[2299,1638], [2305,1621], [2309,1606], [2314,1592]]
# # name = ['f-40.npy', 'f-100.npy', 'f-300.npy']
# # name = ['df4-100.npy', 'df4-130.npy', 'df4-150.npy']
# # name = ['df7-110.npy', 'df7-200.npy', 'df7-250.npy', 'df7-450.npy']
# name = ['df10-450.npy', 'df10-700.npy', 'df10-900.npy', 'df10-1200.npy']
# for n in name:
#     img = np.load(path+n)
#     print(img.dtype)
#     img = crop_img(img, center[3][1], center[3][0], size)
#     np.savez(path+n.split('.')[0]+'_%s' % size, img)
#     plt.figure()
#     plt.imshow(img)
#     plt.show()

# -------- 拼接 --------- #
# ------ 无样品 -------- #
# path = 'D:/00 论文相关/毕设/实验/data1221/无样品/'
# size = 1024
# # files = ['f40_%s.npz' % size, 'f1000_%s.npz' % size, 'f2000_%s.npz' % size, 'f3000_%s.npz' % size]
# files = ['df-4-40_%s.npz' % size, 'df-4-1000_%s.npz' % size, 'df-4-2000_%s.npz' % size, 'df-4-3000_%s.npz' % size]
# # files = ['df-7-40_%s.npz' % size, 'df-7-200_%s.npz' % size, 'df-7-500_%s.npz' % size, 'df-7-1000_%s.npz' % size]
# # files = ['df-10-150_%s.npz' % size, 'df-10-200_%s.npz' % size, 'df-10-300_%s.npz' % size, 'df-10-500_%s.npz' % size, 'df-10-1000_%s.npz' % size]
# f1 = np.load(path+files[0])['arr_0']
# f2 = np.load(path+files[1])['arr_0']
# f3 = np.load(path+files[2])['arr_0']
# f4 = np.load(path+files[3])['arr_0']
# # f5 = np.load(path+files[4])['arr_0']
#
# plt.figure()
# plt.plot(range(size), f1[:, size//2])
# plt.plot(range(size), f2[:, size//2])
# plt.plot(range(size), f3[:, size//2])
# plt.plot(range(size), f4[:, size//2])
# # plt.plot(range(size), f5[:, size//2])
#
# plt.figure()
# plt.imshow(f1)
# plt.figure()
# plt.imshow(f2)
# plt.figure()
# plt.imshow(f3)
# plt.figure()
# plt.imshow(f4)
# plt.show()
# plt.figure()
# # plt.imshow(f5)
# # plt.show()
#
# # x = [124,115,118,125,130,134,132,129]
# # y = [116,125,134,139,138,119,116,116]
# x = [128,111,121,134,142,143,140,133]
# y = [109,119,110,110,118,131,138,143]
# # x = [128,111,121,134,142,143,140,133]
# # y = [109,119,110,110,118,131,138,143]
# print(f2[x[0], y[0]] / f1[x[0], y[0]])
# print(f2[x[1], y[1]] / f1[x[1], y[1]])
# print(f2[x[2], y[2]] / f1[x[2], y[2]])
# print(f2[x[3], y[3]] / f1[x[3], y[3]])
# print(f2[x[4], y[4]] / f1[x[4], y[4]])
# print(f2[x[5], y[5]] / f1[x[5], y[5]])
# combine = combine_img(f2, f1, 17.6) #17.6
# plt.figure()
# plt.imshow(combine)
# plt.figure()
# plt.plot(range(size), f1[:, size//2])
# plt.plot(range(size), f2[:, size//2])
# plt.plot(range(size), f3[:, size//2])
# plt.plot(range(size), f4[:, size//2])
# plt.plot(range(size), combine[:, size//2])
# plt.show()
# np.save(path+'combine_%s/df4' % size, f1)

# - 无样品 data1228
# path = 'D:/00 论文相关/毕设/实验/data1228/无样品/'
# size = 1024
# # files = ['f-40_%s.npz' % size, 'f-100_%s.npz' % size, 'f-400_%s.npz' % size, 'f-600_%s.npz' % size, 'f-1000_%s.npz' % size, 'f-2000_%s.npz' % size]
# files = ['df4-40_%s.npz' % size, 'df4-100_%s.npz' % size, 'df4-200_%s.npz' % size, 'df4-500_%s.npz' % size, 'df4-1000_%s.npz' % size, 'df4-2000_%s.npz' % size]
# # # files = ['df7-40_%s.npz' % size, 'df7-100_%s.npz' % size, 'df7-200_%s.npz' % size, 'df7-400_%s.npz' % size, 'df7-800_%s.npz' % size, 'df7-1000_%s.npz' % size]
# # files = ['df10-40_%s.npz' % size, 'df10-100_%s.npz' % size, 'df10-200_%s.npz' % size, 'df10-500_%s.npz' % size, 'df10-800_%s.npz' % size, 'df10-1000_%s.npz' % size]
# f1 = np.load(path+files[0])['arr_0']
# f2 = np.load(path+files[1])['arr_0']
# f3 = np.load(path+files[2])['arr_0']
# f4 = np.load(path+files[3])['arr_0']
# f5 = np.load(path+files[4])['arr_0']
# #
# # plt.figure()
# # plt.plot(range(size), f1[:, size//2])
# # plt.plot(range(size), f2[:, size//2])
# # plt.plot(range(size), f3[:, size//2])
# # plt.plot(range(size), f4[:, size//2])
# # plt.plot(range(size), f5[:, size//2])
# #
# plt.figure()
# plt.imshow(f1)
# plt.figure()
# plt.imshow(f2)
# plt.figure()
# plt.imshow(f3)
# plt.figure()
# plt.imshow(f4)
# plt.show()
# plt.figure()
# plt.imshow(f5)
# plt.show()
#
# # x = [124,115,118,125,130,134,132,129] # f f3/f1  8.5
# # y = [116,125,134,139,138,119,116,116]
# x = [128,111,121,134,142,143,140,133]  # df4 f2
# y = [109,119,110,110,118,131,138,143]
# # x = [128,111,121,134,142,143,140,133]  # df7 f4
# # y = [109,119,110,110,118,131,138,143]  # df10 f5
# print(f2[x[0], y[0]] / f1[x[0], y[0]])
# print(f2[x[1], y[1]] / f1[x[1], y[1]])
# print(f2[x[2], y[2]] / f1[x[2], y[2]])
# print(f2[x[3], y[3]] / f1[x[3], y[3]])
# print(f2[x[4], y[4]] / f1[x[4], y[4]])
# print(f2[x[5], y[5]] / f1[x[5], y[5]])
# combine = combine_img(f3, f1, 8.5)
# plt.figure()
# plt.imshow(combine)
# plt.figure()
# plt.plot(range(size), f1[:, size//2])
# plt.plot(range(size), f2[:, size//2])
# plt.plot(range(size), f3[:, size//2])
# plt.plot(range(size), f4[:, size//2])
# plt.plot(range(size), combine[:, size//2])
# plt.show()
# np.save(path+'combine_%s/f' % size, combine)

# - 无样品 data0105
path = 'D:/00 论文相关/毕设/实验/data0105/无样品/'
size = 1024
# files = ['f-40_%s.npz' % size, 'f-100_%s.npz' % size, 'f-200_%s.npz' % size]
# files = ['df4-120_%s.npz' % size, 'df4-150_%s.npz' % size, 'df4-200_%s.npz' % size]
# files = ['df7-150_%s.npz' % size, 'df7-300_%s.npz' % size, 'df7-350_%s.npz' % size]
files = ['df10-350_%s.npz' % size, 'df10-470_%s.npz' % size, 'df10-600_%s.npz' % size]
f1 = np.load(path+files[0])['arr_0']
f2 = np.load(path+files[1])['arr_0']
f3 = np.load(path+files[2])['arr_0']
# f4 = np.load(path+files[3])['arr_0']
# f5 = np.load(path+files[4])['arr_0']
#
# plt.figure()
# plt.plot(range(size), f1[:, size//2])
# plt.plot(range(size), f2[:, size//2])
# plt.plot(range(size), f3[:, size//2])
# plt.plot(range(size), f4[:, size//2])
# plt.plot(range(size), f5[:, size//2])
#
plt.figure()
plt.imshow(f1)
plt.figure()
plt.imshow(f2)
plt.figure()
plt.imshow(f3)
# plt.figure()
# plt.imshow(f4)
# plt.show()
# plt.figure()
# plt.imshow(f5)
# plt.show()

# x = [124,115,118,125,130,134,132,129] # f f1
# y = [116,125,134,139,138,119,116,116]
x = [128,111,121,134,142,143,140,133]  # df4 f1
y = [109,119,110,110,118,131,138,143]
# x = [128,111,121,134,142,143,140,133]  # df7 f2
# y = [109,119,110,110,118,131,138,143]  # df10 f2
print(f2[x[0], y[0]] / f1[x[0], y[0]])
print(f2[x[1], y[1]] / f1[x[1], y[1]])
print(f2[x[2], y[2]] / f1[x[2], y[2]])
print(f2[x[3], y[3]] / f1[x[3], y[3]])
print(f2[x[4], y[4]] / f1[x[4], y[4]])
print(f2[x[5], y[5]] / f1[x[5], y[5]])
combine = combine_img(f3, f1, 8.5)
plt.figure()
plt.imshow(combine)
plt.figure()
plt.plot(range(size), f1[:, size//2])
plt.plot(range(size), f2[:, size//2])
plt.plot(range(size), f3[:, size//2])
# plt.plot(range(size), f4[:, size//2])
plt.plot(range(size), combine[:, size//2])
plt.show()
np.save(path+'combine_%s/df10' % size, f2)

# ------ 有样品 -------- #
# size = 256
# files = ['f40_%s.npz' % size, 'f100_%s.npz' % size, 'f400_%s.npz' % size, 'f1000_%s.npz' % size, 'f2000_%s.npz' % size]
# # files = ['df-4-40_%s.npz' % size, 'df-4-200_%s.npz' % size, 'df-4-500_%s.npz' % size, 'df-4-1000_%s.npz' % size, 'df-4-2000_%s.npz' % size]
# # files = ['df-7-40_%s.npz' % size, 'df-7-100_%s.npz' % size, 'df-7-200_%s.npz' % size, 'df-7-1000_%s.npz' % size, 'df-7-2000_%s.npz' % size]
# # files = ['df-10-100_%s.npz' % size, 'df-10-200_%s.npz' % size, 'df-10-400_%s.npz' % size, 'df-10-500_%s.npz' % size, 'df-10-2000_%s.npz' % size]
# f1 = np.load(path+files[0])['arr_0']
# f2 = np.load(path+files[1])['arr_0']
# f3 = np.load(path+files[2])['arr_0']
# f4 = np.load(path+files[3])['arr_0']
# f5 = np.load(path+files[4])['arr_0']
#
# plt.figure()
# plt.plot(range(size), f1[:, size//2])
# plt.plot(range(size), f2[:, size//2])
# plt.plot(range(size), f3[:, size//2])
# plt.plot(range(size), f4[:, size//2])
# plt.plot(range(size), f5[:, size//2])
#
# plt.figure()
# plt.imshow(f1)
# plt.figure()
# plt.imshow(f2)
# plt.figure()
# plt.imshow(f3)
# plt.figure()
# plt.imshow(f4)
# plt.figure()
# plt.imshow(f5)
# plt.show()
#
# x = [114,114,114,115,125,124,127,130,124,140] # f f3 f1 8.76
# y = [124,125,126,125,117,116,117,116,138,128]
# # x = [123,124,127,128,128,125,120,131]  # df4 f3 f1 6.45
# # y = [118,119,118,118,135,135,134,132]
# # x = [128,111,121,134,142,143,140,133]
# # y = [109,119,110,110,118,131,138,143]
# for id in range(len(x)):
#     print(f3[x[id], y[id]] / f1[x[id], y[id]])
# combine = combine_img(f3, f1, 8.76) #17.6
# plt.figure()
# plt.imshow(combine)
# plt.figure()
# plt.plot(range(size), f1[:, size//2])
# plt.plot(range(size), f2[:, size//2])
# plt.plot(range(size), f3[:, size//2])
# plt.plot(range(size), f4[:, size//2])
# plt.plot(range(size), combine[:, size//2])
# plt.show()
# np.save(path+'combine_%s/f' % size, combine)

# ------------ 有样品512 --------------
# size = 1024
# files = ['f40_%s.npz' % size, 'f100_%s.npz' % size, 'f400_%s.npz' % size, 'f1000_%s.npz' % size, 'f2000_%s.npz' % size]
# # files = ['df-4-40_%s.npz' % size, 'df-4-200_%s.npz' % size, 'df-4-500_%s.npz' % size, 'df-4-1000_%s.npz' % size, 'df-4-2000_%s.npz' % size]
# # files = ['df-7-40_%s.npz' % size, 'df-7-100_%s.npz' % size, 'df-7-200_%s.npz' % size, 'df-7-1000_%s.npz' % size, 'df-7-2000_%s.npz' % size]
# # files = ['df-10-100_%s.npz' % size, 'df-10-200_%s.npz' % size, 'df-10-400_%s.npz' % size, 'df-10-500_%s.npz' % size, 'df-10-2000_%s.npz' % size]
# f1 = np.load(path+files[0])['arr_0']
# f2 = np.load(path+files[1])['arr_0']
# f3 = np.load(path+files[2])['arr_0']
# f4 = np.load(path+files[3])['arr_0']
# f5 = np.load(path+files[4])['arr_0']
#
# plt.figure()
# plt.plot(range(size), f1[:, size//2])
# plt.plot(range(size), f2[:, size//2])
# plt.plot(range(size), f3[:, size//2])
# plt.plot(range(size), f4[:, size//2])
# plt.plot(range(size), f5[:, size//2])
#
# # plt.figure()
# # plt.imshow(f1)
# # plt.figure()
# # plt.imshow(f2)
# # plt.figure()
# # plt.imshow(f3)
# # plt.figure()
# # plt.imshow(f4)
# # plt.figure()
# # plt.imshow(f5)
# # plt.show()
#
# x = [114,114,114,115,125,124,127,130,124,140] # f f3 f1 8.76
# y = [124,125,126,125,117,116,117,116,138,128]
# # x = [123,124,127,128,128,125,120,131]  # df4 f3 f1 7.6?
# # y = [118,119,118,118,135,135,134,132]
# # x = [128,111,121,134,142,143,140,133]
# # y = [109,119,110,110,118,131,138,143]
# for id in range(len(x)):
#     print(f3[x[id], y[id]] / f1[x[id], y[id]])
# combine = combine_img(f3, f1, 8.76) #17.6
# plt.figure()
# plt.imshow(combine)
# plt.figure()
# plt.plot(range(size), f1[:, size//2])
# plt.plot(range(size), f2[:, size//2])
# plt.plot(range(size), f3[:, size//2])
# plt.plot(range(size), f4[:, size//2])
# plt.plot(range(size), combine[:, size//2])
# plt.show()
# np.save(path+'combine_%s/f' % size, combine)
# ---------------------------------------- #
# ------------ 有样品1024 by 1228 --------------
# size = 1024
# # files = ['f-40_%s.npz' % size, 'f-100_%s.npz' % size, 'f-200_%s.npz' % size, 'f-500_%s.npz' % size, 'f-1000_%s.npz' % size, 'f-2000_%s.npz' % size]
# # files = ['df4-40_%s.npz' % size, 'df4-100_%s.npz' % size, 'df4-200_%s.npz' % size, 'df4-500_%s.npz' % size, 'df4-1000_%s.npz' % size, 'df4-2000_%s.npz' % size]
# # files = ['df7-40_%s.npz' % size, 'df7-100_%s.npz' % size, 'df7-200_%s.npz' % size, 'df7-500_%s.npz' % size, 'df7-1000_%s.npz' % size, 'df7-2000_%s.npz' % size]
# files = ['df10-100_%s.npz' % size, 'df10-200_%s.npz' % size, 'df10-300_%s.npz' % size, 'df10-500_%s.npz' % size, 'df10-1000_%s.npz' % size, 'df10-2000_%s.npz' % size]
# f1 = np.load(path+files[0])['arr_0']
# f2 = np.load(path+files[1])['arr_0']
# f3 = np.load(path+files[2])['arr_0']
# f4 = np.load(path+files[3])['arr_0']
# f5 = np.load(path+files[4])['arr_0']
#
# plt.figure()
# plt.plot(range(size), f1[:, size//2])
# plt.plot(range(size), f2[:, size//2])
# plt.plot(range(size), f3[:, size//2])
# plt.plot(range(size), f4[:, size//2])
# plt.plot(range(size), f5[:, size//2])
#
# plt.figure()
# plt.imshow(f1)
# plt.figure()
# plt.imshow(f2)
# plt.figure()
# plt.imshow(f3)
# plt.figure()
# plt.imshow(f4)
# plt.figure()
# plt.imshow(f5)
# plt.show()
#
# x = [114,114,114,115,125,124,127,130,124,140] # f f3 f1 4.18
# y = [124,125,126,125,117,116,117,116,138,128]
# # x = [123,124,127,128,128,125,120,131]  # df4 f4 f2 4.68
# # y = [118,119,118,118,135,135,134,132]
# # x = [128,111,121,134,142,143,140,133] # df7 f4
# # y = [109,119,110,110,118,131,138,143]
# # x = [128,111,121,134,142,143,140,133] # df10 f4
# # y = [109,119,110,110,118,131,138,143]
# for id in range(len(x)):
#     print(f3[x[id], y[id]] / f1[x[id], y[id]])
# combine = combine_img(f4, f2, 4.68) #17.6
# plt.figure()
# plt.imshow(combine)
# plt.figure()
# plt.plot(range(size), f1[:, size//2])
# plt.plot(range(size), f2[:, size//2])
# plt.plot(range(size), f3[:, size//2])
# plt.plot(range(size), f4[:, size//2])
# plt.plot(range(size), combine[:, size//2])
# plt.show()
# np.save(path+'combine_%s/df10_2' % size, f3)
# ---------------------------------------------
# ------------ 有样品1024 by 1230 --------------
# path = 'D:/00 论文相关/毕设/实验/data1230/有样品/'
# size = 1024
# # files = ['f-40_%s.npz' % size, 'f-100_%s.npz' % size, 'f-200_%s.npz' % size, 'f-400_%s.npz' % size]
# files = ['df4-40_%s.npz' % size, 'df4-100_%s.npz' % size, 'df4-200_%s.npz' % size, 'df4-300_%s.npz' % size]
# # files = ['df7-100_%s.npz' % size, 'df7-200_%s.npz' % size, 'df7-350_%s.npz' % size, 'df7-550_%s.npz' % size]
# # files = ['df10-500_%s.npz' % size, 'df10-700_%s.npz' % size, 'df10-1000_%s.npz' % size]
# f1 = np.load(path+files[0])['arr_0']
# f2 = np.load(path+files[1])['arr_0']
# f3 = np.load(path+files[2])['arr_0']
# f4 = np.load(path+files[3])['arr_0']
# # f5 = np.load(path+files[4])['arr_0']
#
# plt.figure()
# plt.plot(range(size), f1[:, size//2])
# plt.plot(range(size), f2[:, size//2])
# plt.plot(range(size), f3[:, size//2])
# plt.plot(range(size), f4[:, size//2])
# # plt.plot(range(size), f5[:, size//2])
#
# # plt.figure()
# # plt.imshow(f1)
# plt.figure()
# plt.imshow((f2)[400:-400, 400:-400]/132982, cmap='jet', vmax=1)
# plt.axis('off')
# # plt.figure()
# # plt.imshow(f3)
# plt.figure()
# plt.imshow(f4[400:-400, 400:-400]/132982, cmap='jet', vmax=1)
# plt.axis('off')
# # plt.figure()
# # plt.imshow(f5)
# # plt.show()
#
# x = [114,114,114,115,125,124,127,130,124,140] # f f1 ; f3/f1 3.8
# y = [124,125,126,125,117,116,117,116,138,128]
# # x = [123,124,127,128,128,125,120,131]  # df4 f2 ; f4/f2 2.45
# # y = [118,119,118,118,135,135,134,132]
# # x = [128,111,121,134,142,143,140,133] # df7 f3 ; f4
# # y = [109,119,110,110,118,131,138,143]
# # x = [128,111,121,134,142,143,140,133] # df10 f2
# # y = [109,119,110,110,118,131,138,143]
# for id in range(len(x)):
#     print(f3[x[id], y[id]] / f1[x[id], y[id]])
# combine = combine_img(f4, f2, 2.45) #17.6
# print('max of combined:' , np.max(combine))
# plt.figure()
# plt.imshow(combine[400:-400, 400:-400]/132982, cmap='jet', vmax=1)
# plt.colorbar()
# plt.axis('off')
# # plt.figure()
# # plt.plot(range(size), f1[:, size//2])
# # plt.plot(range(size), f2[:, size//2])
# # plt.plot(range(size), f3[:, size//2])
# # plt.plot(range(size), f4[:, size//2])
# # plt.plot(range(size), combine[:, size//2])
# plt.show()
# np.save(path+'combine_%s/df10_2' % size, f3)
# ------------ 有样品1024 by 0103 --------------
# size = 1024
# # files = ['f-500_%s.npz' % size, 'f-1000_%s.npz' % size, 'f-1500_%s.npz' % size]
# # files = ['df4-500_%s.npz' % size, 'df4-1000_%s.npz' % size, 'df4-1500_%s.npz' % size]
# # files = ['df7-1000_%s.npz' % size, 'df7-1500_%s.npz' % size, 'df7-2000_%s.npz' % size]
# files = ['df10-2000_%s.npz' % size, 'df10-2500_%s.npz' % size, 'df10-3000_%s.npz' % size]
# f1 = np.load(path+files[0])['arr_0']
# f2 = np.load(path+files[1])['arr_0']
# f3 = np.load(path+files[2])['arr_0']
# # f4 = np.load(path+files[3])['arr_0']
# # f5 = np.load(path+files[4])['arr_0']
#
# plt.figure()
# plt.plot(range(size), f1[:, size//2])
# plt.plot(range(size), f2[:, size//2])
# plt.plot(range(size), f3[:, size//2])
# # plt.plot(range(size), f4[:, size//2])
# # plt.plot(range(size), f5[:, size//2])
#
# # plt.figure()
# # plt.imshow(f1)
# # plt.figure()
# # plt.imshow(f2)
# # plt.figure()
# # plt.imshow(f3)
# # plt.figure()
# # plt.imshow(f4)
# # plt.figure()
# # plt.imshow(f5)
# # plt.show()
#
# x = [114,114,114,115,125,124,127,130,124,140] # f f2 ; f1
# y = [124,125,126,125,117,116,117,116,138,128]
# # x = [123,124,127,128,128,125,120,131]  # df4 f2 ; f3
# # y = [118,119,118,118,135,135,134,132]
# # x = [128,111,121,134,142,143,140,133] # df7 f3
# # y = [109,119,110,110,118,131,138,143]
# # x = [128,111,121,134,142,143,140,133] # df10 f3
# # y = [109,119,110,110,118,131,138,143]
# for id in range(len(x)):
#     print(f3[x[id], y[id]] / f1[x[id], y[id]])
# combine = combine_img(f3, f2, 2.45) #17.6
# # plt.figure()
# # plt.imshow(combine)
# # plt.figure()
# # plt.plot(range(size), f1[:, size//2])
# # plt.plot(range(size), f2[:, size//2])
# # plt.plot(range(size), f3[:, size//2])
# # # plt.plot(range(size), f4[:, size//2])
# # plt.plot(range(size), combine[:, size//2])
# # plt.show()
# np.save(path+'combine_%s/df10' % size, f3)
# ------------ 有样品1024 by 0105 --------------
# size = 1024
# # files = ['f-40_%s.npz' % size, 'f-100_%s.npz' % size, 'f-300_%s.npz' % size]
# # files = ['df4-100_%s.npz' % size, 'df4-130_%s.npz' % size, 'df4-150_%s.npz' % size]
# # files = ['df7-110_%s.npz' % size, 'df7-200_%s.npz' % size, 'df7-250_%s.npz' % size, 'df7-450_%s.npz' % size]
# files = ['df10-450_%s.npz' % size, 'df10-700_%s.npz' % size, 'df10-900_%s.npz' % size, 'df10-1200_%s.npz' % size]
# f1 = np.load(path+files[0])['arr_0']
# f2 = np.load(path+files[1])['arr_0']
# f3 = np.load(path+files[2])['arr_0']
# f4 = np.load(path+files[3])['arr_0']
# # f5 = np.load(path+files[4])['arr_0']
#
# plt.figure()
# plt.plot(range(size), f1[:, size//2])
# plt.plot(range(size), f2[:, size//2])
# plt.plot(range(size), f3[:, size//2])
# plt.plot(range(size), f4[:, size//2])
# # plt.plot(range(size), f5[:, size//2])
#
# plt.figure()
# plt.imshow(f1)
# plt.figure()
# plt.imshow(f2)
# plt.figure()
# plt.imshow(f3)
# plt.figure()
# plt.imshow(f4)
# # plt.figure()
# # plt.imshow(f5)
# # plt.show()
#
# x = [114,114,114,115,125,124,127,130,124,140] # f f2 ; f1
# y = [124,125,126,125,117,116,117,116,138,128]
# # x = [123,124,127,128,128,125,120,131]  # df4 f1
# # y = [118,119,118,118,135,135,134,132]
# # x = [128,111,121,134,142,143,140,133] # df7 f2
# # y = [109,119,110,110,118,131,138,143]
# # x = [128,111,121,134,142,143,140,133] # df10 f2
# # y = [109,119,110,110,118,131,138,143]
# for id in range(len(x)):
#     print(f3[x[id], y[id]] / f1[x[id], y[id]])
# combine = combine_img(f3, f2, 2.45) #17.6
# # plt.figure()
# # plt.imshow(combine)
# # plt.figure()
# # plt.plot(range(size), f1[:, size//2])
# # plt.plot(range(size), f2[:, size//2])
# # plt.plot(range(size), f3[:, size//2])
# # # plt.plot(range(size), f4[:, size//2])
# # plt.plot(range(size), combine[:, size//2])
# plt.show()
# np.save(path+'combine_%s/df10' % size, f2)





# 产生1024和2048大小的图片

# 无样品图片处理
# f
# path = 'D:/00 论文相关/毕设/实验/data1217/无样品/'
#
# # files = ['f40_256.npz', 'f400_256.npz', 'f2000_256.npz']
# # f1 = np.load(path+files[0])['arr_0']
# # f2 = np.load(path+files[1])['arr_0']
# # f3 = np.load(path+files[2])['arr_0']
# #
# # plt.figure()
# # plt.plot(range(256), f1[:, 128])
# # plt.plot(range(256), f2[:, 128])
# # plt.plot(range(256), f3[:, 128])
# #
# #
# # plt.figure()
# # plt.imshow(f1)
# # plt.figure()
# # plt.imshow(f2)
# # plt.figure()
# # plt.imshow(f3)
# # plt.show()
# #
# # x = [120,116,135]
# # y = [111,142,146]
# # print(f3[x[0], y[0]] / f1[x[0], y[0]])
# # print(f3[x[1], y[1]] / f1[x[1], y[1]])
# # print(f3[x[2], y[2]] / f1[x[2], y[2]])
# # # print(f2[x[3], y[3]] / f1[x[3], y[3]])
# # # print(f2[x[4], y[4]] / f1[x[4], y[4]])
# # # print(f2[x[5], y[5]] / f1[x[5], y[5]])
# # # print(f2[x[6], y[6]] / f1[x[6], y[6]])
# # combine = f3.copy().astype('uint32')
# # rp = f1.copy()
# # for m in range(256):
# #     for n in range(256):
# #         if combine[m][n] >= 65520:
# #             combine[m][n] = rp[m][n] * 30
# #
# # plt.figure()
# # plt.imshow(combine)
# # plt.figure()
# # plt.plot(range(256), f1[:, 128])
# # plt.plot(range(256), f2[:, 128])
# # plt.plot(range(256), f3[:, 128])
# # plt.plot(range(256), combine[:, 128])
# # plt.show()
# # np.save(path+'combine/f', combine)
#
# f = np.load(path+'f40_256.npz')['arr_0']
# # f = np.load(path+'combine/f-x30.npy')
# full_size_aperture = 24000//3.69 # 目测透镜直径2.4cm
# # full_size_aperture
# tmp = np.arange(-full_size_aperture//2, full_size_aperture//2, full_size_aperture/256)
# x, y = np.meshgrid(tmp, tmp)
#
# aperture = np.zeros(f.shape)
# radius = 920
# aperture[np.sqrt(x**2+y**2)<=radius/2] = 1
# calc = np.square(np.abs(np.fft.fftshift(np.fft.fft2(aperture))))
#
# # f = f / np.max(f)
# # calc = calc / np.max(calc)
# calc = calc / 5.5
#
# plt.subplot(211)
# plt.imshow(f)
# plt.subplot(212)
# plt.imshow(calc)
# plt.show()
#
# plt.figure()
# plt.plot(range(256), f[:, 128])
# plt.plot(range(256), calc[:, 128])
# plt.show()
#
#
# # df
# path = 'D:/00 论文相关/毕设/实验/data1217/无样品/'
#
# files = ['-10df-200.raw', '-10df-500.raw', '-10df-1000.raw']
# # f1 = np.load(path+files[0])['arr_0']
# # f2 = np.load(path+files[1])['arr_0']
# # f3 = np.load(path+files[2])['arr_0']
# #
# # plt.figure()
# # plt.plot(range(256), f1[:, 128])
# # plt.plot(range(256), f2[:, 128])
# # plt.plot(range(256), f3[:, 128])
# #
# #
# # plt.figure()
# # plt.imshow(f1)
# # plt.figure()
# # plt.imshow(f2)
# # plt.figure()
# # plt.imshow(f3)
# # plt.show()
# #
# # x = [120,116,135]
# # y = [111,142,146]
# # print(f3[x[0], y[0]] / f1[x[0], y[0]])
# # print(f3[x[1], y[1]] / f1[x[1], y[1]])
# # print(f3[x[2], y[2]] / f1[x[2], y[2]])
# # # print(f2[x[3], y[3]] / f1[x[3], y[3]])
# # # print(f2[x[4], y[4]] / f1[x[4], y[4]])
# # # print(f2[x[5], y[5]] / f1[x[5], y[5]])
# # # print(f2[x[6], y[6]] / f1[x[6], y[6]])
# # combine = f3.copy().astype('uint32')
# # rp = f1.copy()
# # for m in range(256):
# #     for n in range(256):
# #         if combine[m][n] >= 65520:
# #             combine[m][n] = rp[m][n] * 30
# #
# # plt.figure()
# # plt.imshow(combine)
# # plt.figure()
# # plt.plot(range(256), f1[:, 128])
# # plt.plot(range(256), f2[:, 128])
# # plt.plot(range(256), f3[:, 128])
# # plt.plot(range(256), combine[:, 128])
# # plt.show()
# # np.save(path+'combine/f', combine)
#

# 仿真对比
# path = './exp/1221/wo_sample/combined/1024/'
# # f = np.load(path+'f_less_exposed.npy')
# # f = np.load(path+'combine/f-x30.npy')
# # full_size_aperture = 24000//3.69 # 目测透镜直径2.4cm
# full_size_aperture = 42.88 # 小孔直径5mm
# size = 1024
# tmp = np.arange(-full_size_aperture/2+full_size_aperture/size, full_size_aperture/2+full_size_aperture/size, full_size_aperture/size)
# x, y = np.meshgrid(tmp, tmp)
#
# aperture = np.zeros([size, size])
# diameter = 5
# aperture[np.sqrt(x**2+y**2)<=diameter/2] = 1
# calc = np.square(np.abs(np.fft.fftshift(np.fft.fft2(aperture))))
# np.save(path+'test_f', calc)
#
# # f = f / np.max(f)
# calc = calc / np.max(calc)
# # calc = calc / 5.5
#
# # plt.subplot(211)
# # plt.imshow(f)
# # plt.subplot(212)
# # plt.imshow(calc)
# # plt.show()
# #
# # plt.figure()
# # plt.plot(range(256), f[:, 128], color='r')
# # plt.plot(range(256), calc[:, 128], color='b')
# # plt.show()
#
# defocus_term = -(x**2+y**2) * 2 * np.pi / (632.991e-6 * 4 * 250**2) * 100# 单位mm
# defocus_term[np.sqrt(x**2+y**2)>diameter/2] = 0
#
# df4 = np.load(path+'df10.npy')
# calc_df4 = np.square(np.abs(np.fft.fftshift(np.fft.fft2(aperture*np.exp(1j*defocus_term*0.0)))))
# df4 = df4 / np.max(df4)
# calc_df4 = calc_df4 / np.max(calc_df4)
#
# saved = calc_df4[size//2-256:size//2+256, size//2-256:size//2+256] # 偏心
# print(saved.shape)
# np.save(path+'shift_test_f', saved)
#
# # plt.subplot(211)
# # plt.imshow(df4)
# # plt.subplot(212)
# plt.figure()
# plt.imshow(calc_df4)
# plt.show()
# #
# plt.figure()
# plt.plot(range(256), df4[:, 128])
# plt.plot(range(256), calc_df4[:, 128])
# plt.show()

# 查看一下背景光情况
# path = 'D:/00 论文相关/毕设/实验/data1230/背景光/'
# files = os.listdir(path)
# avg_img = np.zeros((2704,3376))
# for file in files:
#     print(file)
#     img = raw2npy(file)
#     avg_img = avg_img + img
#     print('%s: %f' % (file, np.average(img)))
# avg_img = avg_img / len(files)
# np.save(path+'avg_bkg', avg_img)


# 减背景噪声
# 先从average中根据center裁剪出512x512图片
# size = 1024
# # # center = [[1307,1365], [1312, 1385], [1315, 1402], [1318, 1419]] # for 有样品 1221
# # # center = [[1310,1363], [1314,1381], [1317,1395], [1320,1412]] # for 无样品 1221
# # # center = [[1321,1362], [1318, 1376], [1314, 1388], [1309, 1400]] # for 有样品 1228
# # # center = [[1322,1362], [1318,1377], [1314,1389], [1310,1400]]# for 无样品 1228
# # # center = [[1281,1362], [1277,1378], [1273,1390], [1270,1402]] # for 有样品 1230
# # #
# # # path = 'D:/11projects/Deep-phase-decoder/auto_defocus/exp/1230/w_sample/combined/avg_bkg/'
# # # avg_bkg = np.load(path+'avg_bkg.npy')
# # # name = ['f', 'df4', 'df7', 'df10']
# # # for c, n in zip(center, name):
# # #     img = crop_img(avg_bkg,c[1],c[0],size)
# # #     np.save(path + '%s/' % size + n, img)
# # # # 再将原图片去噪
# # # path1 = 'D:/11projects/Deep-phase-decoder/auto_defocus/exp/1228/wo_sample/combined/%s/'%size
# # # path2 = 'D:/11projects/Deep-phase-decoder/auto_defocus/exp/1228/wo_sample/combined/avg_bkg/%s/'%size
# # # path3 = 'D:/11projects/Deep-phase-decoder/auto_defocus/exp/1228/wo_sample/combined/denoised/%s/'%size
# path1 = 'D:/11projects/Deep-phase-decoder/auto_defocus/exp/1230/w_sample/combined/%s/'%size
# path2 = 'D:/11projects/Deep-phase-decoder/auto_defocus/exp/1230/w_sample/combined/avg_bkg/%s/'%size
# path3 = 'D:/11projects/Deep-phase-decoder/auto_defocus/exp/1230/w_sample/combined/denoised/%s/'%size
# names = os.listdir(path1)
# for name in names:
#     original_img = np.load(path1+name)
#     corresponding_bkg = np.load(path2+name)
#     # print(np.average(corresponding_bkg))
#     new_img = original_img - corresponding_bkg
#     print(np.min(new_img), np.max(new_img))
#     new_img[new_img<0] = 0
#     new_img = new_img.astype('uint32')
#     print(np.min(new_img), np.max(new_img))
#     plt.figure()
#     plt.plot(range(size), original_img[:, size//2], color='r')
#     plt.plot(range(size), new_img[:, size//2], color='b')
#     plt.show()
#     # print(new_img.shape)
#     np.save(path3+name, new_img)

# img_size = 256
# full_size_aperture = 42.77   #单位mm
# tmp = np.arange(-full_size_aperture/2+full_size_aperture/img_size, full_size_aperture/2+full_size_aperture/img_size, full_size_aperture/img_size)
# # print(tmp.shape)
# x, y = np.meshgrid(tmp, tmp)
# diameter = 5
# aperture = np.zeros((img_size,img_size))
# aperture[np.sqrt(x**2+y**2)<=diameter/2] = 1
# ideal = np.square(np.abs(np.fft.fftshift(np.fft.fft2(aperture))))
# plt.subplot(131)
# plt.imshow(aperture, cmap='gray')
# plt.title('aperture')
# plt.axis('off')
# plt.colorbar()
# plt.subplot(132)
# plt.imshow(ideal, cmap='gray')
# plt.title('ideal')
# plt.axis('off')
# plt.colorbar()
# plt.subplot(133)
# plt.plot(range(img_size), ideal[:, img_size//2], color='b')
# plt.show()
