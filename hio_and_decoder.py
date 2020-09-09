import numpy as np
import imageio
import matplotlib.pyplot as plt
from phase_retrieval import fienup_phase_retrieval
import cv2

from include import *

def noise_intensity(original, noisy):
    return  np.sum(np.abs(np.abs(original)-noisy)) / np.sum(np.abs(original))

def get_noisy_img(img_np, sig=30000,noise_same = False):
    sigma = sig/255.
    if noise_same: # add the same noise in each channel
        noise = np.random.normal(scale=sigma, size=img_np.shape[1:])
        noise = np.array( [noise]*img_np.shape[0] )
    else: # add independent noise in each channel
        noise = np.random.normal(scale=sigma, size=img_np.shape)

    # img_noisy_np = np.clip( img_np + noise , 0, 1).astype(np.float32)
    # 对于振幅加噪，无需clip处理，故无需归一化，只需要确保噪声强度在合理范围即可
    img_noisy_np = (img_np+noise)

    # img_noisy_var = np_to_var(img_noisy_np).type(dtype)
    return img_noisy_np   # ,img_noisy_var


np.random.seed(233)
image = imageio.imread('cameraman.png', as_gray=True)
image = cv2.resize(image, (304,304))
np.save("./original_img", image)
##### 获取加噪图像  x : 应该是给测量值加噪声，孙天宇论文也是如此
original_image = image
# image /= 255.
# image = get_noisy_img(image)
# image *= 255.


print(image.shape)
w, h = image.shape
w_mask, h_mask = 512, 512
mask = np.zeros(shape=(w_mask, h_mask), dtype=np.int8)

# print(w_mask//2-w//2, w_mask//2+w//2)
mask[w_mask//2-w//2:w_mask//2+w//2, h_mask//2-h//2:h_mask//2+h//2] = 1
# cv2.imshow("mask", mask)
image_mask = np.zeros(shape=mask.shape,dtype=np.float32)
# print(mask[w_mask//2,:])


#np.putmask(image_mask, mask==1, image)
# indices = np.logical
print(image_mask.shape, mask.shape, image.shape)
# indices = mask ==1
# print(indices.shape)
image_mask[w_mask//2-w//2:w_mask//2+w//2, h_mask//2-h//2:h_mask//2+h//2] = image
# cv2.imshow("mask", mask)

# 很多论文将8bit灰度级映射到 0-2π之间的phase image，此处是否也应如此？

magnitudes = np.abs(np.fft.fft2(image_mask))
# 给测量值加上噪声
original_mag = magnitudes
max_mag = np.max(magnitudes)
np.save("./original_magnitudes", magnitudes)
# magnitudes = get_noisy_img(magnitudes)
np.save("./noise_magnitudes", magnitudes)

# 这里有点问题，在测量值上，仅用噪声水平（强度）进行评估，而对重建图像，采用psnr比较好坏
# print("psnr: %.2f" % psnr(original_mag, magnitudes, 255.))  # 若归一化了，最大值用1，否则用255
print("noise intensity on magnitudes: %.10f" % noise_intensity(original_mag, magnitudes))


print(image_mask.dtype)
print(image.dtype)

# plt.subplot(121)
# plt.imshow(image_mask, cmap='gray')
# plt.title('image_mask')
# plt.subplot(122)
# plt.imshow(image, cmap='gray');
# plt.title('image')
# plt.show()

result = fienup_phase_retrieval(magnitudes,
                                mask=mask,
                                steps=5000,    # 次数参考孙天宇论文
                                verbose=True,
                                use_decoder=True,
                                support_shape=(w,h))
# cv2.imshow("result", result)
# plt.show()
result = result[w_mask//2-w//2:w_mask//2+w//2, h_mask//2-h//2:h_mask//2+h//2]

# 以ndarray的形式保存恢复结果
np.save("./result", result)

print("psnr: %.2f" % psnr(original_image, result, 255.))

plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Image')
plt.subplot(122)
plt.imshow(result, cmap='gray');
plt.title('Reconstruction')
plt.show()
