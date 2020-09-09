import numpy as np
import imageio
import matplotlib.pyplot as plt
from phase_retrieval import fienup_phase_retrieval
import cv2

# np.random.seed(1)
image = imageio.imread('lena.jpg', as_gray=True)
image = cv2.resize(image, (318,318))
print(image.shape)
w, h = image.shape
w_mask, h_mask = 512, 512
mask = np.zeros(shape=(w_mask, h_mask), dtype=np.int8)

# print(w_mask//2-w//2, w_mask//2+w//2)
mask[w_mask//2-w//2:w_mask//2+w//2, h_mask//2-h//2:h_mask//2+h//2] = 1
# cv2.imshow("mask", mask)
image_mask = np.zeros(shape=mask.shape,dtype=np.float32)
print(mask[w_mask//2,:])


#np.putmask(image_mask, mask==1, image)
# indices = np.logical
print(image_mask.shape, mask.shape, image.shape)
# indices = mask ==1
# print(indices.shape)
image_mask[w_mask//2-w//2:w_mask//2+w//2, h_mask//2-h//2:h_mask//2+h//2] = image
# cv2.imshow("mask", mask)
magnitudes = np.abs(np.fft.fft2(image_mask))

print(image_mask.dtype)
print(image.dtype)
plt.show()
plt.subplot(121)
plt.imshow(image_mask, cmap='gray')
plt.title('image_mask')
plt.subplot(122)
plt.imshow(image, cmap='gray');
plt.title('image')
plt.show()

result = fienup_phase_retrieval(magnitudes, 
                                mask=mask,
                                steps=5000,
                                verbose=True)
cv2.imshow("result", result)
# plt.show()
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('Image')
plt.subplot(122)
plt.imshow(result, cmap='gray');
plt.title('Reconstruction')
plt.show()
