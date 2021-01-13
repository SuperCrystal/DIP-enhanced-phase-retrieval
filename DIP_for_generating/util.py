import matplotlib.pyplot as plt
import numpy as np
import torchvision

# Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a
# torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image
# belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the
# numpy.ndarray has dtype = np.uint8
# In the other cases, tensors are returned without scaling.
data_transform = torchvision.transforms.Compose(
                    [torchvision.transforms.ToTensor()])

# 注意传进来的灰度图最好不要有多余的通道数
# fftshift是否必要？？
def ft2(g, delta):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(g))) * delta**2

def ift2(G, delta_f):
    N = G.shape[0]
    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(G))) * (N *delta_f)**2

def cyl(x, y, r0):
    r = np.sqrt(x**2 + y**2)
    z = (r<r0) * 1.0
    return z


def two_step_prop_ASM(Uin, wvl, d1, d2, Dz):
    N = Uin.shape[0]
    k = 2 * np.pi / wvl

    tmp = np.arange(-N//2, N//2)
    x1, y1 = np.meshgrid(tmp*d1, tmp*d1)
    r1sq = x1**2 + y1**2

    df1 = 1 / (N*d1)
    fx, fy = np.meshgrid(tmp*df1, tmp*df1)
    fsq = fx**2 + fy**2

    m = d2 / d1

    x2, y2 = np.meshgrid(tmp*d2, tmp*d2)
    r2sq = x2**2 + y2**2

    # fsq[128,128] = 1e-9
    # fsq[fsq==0] = 1e-7    # 避免除0的问题
    Q1 = np.exp(1j*k/2*(1-m)/Dz*r1sq)
    Q2 = np.exp(-1j*(np.pi**2)*2*Dz/m/k*fsq)
    Q3 = np.exp(1j*k/2*(m-1)/(m*Dz)*r2sq)

    Uout = Q3 * ift2(Q2 * ft2(Q1*Uin/m, d1), df1)

    return Uout

def two_step_prop_fresnel(Uin, wvl, d1, d2, Dz):
    N = Uin.shape[0]
    k = 2 * np.pi / wvl

    tmp = np.arange(-N//2, N//2)
    x1, y1 = np.meshgrid(tmp*d1, tmp*d1)

    m = d2 / d1
    Dz1 = Dz / (1-m)
    d1a = wvl * np.abs(Dz1) / (N * d1)
    x1a, y1a = np.meshgrid(tmp*d1a, tmp*d1a)

    Uitm = (1 / (1j*wvl*Dz1)
            * np.exp(1j*k/(2*Dz1)*(x1a**2+y1a**2))
            * ft2(Uin*np.exp(1j*k/(2*Dz1)*(x1**2+y1**2)), d1))

    Dz2 = Dz - Dz1

    x2, y2 = np.meshgrid(tmp*d2, tmp*d2)

    Uout = (1 / (1j*wvl*Dz2)
            * np.exp(1j*k/(2*Dz2)*(x2**2+y2**2))
            * ft2(Uitm*np.exp(1j*k/(2*Dz2)*(x1a**2+y1a**2)), d1a))
    return Uout


def test_asm():
    wavelength = 632.8e-6
    N = 256
    width = 10
    k = 2*np.pi/wavelength
    f = 100
    z = f

    tmp = np.arange(-width//2, width//2, width/N)
    # print(tmp.shape)
    x, y = np.meshgrid(tmp, tmp)
    # print(x)
    # print(y)
    R = np.sqrt(x**2 + y**2)
    R[R>width/2] = 0

    lens = np.exp(-1j * k * R**2 / (2*f))* cyl(x,y,width/2)

    u0 = lens
    uz1 = two_step_prop_ASM(u0, wavelength, width/N, 0.1*width/N, z)
    uz2 = two_step_prop_fresnel(u0, wavelength, width/N, 0.1*width/N, z)

    # print(np.angle(u0))
    # print(np.log(np.abs(uz))[120:130,120:130])
    plt.subplot(221)
    plt.imshow(np.abs(uz1)**2, cmap='gray')
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(np.abs(uz2)**2, cmap='gray')
    plt.colorbar()
    # plt.imshow(np.log(np.abs(uz)**2+1), cmap='gray')
    plt.subplot(212)
    # plt.plot((np.abs(uz1)**2)[N//2,:], color='r')
    # plt.plot((np.abs(uz2)**2)[N//2,:], color='b')
    plt.imshow(np.angle(uz2))
    plt.show()


if __name__ == '__main__':
    test_asm()
