import numpy as np
# from denoising import do_denoise
from denoise_and_compare import do_denoise

def fienup_phase_retrieval(mag, mask=None, beta=0.9,
                           steps=200, mode='hybrid', verbose=True, use_decoder=False, support_shape=(None,None)):
    """
    Implementation of Fienup's phase-retrieval methods. This function
    implements the input-output, the output-output and the hybrid method.

    Note: Mode 'output-output' and beta=1 results in
    the Gerchberg-Saxton algorithm.

    Parameters:
        mag: Measured magnitudes of Fourier transform
        mask: Binary array indicating where the image should be
              if padding is known
        beta: Positive step size
        steps: Number of iterations
        mode: Which algorithm to use
              (can be 'input-output', 'output-output' or 'hybrid')
        verbose: If True, progress is shown

    Returns:
        x: Reconstructed image

    Author: Tobias Uelwer
    Date: 30.12.2018

    References:
    [1] E. Osherovich, Numerical methods for phase retrieval, 2012,
        https://arxiv.org/abs/1203.4756
    [2] J. R. Fienup, Phase retrieval algorithms: a comparison, 1982,
        https://www.osapublishing.org/ao/abstract.cfm?uri=ao-21-15-2758
    [3] https://github.com/cwg45/Image-Reconstruction
    """

    assert beta > 0, 'step size must be a positive number'
    assert steps > 0, 'steps must be a positive number'
    assert mode == 'input-output' or mode == 'output-output'\
        or mode == 'hybrid',\
    'mode must be \'input-output\', \'output-output\' or \'hybrid\''

    if mask is None:
        mask = np.ones(mag.shape)

    assert mag.shape == mask.shape, 'mask and mag must have same shape'

    # sample random phase and initialize image x
    y_hat = mag*np.exp(1j*2*np.pi*np.random.rand(*mag.shape))
    x = np.zeros(mag.shape)

    # previous iterate
    x_p = None

    # 在振幅平面去噪
    # 行不通，强度不是自然图像，不能用deep decoder去噪
    # if use_decoder:
    #     mag = do_denoise(mag)

    # main loop
    for i in range(1, steps+1):
        # show progress
        if i % 100 == 0 and verbose:
            print("step", i, "of", steps)

        # inverse fourier transform
        y = np.real(np.fft.ifft2(y_hat))

        # 此处表示将y经过decoder去噪。（x、y都是物面，x'、y'都是像面）
        # 问题在于，测量值上的加性高斯噪声，到原图变成乘性的高斯噪声，原图结构恢复的不好，故decoder去噪失效
        if (use_decoder and (i >= 1000 and i < steps-200) and (i % 200 == 0)):  # 每500 iter做一次去噪？
            # 支撑域包括(304,304)范围，还包括非负
            # 只对(304,304)域内部进行！
            w = support_shape[0]
            h = support_shape[1]
            w_mask = mask.shape[0]
            h_mask = mask.shape[1]
            y[w_mask//2-w//2:w_mask//2+w//2, h_mask//2-h//2:h_mask//2+h//2] = do_denoise(y[w_mask//2-w//2:w_mask//2+w//2, h_mask//2-h//2:h_mask//2+h//2])

        # previous iterate
        if x_p is None:
            x_p = y
        else:
            x_p = x

        # updates for elements that satisfy object domain constraints
        if mode == "output-output" or mode == "hybrid":
            x = y

        # find elements that violate object domain constraints
        # or are not masked
        indices = np.logical_or(np.logical_and(y<0, mask),
                                np.logical_not(mask))

        # updates for elements that violate object domain constraints
        if mode == "hybrid" or mode == "input-output":
            x[indices] = x_p[indices]-beta*y[indices]
        elif mode == "output-output":
            x[indices] = y[indices]-beta*y[indices]

        # 如果是接在y后面，y还有一些不合理的负值，尝试接在x后面呢？
        # if (use_decoder and (i > 3000) and (i % 500 == 0)):  # 每500 iter做一次去噪？
        #     # 支撑域包括(304,304)范围，还包括非负
        #     # 只对(304,304)域内部进行！
        #     w = support_shape[0]
        #     h = support_shape[1]
        #     w_mask = mask.shape[0]
        #     h_mask = mask.shape[1]
        #     x[w_mask//2-w//2:w_mask//2+w//2, h_mask//2-h//2:h_mask//2+h//2] = do_denoise(x[w_mask//2-w//2:w_mask//2+w//2, h_mask//2-h//2:h_mask//2+h//2])


        # fourier transform
        x_hat = np.fft.fft2(x)

        # satisfy fourier domain constraints
        # (replace magnitude with input magnitude)
        y_hat = mag*np.exp(1j*np.angle(x_hat))
    return x
