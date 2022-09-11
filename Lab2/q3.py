import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    H[u, v] = 1 / (1 + (D0 / D) ** (2 * n))
    RuntimeWarning: divide by zero encountered in double_scalars
    Since D is zero, (D0 / D) will be infinity and the final value will be set to 0
"""


def filter_butterworth(f_img, D0, n, type_pass='high'):
    type_high = True if type_pass == 'high' else False
    H, W = f_img.shape
    H_filter = np.zeros((H, W), dtype=np.float)

    for h in range(H):
        for w in range(W):
            D = np.sqrt((h - H / 2) ** 2 + (w - W / 2) ** 2)
            if type_high:
                H_filter[h, w] = 1 / (1 + (D0 / D) ** (2 * n))
            else:
                H_filter[h, w] = 1 / (1 + (D / D0) ** (2 * n))

    ff_img = f_img * H_filter
    return np.abs(np.fft.ifft2(np.fft.ifftshift(ff_img))), H_filter


def main():
    img = cv2.imread('lena.tif', flags=0)  # flags = 0 to read grayscale images
    f_img = np.fft.fft2(img)
    fs_img = np.fft.fftshift(f_img)

    img_hp, hpf = filter_butterworth(fs_img, D0=10, n=2, type_pass='high')
    img_lp, lpf = filter_butterworth(fs_img, D0=10, n=2, type_pass='low')

    plt.figure(0)
    fig, axs = plt.subplots(2, 4)
    fig.set_size_inches(15, 10)
    axs[0, 0].imshow(img, cmap='Greys_r')
    axs[0, 0].set_title(f'Original image: lena.tif')
    axs[1, 0].imshow(img, cmap='Greys_r')
    axs[1, 0].set_title(f'Original image: lena.tif')

    axs[0, 1].imshow(np.log1p(np.abs(f_img)), cmap='Greys_r')
    axs[0, 1].set_title(f'images in freq domain')
    axs[1, 1].imshow(np.log1p(np.abs(fs_img)), cmap='Greys_r')
    axs[1, 1].set_title(f'images in freq domain shifted')

    axs[0, 2].imshow(hpf, cmap='Greys_r')
    axs[0, 2].set_title(f'High-pass filter')
    axs[1, 2].imshow(lpf, cmap='Greys_r')
    axs[1, 2].set_title(f'Low-pass filter')

    axs[0, 3].imshow(img_hp, cmap='Greys_r')
    axs[0, 3].set_title(f'High-pass output')
    axs[1, 3].imshow(img_lp, cmap='Greys_r')
    axs[1, 3].set_title(f'Low-pass output')

    plt.show()


if __name__ == '__main__':
    main()
