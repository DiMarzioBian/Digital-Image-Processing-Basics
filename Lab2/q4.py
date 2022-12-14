import cv2
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import matplotlib.pyplot as plt

"""
    H[u, v] = 1 / (1 + (D0 / D) ** (2 * n))
    RuntimeWarning: divide by zero encountered in double_scalars
    Since D is zero, (D0 / D) will be infinity and the final value will be set to 0
"""


def filter_band_pass_ellipse(img_ffts, D0_LP, D0_HP, n):
    H, W = img_ffts.shape
    H_filter = np.zeros((H, W), dtype=np.float)

    for h in range(H):
        for w in range(W):
            D = np.sqrt(((h - H / 2) / 3) ** 2 + ((w - W / 2) / 1) ** 2)
            H_filter[h, w] += 1 / (1 + (D0_LP / D) ** (2 * n))
            H_filter[h, w] += 1 / (1 + (D / D0_HP) ** (2 * n))

    img_ffts *= H_filter
    return np.abs(ifft2(ifftshift(img_ffts))), img_ffts, H_filter


def main():
    img = cv2.imread('pepper_corrupt.tif', flags=0)  # flags = 0 to read grayscale images
    img_fft = fft2(img)
    img_ffts = fftshift(img_fft)

    D0_LP = 18
    D0_HP = 10
    n = 100
    img_r, img_fft_r, h_f = filter_band_pass_ellipse(img_ffts, D0_LP=D0_LP, D0_HP=D0_HP, n=n)

    plt.figure(0)
    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(15, 10)
    axs[0, 0].imshow(img, cmap='Greys_r')
    axs[0, 0].set_title(f'Original image: pepper_corrupt.tif')
    axs[0, 1].imshow(np.log1p(np.abs(img_fft)), cmap='Greys_r')
    axs[0, 1].set_title(f'images in freq domain')
    axs[0, 2].imshow(np.log1p(np.abs(fftshift(img_fft))), cmap='Greys_r')
    axs[0, 2].set_title(f'images in freq domain shifted')

    axs[1, 0].imshow(h_f, cmap='Greys_r')
    axs[1, 0].set_title(f'Band-pass filter | ({D0_LP}, {D0_HP}, n={n})')
    axs[1, 1].imshow(np.log1p(np.abs(img_fft_r)), cmap='Greys_r')
    axs[1, 1].set_title(f'Restored image in freq domain shifted')
    axs[1, 2].imshow(img_r, cmap='Greys_r')
    axs[1, 2].set_title(f'Restored image')

    plt.show()


if __name__ == '__main__':
    main()
