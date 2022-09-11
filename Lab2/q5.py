import cv2
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import matplotlib.pyplot as plt

"""
    H[u, v] = 1 / (1 + (D0 / D) ** (2 * n))
    RuntimeWarning: divide by zero encountered in double_scalars
    Since D is zero, (D0 / D) will be infinity and the final value will be set to 0
"""


def get_psf(shape, angle, dist):
    assert shape[0] == shape[1]
    center = shape[0] / 2
    sinVal = np.sin(angle * np.pi / 180)
    cosVal = np.cos(angle * np.pi / 180)
    psf = np.zeros(shape)
    for i in range(dist):
        x_offset, y_offset = np.round(sinVal * i), np.round(cosVal * i)
        psf[int(center - x_offset), int(center + y_offset)] = 1

    return psf / psf.sum()


def filter_wiener(img_fft, psf_fft, snr=0.01):
    psf_fft_conj = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + 1 / snr)
    return np.abs(fftshift(ifft2(img_fft * psf_fft_conj)))


def filter_inverse(img_fft, psf_fft):
    return np.abs(fftshift(ifft2(img_fft / psf_fft)))


def main():
    angle, dist, snr = 180, 3, 0.99
    eps = 1e-5

    img = cv2.imread('coins_blurred.tif', flags=0)  # flags = 0 to read grayscale images
    img_fft = fft2(img)

    psf = get_psf(img_fft.shape, angle, dist)
    psf_fft = fft2(psf) + eps

    img_i = filter_inverse(img_fft, psf_fft)
    img_w = filter_wiener(img_fft, psf_fft, snr=snr)

    plt.figure(0)
    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(15, 10)
    axs[0, 0].imshow(img, cmap='Greys_r')
    axs[0, 0].set_title(f'Original image: coins_blurred.tif')
    axs[0, 1].imshow(np.log1p(np.abs(img_fft)), cmap='Greys_r')
    axs[0, 1].set_title(f'images in freq domain')
    axs[0, 2].imshow(psf, cmap='Greys_r')
    axs[0, 2].set_title(f'PSF | angle={angle}, dist={dist}')

    axs[1, 0].imshow(np.log1p(np.abs(psf_fft)), cmap='Greys_r')
    axs[1, 0].set_title(f'PSF in freq domain | eps={eps}')
    axs[1, 1].imshow(img_i, cmap='Greys_r')
    axs[1, 1].set_title(f'Inverse filtered image')
    axs[1, 2].imshow(img_w, cmap='Greys_r')
    axs[1, 2].set_title(f'Wiener filtered image | snr={snr}')

    plt.show()


if __name__ == '__main__':
    main()
