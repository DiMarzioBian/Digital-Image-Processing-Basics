import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
    Statistically, median filter outperforms gaussian filter on MSE.

    Subjectively, median filter also gives a better view. One reason could be that gaussian filter
    cannot handle salt and pepper noise very well since they are sensitive to neighbor values.

    And since we are not using very big sigma for gaussian noise, median filter output still preserve
    many details of original images, I've tried bigger sigma and the result is getting worse on details.
"""


def add_noise(img, mean=0, sigma=5, snr_sp=0.8):
    n_g = np.random.normal(mean, sigma, img.shape)  # gaussian noise
    n_sp = np.random.choice([0, -1, 1], size=img.shape, replace=True, p=[snr_sp, (1 - snr_sp) / 2, (1 - snr_sp) / 2])

    img_n = np.clip(np.round(img + n_g), a_min=0, a_max=255)
    img_n[n_sp == -1] = 0
    img_n[n_sp == 1] = 255
    return img_n


def filter_median(img, k_size=5):
    assert k_size > 0 and k_size % 2 == 1
    H, W = img.shape
    img_new = np.zeros_like(img)
    img_pad = np.pad(img, k_size // 2, 'reflect')
    for h in range(H):
        for w in range(W):
            img_new[h, w] = np.median(img_pad[h: h + k_size, w: w + k_size])
    return img_new


def main():
    img_l = cv2.imread('lena.tif', flags=0)  # flags = 0 to read grayscale images
    img_f = cv2.imread('flower.tif', flags=0)

    img_ln = add_noise(img_l)
    mse_ln = np.mean(np.power((img_ln - img_l), 2))

    img_fn = add_noise(img_f)
    mse_fn = np.mean(np.power((img_fn - img_f), 2))

    # 3: 98.0602, 5: 87.2455, 7: 121.1098, 11: 183.7591, 21: 356.2935
    img_lnm = filter_median(img_ln, k_size=5)
    mse_lnm = np.mean(np.power((img_lnm - img_l), 2))

    # 3: 376.9795, 5: 386.0963, 7: 389.4670, 9: 390.9661
    img_lng = cv2.GaussianBlur(img_ln, (11, 11), 3)
    mse_lng = np.mean(np.power((img_lng - img_l), 2))

    # 3: 55.0973, 5: 23.7782, 7: 32.8653, 9: 41.8818
    img_fnm = filter_median(img_fn, k_size=5)
    mse_fnm = np.mean(np.power((img_fnm - img_f), 2))

    # 3: 598.5196, 9: 216.4980, 13 :184.5909, 15: 179.8268, 17: 178.5319, 19: 179.4055, 21: 181.6983
    img_fng = cv2.GaussianBlur(img_fn, (17, 17), 0)
    mse_fng = np.mean(np.power((img_fng - img_f), 2))

    plt.figure(0)
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(10, 10)
    axs[0, 0].imshow(img_l, cmap='Greys_r')
    axs[0, 0].set_title(f'Original image: lena.tif')
    axs[0, 1].imshow(img_ln, cmap='Greys_r')
    axs[0, 1].set_title(f'Added 2 noises, MSE = {mse_ln:.4f}')
    axs[1, 0].imshow(img_lnm, cmap='Greys_r')
    axs[1, 0].set_title(f'Median filter, MSE = {mse_lnm:.4f}')
    axs[1, 1].imshow(img_lng, cmap='Greys_r')
    axs[1, 1].set_title(f'Gaussian filter, MSE = {mse_lng:.4f}')
    plt.show()

    plt.figure(1)
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(10, 10)
    axs[0, 0].imshow(img_f, cmap='Greys_r')
    axs[0, 0].set_title(f'Original image: lena.tif')
    axs[0, 1].imshow(img_fn, cmap='Greys_r')
    axs[0, 1].set_title(f'Added 2 noises, MSE = {mse_fn:.4f}')
    axs[1, 0].imshow(img_fnm, cmap='Greys_r')
    axs[1, 0].set_title(f'Median filter, MSE = {mse_fnm:.4f}')
    axs[1, 1].imshow(img_fng, cmap='Greys_r')
    axs[1, 1].set_title(f'Gaussian filter, MSE = {mse_fng:.4f}')
    plt.show()


if __name__ == '__main__':
    main()
