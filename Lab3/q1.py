import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
"""


def rotate_hsv_clockwise(img, angle):
    img = img.astype(np.float)
    img[:, :, 0] += angle
    img[:, :, 0][img[:, :, 0] >= 180] -= 180
    return img.astype(np.uint8)


def rgb_to_hsv(img):
    img = img.astype(np.float)
    Height, Width, Channels = img.shape
    img_hsv = np.zeros_like(img, dtype=np.float)

    for height in range(Height):
        for width in range(Width):
            (r, g, b) = img[height, width, :]

            c_max = np.max([r, g, b])
            c_min = np.min([r, g, b])
            c_range = (c_max-c_min)

            if c_min == c_max:
                img_hsv[height, width, :] = [0, 0, c_max]
                continue

            rc = (c_max - r) * 60 / c_range
            gc = (c_max - g) * 60 / c_range
            bc = (c_max - b) * 60 / c_range
            if r == c_max:
                h = bc - gc
            elif g == c_max:
                h = 120 + rc - bc
            else:
                h = 240 + gc - rc

            h += 360 if h < 0 else 0
            img_hsv[height, width, :] = [h / 2, (c_max - c_min) * 255 / c_max, c_max]

    return np.floor(img_hsv).astype(np.uint8)


def hsv_to_bgr(img):
    img = img.astype(np.float)
    Height, Width, Channels = img.shape
    img_rgb = np.zeros_like(img, dtype=np.float)

    for height in range(Height):
        for width in range(Width):
            (h, s, v) = img[height, width, :]
            s /= 255
            f = h / 30 % 1
            hi = int(h / 30 % 6)

            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)

            if hi == 0:
                r, g, b = v, t, p
            elif hi == 1:
                r, g, b = q, v, p
            elif hi == 2:
                r, g, b = p, v, t
            elif hi == 3:
                r, g, b = p, q, v
            elif hi == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q

            img_rgb[height, width, :] = [r, g, b]

    return np.floor(img_rgb).astype(np.uint8)


def main():
    img = cv2.imread('flowers.jpg')[:, :, ::-1]  # convert bgr to rgb
    mask = np.clip(np.repeat(np.expand_dims(cv2.imread('mask.tif', flags=0), axis=-1), 3, axis=-1), a_min=0, a_max=1)

    img_hsv = rgb_to_hsv(img)
    img_hsv_cv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    img_hsv_r1 = rotate_hsv_clockwise(img_hsv, 60)
    img_hsv_r2 = rotate_hsv_clockwise(img_hsv, 120)
    img_hsv_r1_cv = rotate_hsv_clockwise(img_hsv_cv, 60)
    img_hsv_r2_cv = rotate_hsv_clockwise(img_hsv_cv, 120)

    img_r1 = hsv_to_bgr(img_hsv_r1)
    img_r2 = hsv_to_bgr(img_hsv_r2)
    img_r1_cv2 = cv2.cvtColor(img_hsv_r1_cv, cv2.COLOR_HSV2RGB)
    img_r2_cv2 = cv2.cvtColor(img_hsv_r2_cv, cv2.COLOR_HSV2RGB)

    plt.figure(0)
    fig, axs = plt.subplots(3, 2)
    fig.set_size_inches(10, 12)

    axs[0, 0].imshow(img * mask)
    axs[0, 0].set_title(f'Masked image: flowers.jpg')

    axs[0, 1].imshow(img[:, :, ::-1] * mask)
    axs[0, 1].set_title(f'Image swapped R and B')

    axs[1, 0].imshow(img_r1 * mask)
    axs[1, 0].set_title(f'Manual rotation | angle=60')

    axs[1, 1].imshow(img_r2 * mask)
    axs[1, 1].set_title(f'Manual rotation | angle=120')

    axs[2, 0].imshow(img_r1_cv2 * mask)
    axs[2, 0].set_title(f'OpenCV rotation | angle=60')

    axs[2, 1].imshow(img_r2_cv2 * mask)
    axs[2, 1].set_title(f'OpenCV rotation | angle=120')
    plt.show()


if __name__ == '__main__':
    main()
