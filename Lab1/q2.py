import cv2
import matplotlib.pyplot as plt

import keyboard


def main():
    img_l = cv2.imread('Lena.bmp', flags=0)  # flags = 0 to read grayscale images
    img_m = cv2.imread('Mandrill.bmp', flags=0)
    img_p = cv2.imread('Peppers.bmp', flags=0)

    i = 0
    map_level = {0: 2, 1: 4, 2: 8, 3: 16}
    level = map_level[i]

    while True:
        print('\nPress esc to terminate.')
        print('Press any key to continue.')
        if keyboard.read_key() != 'esc':
            print(f'Drawing resized images by {level}...\n')

            img_l_new = cv2.resize(img_l, (img_l.shape[0] // level, img_l.shape[1] // level), cv2.INTER_NEAREST)
            img_m_new = cv2.resize(img_m, (img_l.shape[0] // level, img_l.shape[1] // level), cv2.INTER_NEAREST)
            img_p_new = cv2.resize(img_p, (img_l.shape[0] // level, img_l.shape[1] // level), cv2.INTER_NEAREST)

            fig, axs = plt.subplots(1, 3)
            fig.set_size_inches(10, 3)
            fig.suptitle(f'Resized images by {level}.')
            axs[0].imshow(img_l_new, cmap='Greys_r')
            axs[1].imshow(img_m_new, cmap='Greys_r')
            axs[2].imshow(img_p_new, cmap='Greys_r')
            plt.show()
        else:
            break

        i += 1
        if i >= len(map_level):
            break
        else:
            level = map_level[i]


if __name__ == '__main__':
    main()
