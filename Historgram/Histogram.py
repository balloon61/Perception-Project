import cv2
import numpy as np
filelist = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
folder_dir = "adaptive_hist_data/00000000"
file_type = ".png"
filetest = ['00']

def Calculate_cdf(image, h, w):

    cdf = np.zeros((256, 3), dtype=float) # BGR
    Accumulate = np.zeros((256, 3), dtype=float) # BGR
    Acc = np.zeros((256, 3), dtype=int)  # BGR
    for i in range(h):
        for j in range(w):
            for l in range(3):
                cdf[image[i][j][l]][l] = cdf[image[i][j][l]][l] + 1   # l = 0 B, l = 1 G, l = 2 R
    cdf = cdf / (h * w)
    for length in range(len(cdf)):
        for color in range(3):
            if length == 0:
                Accumulate[length][color] = cdf[length][color]
            else:
                Accumulate[length][color] = cdf[length][color] + Accumulate[length - 1][color]
            Acc[length][color] = round(Accumulate[length][color] * 255)


    return cdf, Acc


def histogram(image):

    height = image.shape[0]
    width = image.shape[1]
    H_image = np.zeros((height, width, 3), np.uint8) # BGR

    _, A = Calculate_cdf(image, height, width)
    # print(A)
    for i in range(height):
        for j in range(width):
            for l in range(3):
                H_image[i][j][l] = A[image[i][j][l]][l]

    return H_image


def adaptive_histogram(image):

    height = image.shape[0]
    width = image.shape[1]
    A_H_image = np.zeros((height, width, 3), np.uint8) # BGR
    M = height // 10
    N = width // 8
    tiles = [image[x:x + M, y:y + N] for x in range(0, height, M) for y in range(0, width, N)]
    row_count = 0
    col_count = 0
    for t in tiles:

        _, A = Calculate_cdf(t, M, N)
        for i in range(M):
            for j in range(N):
                for l in range(3):
                    A_H_image[i + row_count * M][j + col_count * N][l] = A[t[i][j][l]][l]
        if (col_count + 1) % 8 == 0:
            row_count = row_count + 1
        col_count = (col_count + 1) % 8
    return A_H_image



if __name__ == '__main__':
    # check if the image ends with png
    for images_name in filelist:

        im = cv2.imread(folder_dir + images_name + file_type)
        original_image = im
        H = histogram(im)
        A_H = adaptive_histogram(im)

        image_concat = np.concatenate((original_image, H, A_H), axis=0)
        cv2.imshow("Top: Original, Mid: Histogram, Bot: Adaptive_Histogram", image_concat)

        cv2.imwrite("output" + images_name + ".png", image_concat)
        cv2.waitKey(1000)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

