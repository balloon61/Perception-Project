import numpy as np
import cv2

import matplotlib.pyplot as plt
from functools import reduce
import operator
import math
import operator

kernel = np.ones((7, 7), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)

bottom_is_square_count = 0



def My_fft(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()
    rows, cols = gray.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.subplot(121), plt.imshow(gray, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_back, cmap='gray')
    plt.title('FFT and HPF'), plt.xticks([]), plt.yticks([])

    plt.show()


def Get_ID(img):
    tag = []
    topRight_img = img[75: 100, 100: 125]
    topLeft_img = img[75:100, 75:100]
    bottomRight_img = img[100:125, 100:125]
    bottom_left_img = img[100:125, 75:100]

    TR_black_number = 0
    TR_white_number = 0
    TL_black_number = 0
    TL_white_number = 0
    BR_black_number = 0
    BR_white_number = 0
    BL_black_number = 0
    BL_white_number = 0
    for i in range(25):
        for j in range(25):
            if topRight_img[i][j] == 0:
                TR_black_number = TR_black_number + 1
            else:
                TR_white_number = TR_white_number + 1
            if topLeft_img[i][j] == 0:
                TL_black_number = TL_black_number + 1
            else:
                TL_white_number = TL_white_number + 1
            if bottomRight_img[i][j] == 0:
                BR_black_number = BR_black_number + 1
            else:
                BR_white_number = BR_white_number + 1
            if bottom_left_img[i][j] == 0:
                BL_black_number = BL_black_number + 1
            else:
                BL_white_number = BL_white_number + 1
    if TL_black_number > TL_white_number:
        tag.append(0)
    else:
        tag.append(1)
    if TR_black_number > TR_white_number:
        tag.append(0)
    else:
        tag.append(1)
    if BR_black_number > BR_white_number:
        tag.append(0)
    else:
        tag.append(1)
    if BL_black_number > BL_white_number:
        tag.append(0)
    else:
        tag.append(1)

    return tag

def Homography(c1, c2):
    A = []
    for i in range(len(c1)):
        x, y = c1[i][0], c1[i][1]
        u, v = c2[i][0], c2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A)
    U, S, V = np.linalg.svd(A)

    return (V[-1, :] / V[-1, -1]).reshape(3, 3)


def projectionMatrix(h, K):
    b = np.abs(2 / (np.linalg.norm(np.linalg.inv(K).dot(h[:, 0])) + np.linalg.norm(np.linalg.inv(K).dot(h[:, 1]))) * np.linalg.inv(K).dot(h))
    Rt = np.column_stack((b[:, 0], b[:, 1], np.cross(b[:, 0], b[:, 1]), b[:, 2]))
    return np.matmul(K, Rt), Rt, b[:, 2]


def warpPerspective(H, Height, Width, AR_img):
    warped = np.zeros((Height, Width, 3), np.uint8)
    for i in range(Width):
        for j in range(Height):
            x, y, z = np.linalg.inv(H).dot(np.reshape([j, i, 1], (3, 1)))
            warped[j][i] = AR_img[int(y / z)][int(x / z)]
    return warped


def determinePoints(out):
    tl = out[0]
    tr = out[1]
    br = out[2]
    bl = out[3]
    MaxW = max(int(np.sqrt((br[0] - bl[0]) * (br[0] - bl[0]) + (br[1] - bl[1]) * (br[1] - bl[1]))),
               int(np.sqrt((tr[0] - tl[0]) * (tr[0] - tl[0]) + (tr[1] - tl[1]) * (tr[1] - tl[1]))))
    MaxH = max(int(np.sqrt((tr[0] - br[0]) * (tr[0] - br[0]) + (tr[1] - br[1]) * (tr[1] - br[1]))),
               int(np.sqrt((tl[0] - bl[0]) * (tl[0] - bl[0]) + ((tl[1] - bl[1]) * (tl[1] - bl[1])))))

    dst = np.array([[0, 0], [MaxW - 1, 0], [MaxW - 1, MaxH - 1], [0, MaxH - 1]], dtype="float32")
    return tl, tr, br, bl, MaxW, MaxH, dst


def rotateImage(image, angle):
    rot_mat = cv2.getRotationMatrix2D(tuple(np.array(image.shape[1::-1]) / 2), angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)


def Draw(img):
    ret, threshed = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
    resized_image = cv2.resize(threshed, (200, 200))
    return draw8Grid(resized_image)


def drawContour(img):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(img2, 240, 250, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # finding contours

    len_of_all_contours = []
    getclockwise_x = []
    getclockwise_y = []
    for c in contours:
        len_of_all_contours.append(len(c))
    max_ind = len_of_all_contours.index(max(len_of_all_contours))


    for i in range(len(contours[max_ind])):
        for c in contours[max_ind][i]:
            getclockwise_x.append(c[0])
            getclockwise_y.append(c[1])
    tup = [(min(getclockwise_x), min(getclockwise_y)), (max(getclockwise_x), min(getclockwise_y)), (min(getclockwise_x), max(getclockwise_y)), (max(getclockwise_x), max(getclockwise_y))]

    rect = sorted(tup, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, tuple(
        map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), tup), [len(tup)] * 2))))[::1]))) % 360)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)
    return img, rect

def draw8Grid(img):

    for i in range(1, 9):
        cv2.line(img, (int(img.shape[0] / 8) * i, 0), (int(img.shape[0] / 8) * i, img.shape[0]), (125, 0, 0), 1, 1)
        cv2.line(img, (0, int(img.shape[1] / 8) * i), (img.shape[1], int(img.shape[1] / 8) * i), (125, 0, 0), 1, 1)
    return img


def CheckWhite(img):
    checksquare = img[125:150, 125:150]
    list_black_or_white_1 = []
    for i in range(25):
        for j in range(25):
            list_black_or_white_1.append(checksquare[i][j])
    count_black_1 = list_black_or_white_1.count(0)  # checking count of black
    count_white_1 = list_black_or_white_1.count(255)  # checking count of white
    return count_black_1 < count_white_1

cap = cv2.VideoCapture('1tagvideo.mp4')
count = 0

while (cap.isOpened()):
    ret, image = cap.read()
    if ret == True:
        image2 = image

        gray_img = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        # blur the image
        blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
        # get the binary image, and find out the edge
        (threshold, bw) = cv2.threshold(blur_img, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_erosion = cv2.erode(bw, kernel, iterations=1)
        img_dilation = cv2.dilate(img_erosion, kernel2, iterations=1)
        canny = cv2.Canny(img_dilation, 100, 200)

        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[3]
        cv2.drawContours(image, [cnt], 0, (0, 0, 255), 2)
        cv2.imshow("Original", image)

        x, y, w, h = cv2.boundingRect(cnt)
        min_x, max_x = min(x, 5000), max(x + w, 0)
        min_y, max_y = min(y, 5000), max(y + h, 0)
        (threshold2, bw2) = cv2.threshold(cv2.cvtColor(image2[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY), 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imshow("", bw2)
        C = 0
        for i in range(len(bw2)):
            if bw2[i][0] == 0:
                print(i)
                C = i
                break
        S = 0
        for i in range(len(bw2[0])):
            if bw2[0][i] == 0:
                S = i
                break
        print(np.arctan2(S, C) * 180 / np.pi)
        image = rotateImage(image, -np.arctan2(C, S) * 180 / np.pi)

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # blur the image
        blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
        # get the binary image, and find out the edge
        (threshold, bw) = cv2.threshold(blur_img, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_erosion = cv2.erode(bw, kernel, iterations=1)
        img_dilation = cv2.dilate(img_erosion, kernel2, iterations=1)
        canny = cv2.Canny(img_dilation, 100, 200)

        contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[3]
        x, y, w, h = cv2.boundingRect(cnt)
        min_x, max_x = min(x, 5000), max(x + w, 0)
        min_y, max_y = min(y, 5000), max(y + h, 0)
        (threshold2, bw2) = cv2.threshold(cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY), 230, 255,
                                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        Corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
        print("Corners:\n", Corners)
        contour_img, out = drawContour(image[y:y + h, x:x + w])
        topLeft, topRight, bottomRight, bottomLeft, maxWidth, maxHeight, d = determinePoints(out)
        warped = warpPerspective(Homography(np.array([topLeft, topRight, bottomRight, bottomLeft], np.float32), d),
                                 maxWidth, maxHeight, image[y:y + h, x:x + w])

        eight_grid_threshed = cv2.cvtColor(Draw(warped), cv2.COLOR_BGR2GRAY)
        target = image[y:y + h, x:x + w]
        target = cv2.resize(target, (200, 200))
        cv2.imshow("ARTag", target)

        MyTag = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        MyTag = Draw(MyTag)
        cv2.imshow('Grid Image', MyTag)
        rotate_times = 0
        while CheckWhite(MyTag) == False and rotate_times < 4:
            MyTag = rotateImage(MyTag, 90)
            rotate_times = rotate_times + 1
        #    cv2.imshow('ROTATED', rotated)
        ID = Get_ID(MyTag)
        ID_num = 8 * int(ID[0]) + 4 * int(ID[1]) + 2 * int(ID[2]) + int(ID[3])
        print('Tag ID: ', ID, "\nNumber:", ID_num)

    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()