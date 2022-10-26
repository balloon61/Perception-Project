import cv2
import numpy as np



if __name__ == '__main__':
    # input image
    image = cv2.imread("Q1image.png")
    # gray scale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # get the binary image
    (threshold, bw) = cv2.threshold(gray_img, 230, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # obtain its size
    M, N = bw.shape
    # setup the kernel size
    kernel1 = np.ones((25, 25), np.uint8)
    kernel2 = np.ones((13, 13), np.uint8)
    # erode the image
    erosion = cv2.erode(bw, kernel1)
    # dilate the iamge
    dilation = cv2.dilate(erosion, kernel2)
    cv2.imshow("", dilation)
    # label how many coins in this image
    count, labels = cv2.connectedComponents(dilation)

    max = 0
    # count how many coins
    for i in range(M-1):
        for j in range(N-1):
            if labels[i][j] > max:
                max = labels[i][j]
    print("There have", max, "coins")
    cv2.waitKey(10000)