import numpy as np
import cv2


def Estimate_F(x1, x2):
    n = x1.shape[1]
    x1 = x1 / 1080
    x2 = x2 / 1080

    A = np.zeros((x1.shape[0], 9))
    for i in range(x1.shape[0]):
        A[i, :] = [x2[i, 0] * x1[i, 0], x2[i, 0] * x1[i, 1], x2[i, 0], x2[i, 1] * x1[i, 0], x2[i, 1] * x1[i, 1],
                   x2[i, 1], x1[i, 0], x1[i, 1], 1]

    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    F = F / F[2, 2]

    return F


def Get_Features_Array(matches, kp1, kp2):
    pt1 = []
    pt2 = []
    for i in range(len(matches)):
        pt1.append(kp1[matches[i][0].queryIdx].pt)
        pt2.append(kp2[matches[i][0].trainIdx].pt)
    return np.array(pt1), np.array(pt2)


def decompose_R_T(E):
    u, s, vt = np.linalg.svd(E)
    t = u.dot(np.matrix([[0, -1, 0], [1, 0, 0], [0, 0, 0]])).dot(u.T)
    C = u[:, 2]
    R1 = np.matmul(np.matmul(u, np.matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])), vt)
    R2 = np.matmul(np.matmul(u, np.matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T), vt)
    if np.linalg.det(R1) > 0:
        return t, C, R1, R2
    else:
        return t, -C, R1, R2


def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def check_window(window_l, right, size, x_lower_bound, x_upper_bound, y_lower_bound,
                y_upper_bound):
    min_ssd = 10000
    min_index = (0, 0)

    for y in range(y_lower_bound, y_upper_bound):
        for x in range(x_lower_bound, x_upper_bound):
            window_r = right[y: y + size, x: x + size]
            if window_l.shape == window_r.shape:
                ssd = np.sum((window_l - window_r) * (window_l - window_r))
                if ssd < min_ssd:
                    min_ssd = ssd
                    min_index = (y, x)
            else:
                continue

    return min_index


def correspondence(img1, img2, height, width, window, x_window, y_window):

    disparity_map = np.zeros((height, width))

    for y in range(window, height - window - 1):
        for x in range(window, width - window - 1):
            block_left = img1[y:y + window, x:x + window]
            if x - x_window < 0:
                x_lower_bound = 0
            else:
                x_lower_bound = x - x_window
            if x - x_window > img2.shape[1]:
                x_upper_bound = img2.shape[1]
            else:
                x_upper_bound = x + x_window
            if y - y_window < 0:
                y_lower_bound = 0
            else:
                y_lower_bound = y - y_window
            if y - y_window > img2.shape[0]:
                y_upper_bound = img2.shape[0]
            else:
                y_upper_bound = y + y_window

            min_index = check_window(block_left, img2, window, x_lower_bound, x_upper_bound, y_lower_bound,
                                         y_upper_bound)


            disparity_map[y, x] = np.abs(min_index[1] - x)
        print(y)

    return disparity_map

