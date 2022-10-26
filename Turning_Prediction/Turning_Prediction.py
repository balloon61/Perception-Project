import cv2
import numpy as np


thre = 2
m_avg_pre = 0
curvature_right = 0
curvature_left = 0
d = ''
rem_slope = 0
length = 50
points_of_interest = np.array([[310, 40], [550, 40], [900, 330], [1, 330]])
projection_area = np.array([[100, 40], [300, 40], [300, 330], [100, 330]])

# Desired points value in output images
points_tl = [310, 40]
points_tr = [550, 40]
points_br = [2000, 330]
points_bl = [10, 330]

converted_tl = [50, 50]
converted_tr = [360, 50]
converted_br = [360, 360]
converted_bl = [50, 360]
kernel = np.ones((3, 3), np.uint8)
# Convert points
point_matrix = np.float32([points_tl, points_tr, points_br, points_bl])
projection_area = np.float32([converted_tl, converted_tr, converted_br, converted_bl])

def Find_Curvature(l):
    m1 = -float((l[0][0][0] - l[0][0][2]) / (l[0][0][1] - l[0][0][3]))
    a1 = (l[0][0][0] + l[0][0][2]) / 2
    b1 = (l[0][0][1] + l[0][0][3]) / 2
    m2 = -float((l[1][0][0] - l[1][0][2]) / (l[1][0][1] - l[1][0][3]))
    a2 = (l[1][0][0] + l[1][0][2]) / 2
    b2 = (l[1][0][1] + l[1][0][3]) / 2

    intersect_x = (b2 - b1 + m1 * a1 - m2 * a2) / (m1 - m2)
    intersect_y = m2 * intersect_x - m2 * a2 + b2
    if intersect_x > l[0][0][0]:
        direction = 'TURN RIGHT'  # turn right
    elif intersect_x == l[0][0][0]:
        direction = 'GO STRAIGHT'  # go straight
    else:
        direction = 'TURN LEFT'  # tuen left
    return np.sqrt((intersect_y - l[0][0][1]) * (intersect_y - l[0][0][1]) + (intersect_x - l[0][0][0]) * (intersect_x - l[0][0][0])).astype('str'), direction

if __name__ == '__main__':
    cap = cv2.VideoCapture('challenge.mp4')
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            height, width, channels = frame.shape
            clp = frame[5*height // 9:height, width // 4:width * 7 // 8]

            perspective_transform = cv2.getPerspectiveTransform(point_matrix, projection_area)
            img_Output = cv2.warpPerspective(clp, perspective_transform, (300, 330))
            img_Output = img_Output[0:height, 0:width // 6]
            gray = cv2.cvtColor(img_Output, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            _, th1 = cv2.threshold(blur, 140, 255, cv2.THRESH_BINARY)
            th1 = cv2.erode(th1, kernel)
            th1 = cv2.Canny(th1, 100, 200, None, 3)
            gray_BGR = cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR)
            gray_BGR_copy = np.copy(gray_BGR)
            height2, width2, channels2 = gray_BGR.shape

            linesP2 = cv2.HoughLinesP(th1[0:height2, 0:width2 // 2], 1, np.pi / 180, 1, None, 20, 10)
            have_horizontal_line1 = 0
            have_horizontal_line2 = 0
            l1 = []
            l2 = []
            if linesP2 is not None:
                if len(linesP2) > 1:
                    curvature_left, d = Find_Curvature(linesP2)
                for i in range(0, len(linesP2)):
                    l = linesP2[i][0]
                    if l[0] - l[2] != 0:
                        m = (float((l[1] - [3]) / (l[0] - l[2])))

                        if abs(m) < 1:
                            have_horizontal_line2 = have_horizontal_line2 + 1
                        else:
                            rem_slope = m
                            cv2.line(gray_BGR_copy, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 3, cv2.LINE_AA)
                if have_horizontal_line2 == 0:
                    if len(linesP2) > 1:
                        curvature_right, d = Find_Curvature(linesP2)
                else:
                    cv2.imshow("Result", gray_BGR)
                    continue
            linesP1 = cv2.HoughLinesP(th1[0:height2, width2 // 2:width2], 1, np.pi / 180, 1, None, 10, 3)


            if linesP1 is not None:

                for i in range(0, len(linesP1)):
                    l = linesP1[i][0]

                    if l[0] - l[2] != 0:
                        m = (float((l[1] - [3]) / (l[0] - l[2])))
                        if abs(m) < 1:
                            have_horizontal_line1 = have_horizontal_line1 + 1
                        else:
                            cv2.line(gray_BGR_copy, (l[0] + width2 // 2, l[1]), (l[2] + width2 // 2, l[3]), (0, 0, 255), 3, cv2.LINE_AA)
                if have_horizontal_line1 == 0:
                    if len(linesP1) > 1:
                        curvature_right, d = Find_Curvature(linesP1)
            # Copy edges to the images that will display the results in BGR

           # gray_BGR = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
           # gray_BGR_copy = np.copy(gray_BGR)

            print("Left curvature:", curvature_left, "Right curvature:", curvature_right)
            cv2.imshow("Result", gray_BGR_copy)
            cv2.putText(frame, "Right Radius:" + curvature_right, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Left Radius:" + curvature_left, (100, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, d, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            frame = cv2.arrowedLine(frame, (673, 619), (673+int(np.floor(np.sqrt(length * length / (1+rem_slope*rem_slope)))), 619 + int(np.floor(rem_slope * np.sqrt(length * length/(1+rem_slope*rem_slope))))), (0, 0, 255), 5, 8, 0, 0.3)
            cv2.imshow("Original", frame)
            # cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
            # cv2.imshow("Lane detection", th1)
            # cv2.imshow("Warping", img_Output)
            # cv2.imshow("Detected Lines (in Green) - Probabilistic Line Transform", img_Output)

            # cv2.imshow('Frame', th1)

            # Press Q on keyboard to  exit

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# Copy edges to the images that will display the results in BGR
gray_BGR = cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR)
gray_BGR_copy = np.copy(gray_BGR)
linesP2 = cv2.HoughLinesP(th1, 1, np.pi / 180, 30, None, 10, 10)

l1 = []
l2 = []
m = []
if linesP2 is not None:
    for i in range(0, len(linesP2)):
        l = linesP2[i][0]
        l2.append(l)
        m.append(float((l[1] - [3]) / (l[0] - l[2])))
        cv2.line(gray_BGR_copy, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 3, cv2.LINE_AA)
# cv2.imshow("Source", th1)
"""
linesP = cv2.HoughLinesP(th1, 1, np.pi / 180, 5, None, 10, 10)
m_avg = 0
for i in range(len(m)):
    # print(m[i])
    m_avg = m_avg + m[i]
if len(m) > 0:
    m_avg = m_avg / len(m)
else:
    # print("m = 0")
    m_avg = m_avg_pre
# print(m_avg)
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        l1.append(l)
        print("13", ((l[1] - l[3]) / (l[0] - l[2])))
        if np.abs(m_avg - ((l[1] - l[3]) / (l[0] - l[2]))) < thre:
            cv2.line(gray_BGR_copy, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

m_avg_pre = m_avg
"""