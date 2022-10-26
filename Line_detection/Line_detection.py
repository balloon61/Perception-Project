import cv2
import numpy as np
import math

thre = 2
m_avg_pre = 0
if __name__ == '__main__':
    cap = cv2.VideoCapture('whiteline.mp4')
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (17, 17), 0)
            _, th1 = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

            #th1 = cv2.Canny(th1, 100, 200, None, 3)

            # Copy edges to the images that will display the results in BGR
            gray_BGR = cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR)
            gray_BGR_copy = np.copy(gray_BGR)
            linesP2 = cv2.HoughLinesP(th1, 1, np.pi / 180, 200, None, 50, 10)

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

            linesP = cv2.HoughLinesP(th1, 1, np.pi / 180, 50, None, 50, 10)
            m_avg = 0
            for i in range(len(m)):
                #print(m[i])
                m_avg = m_avg + m[i]
            if len(m) > 0:
                m_avg = m_avg / len(m)
            else:
                #print("m = 0")
                m_avg = m_avg_pre
            #print(m_avg)
            if linesP is not None:
                for i in range(0, len(linesP)):
                    l = linesP[i][0]
                    l1.append(l)
                    print("13", ((l[1] - l[3]) / (l[0] - l[2])))
                    if np.abs(m_avg - ((l[1] - l[3]) / (l[0] - l[2]))) < thre:
                        cv2.line(gray_BGR_copy, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

            m_avg_pre = m_avg
            #cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
            cv2.imshow("Detected Lines (in Green) - Probabilistic Line Transform", gray_BGR_copy)
            #cv2.imshow('Frame', th1)

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
