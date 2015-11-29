import cv2
import cv2.cv as cv
import numpy as np

# Read the video
vid = cv2.VideoCapture("football/rolling_ball.avi")
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 10.0, (320,240))
while(vid.isOpened()):
    _, f = vid.read()
    if f == None: break
    # Hough dinges
    # height, width = f.shape[:2]
    # f = cv2.resize(f,(width/2, height/2))
    hsv = cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
    im = cv2.GaussianBlur(hsv, (15,15), 0, 0)
    green = cv2.inRange(im,np.array((53,100, 50)),np.array((90,255,255)))
    res = cv2.bitwise_or(f, f, mask= green)
    # s = green[:,1:]
    edges = cv2.Canny(green,15,30,apertureSize = 3,  L2gradient=True)
    cimg = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(cimg,cv.CV_HOUGH_GRADIENT,1,30,
                                param1=30,param2=22,minRadius=5,maxRadius=60)

    # cv2.imshow("cimg",edges)
    # print(circles)
    best_circle = None
    best_val = 0
    best_roi = 0;
    if circles is not None:
        print("found ball!")
        circles = np.uint16(np.around(circles))
        #sums = []

        for i in circles[0,:]:
            # cv::Mat roi = img(cv::Range(circle[1]-circle[2], circle[1]+circle[2]+1), cv::Range(circle[0]-circle[2], circle[0]+circle[2]+1))
            # 4 Points of Region of interest square
            x_min = i[0] - i[2]
            x_max = i[0] + i[2]
            y_min = i[1] - i[2]
            y_max = i[1] + i[2]

            # For now just take the square
            roi = f[y_min:y_max, x_min:x_max]

            # Calculate average
            roi_sum = np.sum(np.sum(roi, axis=0), axis=0) / (np.shape(roi)[0]*np.shape(roi)[1])
            print roi_sum

            # The ball has the most blue compared to other ROI's
            # Not really robust, probably gets confused when a nao is in the image
            # Works for now
            if roi_sum[2] > best_val:
                best_val = roi_sum[2]
                best_circle = i
                best_roi = roi



            #sums.append(roi_sum)
    if not best_circle == None:
        cv2.circle(f,(best_circle[0],best_circle[1]),best_circle[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(f,(best_circle[0],best_circle[1]),2,(0,0,255),3)
    cv2.imshow( "ROI", f)
    out.write(f)
    if cv2.waitKey(100) & 0xFF == ord('q'):
    	break



vid.release()
out.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
