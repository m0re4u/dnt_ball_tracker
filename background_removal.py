import cv2
import numpy as np

pixToMeters = 19.98

# Read the video
vid = cv2.VideoCapture("football/rolling_ball.avi")

# f = cv2.imread("football/image1.jpg")
# height, width = f.shape[:2]
# f = cv2.resize(f, (width/2, height/2) )
while(vid.isOpened()):
    _, f = vid.read()

    if f == None: break
    hsv = cv2.cvtColor(f,cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv,np.array((53,100, 100)),np.array((90,255,255)))
    s = green[:,1:]
    im = cv2.GaussianBlur(s, (23,23), 0, 0)

    edges = cv2.Canny(im,60,180,apertureSize = 3,  L2gradient=True)

    contours,hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(edges, contours, -1, (255,0,0), 3)

    params = cv2.SimpleBlobDetector_Params()

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.75

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 120

    detector = cv2.SimpleBlobDetector(params)

    # Detect blobs.
    keypoints = detector.detect(edges)
    if len(keypoints) > 0:
        print keypoints[0].pt[0]
        print keypoints[0].pt[1]
        print keypoints[0].size

        sizeBall = keypoints[0].size
        distance = (pixToMeters / sizeBall)
        print distance
        print "-------------"

    # Draw detected blobs as blue circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(f, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    cv2.imshow('detect ball',im_with_keypoints)
    # cv2.imshow('edges',edges)

    if cv2.waitKey(100) & 0xFF == ord('q'):
    	break

#cv2.waitKey(0)
vid.release()

cv2.waitKey(0)
cv2.destroyAllWindows()
