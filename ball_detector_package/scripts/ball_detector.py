#!/usr/bin/env python
import sys
import rospy
import cv2
import cv2.cv as cv
import tf
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class balldetector():
    def __init__(self):
        self.bridge = CvBridge()
        self.pixToMeters = 19.975
        self.br = tf.TransformBroadcaster()
        self.image_sub = rospy.Subscriber("/nao_camera/image_raw",Image,self.callback)
        rospy.loginfo("Initialised balldetector")

    def callback(self,data):
        rospy.loginfo("Received data")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # BLOB DETECTION
        # height, width = cv_image.shape[:2]
        # cv_image = cv2.resize(cv_image,(width/2, height/2))
        # hsv = cv2.cvtColor(cv_image,cv2.COLOR_BGR2HSV)
        # green = cv2.inRange(hsv,np.array((53,100, 50)),np.array((90,255,255)))
        # s = green[:,1:]
        # im = cv2.GaussianBlur(s, (23,23), 0, 0)
        #
        # edges = cv2.Canny(im,60,180,apertureSize = 3,  L2gradient=True)
        #
        # contours,hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(edges, contours, -1, (255,0,0), 3)
        #
        # params = cv2.SimpleBlobDetector_Params()
        # # Filter by Circularity
        # params.filterByCircularity = False
        # params.minCircularity = 0.5
        #
        # # Filter by Convexity
        # params.filterByConvexity = True
        # params.minConvexity = 0.75
        #
        # # Filter by Area.
        # params.filterByArea = True
        # params.minArea = 120
        #
        # # Filter by Inertia
        # params.filterByInertia = False
        # params.minInertiaRatio = 0.01
        #
        #
        # detector = cv2.SimpleBlobDetector(params)
        #
        # # Detect blobs.
        # keypoints = detector.detect(edges)
        # print(len(keypoints))
        # if len(keypoints) > 0:
        #     # Keypoints x and y
        #     # print keypoints[0].pt[0]
        #     # print keypoints[0].pt[1]
        #     print keypoints[0].size
        #
        #     sizeBall = keypoints[0].size
        #     distance = (self.pixToMeters / sizeBall)
        #     print distance
        #     print "-------------"
        #     self.br.sendTransform((distance, 0.0, 0.0),
        #                  tf.transformations.quaternion_from_euler(0, 0, 0),
        #                  rospy.Time.now(),
        #                  "ball",
        #                  "CameraTop_frame")
        #
        # im_with_keypoints = cv2.drawKeypoints(edges, keypoints, np.array([]), (0,255,100), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.imshow('detect ball',im_with_keypoints)
        # cv2.waitKey(3)

        # Hough dinges
        height, width = cv_image.shape[:2]
        cv_image = cv2.resize(cv_image,(width/2, height/2))
        hsv = cv2.cvtColor(cv_image,cv2.COLOR_BGR2HSV)
        im = cv2.GaussianBlur(hsv, (15,15), 0, 0)
        green = cv2.inRange(im,np.array((53,100, 50)),np.array((90,255,255)))
        res = cv2.bitwise_or(cv_image, cv_image, mask= green)
        # s = green[:,1:]
        edges = cv2.Canny(green,15,30,apertureSize = 3,  L2gradient=True)
        cimg = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(cimg,cv.CV_HOUGH_GRADIENT,1,30,
                                    param1=30,param2=25,minRadius=20,maxRadius=60)

        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # cv::Mat roi = img(cv::Range(circle[1]-circle[2], circle[1]+circle[2]+1), cv::Range(circle[0]-circle[2], circle[0]+circle[2]+1))
            # 4 Points of Region of interest square
            x_min = i[0] - i[2]
            x_max = i[0] + i[2]
            y_min = i[1] - i[2]
            y_max = i[1] + i[2]
            roi = cv_image[x_min:x_max, y_min:y_max]
            print roi

            # Working on creating a mask on the coloured image
            rows, columns = roi.shape[:2]
            mask = [rows, columns]
            print mask


            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
            # break
        cv2.imshow('detected circles',cimg)
        cv2.imshow("edges", edges)
        cv2.waitKey(3)

    # Draw detected blobs as blue circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # im_with_keypoints = cv2.drawKeypoints(cv_image, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow('detect ball',im_with_keypoints)

def main():
    bd = balldetector()
    rospy.init_node('ball_detector', anonymous=True)
    rospy.loginfo("Starting ball detector")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()
