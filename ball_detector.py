#!/usr/bin/env python
import sys
import rospy
import cv2
import tf
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


bridge = CvBridge()

def callback(data):
    rospy.loginfo("Received data")
    try:
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    hsv = cv2.cvtColor(cv_image,cv2.COLOR_BGR2HSV)
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
        br.sendTransform((distance, 0, 0),
                     (0,0,0),
                     rospy.Time.now(),
                     "ball",
                     "CameraTop_frame")




    # Draw detected blobs as blue circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # im_with_keypoints = cv2.drawKeypoints(cv_image, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow('detect ball',im_with_keypoints)

def main():
    image_sub = rospy.Subscriber("/nao_camera/image_raw",Image,callback)
    rospy.init_node('ball_detector', anonymous=True)
    br = tf.TransformBroadcaster()
    rospy.loginfo("Starting ball detector")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()
