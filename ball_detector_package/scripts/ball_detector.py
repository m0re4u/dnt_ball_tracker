#!/usr/bin/env python
import sys
import rospy
import cv2
import tf
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class balldetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.pixToMeters = 20
        self.br = tf.TransformBroadcaster()
        self.image_sub = rospy.Subscriber("/nao_camera/image_raw",Image,self.callback)

    def callback(self,data):
        rospy.loginfo("Received data")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        hsv = cv2.cvtColor(cv_image,cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv,np.array((53,100, 50)),np.array((90,255,255)))
        s = green[:,1:]
        im = cv2.GaussianBlur(hsv, (23,23), 0, 0)

        edges = cv2.Canny(im,60,180,apertureSize = 3,  L2gradient=True)

        contours,hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(edges, contours, -1, (255,0,0), 3)

        params = cv2.SimpleBlobDetector_Params()
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.5

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 120

        # Filter by Inertia
        params.filterByInertia = False
        params.minInertiaRatio = 0.01


        detector = cv2.SimpleBlobDetector(params)

        # Detect blobs.
        keypoints = detector.detect(edges)
        if len(keypoints) > 0:
            print keypoints[0].pt[0]
            print keypoints[0].pt[1]
            print keypoints[0].size

            sizeBall = keypoints[0].size
            distance = (self.pixToMeters / sizeBall)
            print distance
            print "-------------"
            self.br.sendTransform((distance, 0.0, 0.0),
                         tf.transformations.quaternion_from_euler(0, 0, 0),
                         rospy.Time.now(),
                         "ball",
                         "CameraTop_frame")

        im_with_keypoints = cv2.drawKeypoints(edges, keypoints, np.array([]), (0,255,100), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('detect ball',im_with_keypoints)
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
