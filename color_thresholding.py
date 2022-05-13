#!/usr/bin/env python3 

import rospy 
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge, CvBridgeError 
import cv2 
import numpy as np

class finding_center(object):

    def __init__ (self):
            self.image2_sub = rospy.Subscriber("/vbmbot/camera1/image_raw",Image,self.camera_callback) 
            self.bridge_object = CvBridge() 

 
    def camera_callback(self,data): 
        try: 
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8") 
        except CvBridgeError as e: 
            print(e) 
#       image_path = '/home/rbe450X-ros/src/gazebo_ros_demos/platform/Scripts/rosraw_image_object.png' 
        image2 = cv_image
          


        hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV) 


        min_green = np.array([36,50,70]) 
        max_green = np.array([89,255,255]) 

        min_red = np.array([0,50,70]) 
        max_red = np.array([3,255,255]) 

        min_blue = np.array([90,50,70]) 
        max_blue = np.array([128,255,255]) 
        
        min_purple = np.array([129,50,70]) 
        max_purple = np.array([158,255,255])


        mask_g = cv2.inRange(hsv, min_green, max_green) 
        mask_r = cv2.inRange(hsv, min_red, max_red) 
        mask_b = cv2.inRange(hsv, min_blue, max_blue) 
        mask_p = cv2.inRange(hsv, min_purple, max_purple)
 
        res_b = cv2.bitwise_and(image2,image2, mask= mask_b) 
        res_g = cv2.bitwise_and(image2,image2, mask= mask_g) 
        res_r = cv2.bitwise_and(image2,image2, mask= mask_r) 
        res_p = cv2.bitwise_and(image2,image2, mask= mask_p) 
 
        # calculate moments of binary image

        m = cv2.moments(mask_r)
        mb = cv2.moments(mask_b)
        mg = cv2.moments(mask_g)
        mp = cv2.moments(mask_p)



	# calculate x,y coordinate of center

        cXr = int(m["m10"] / m["m00"])
        cYr = int(m["m01"] / m["m00"])
        
        cXb = int(mb["m10"] / mb["m00"])
        cYb = int(mb["m01"] / mb["m00"])
        
        cXg = int(mg["m10"] / mg["m00"])
        cYg = int(mg["m01"] / mg["m00"])
        
        cXp = int(mp["m10"] / mp["m00"])
        cYp = int(mp["m01"] / mp["m00"])

	# put text and highlight the center

        cv2.circle(image2, (cXr, cYr), 3, (255, 255, 255), -1)
        cv2.circle(image2, (cXb, cYb), 3, (255, 255, 255), -1)
        cv2.circle(image2, (cXg, cYg), 3, (255, 255, 255), -1)
        cv2.circle(image2, (cXp, cYp), 3, (255, 255, 255), -1)

        cv2.putText(image2, "centroid", (cXr - 25, cYr - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(image2, "centroid", (cXb - 25, cYb - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(image2, "centroid", (cXg - 25, cYg - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(image2, "centroid", (cXp - 25, cYp - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.imshow('Orignal',image2)
        if self.run_once == 0:
            cv2.imwrite('orignal.png', image2)
            self.run_once = 1
        
        print(cXr, cYr) 
        print(cXb, cYb)
        print(cXg, cYg)
        print(cXp, cYp)
    #    cv2.imshow('Green',res_g) 
    #    cv2.imshow('Red',res_r) 
    #    cv2.imshow('Blue',res_b) 
    #    cv2.imshow('Purple',res_p) 

        cv2.waitKey(1) 
   

def main(): 
    finding_center_object = finding_center() 
    rospy.init_node('center_detection_node', anonymous=True) 
    try: 
        rospy.spin() 
    except KeyboardInterrupt: 
        print("Shutting down") 
    cv2.destroyAllWindows() 

if __name__ == '__main__': 
    main() 
