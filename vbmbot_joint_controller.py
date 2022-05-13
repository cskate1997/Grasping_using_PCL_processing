#!/usr/bin/env python3
# license removed for brevity
#!/usr/bin/env python3
# license removed for brevity
import math
import random
import rospy
from controller_manager_msgs.srv import SwitchController
from std_msgs.msg import Float64
import numpy as np
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge, CvBridgeError 
import cv2
from sensor_msgs.msg import JointState
from csv import writer
import matplotlib.pyplot as plt


class servo():

	def __init__ (self):
		self.image2_sub = rospy.Subscriber("/vbmbot/camera1/image_raw",Image,self.camera_callback) 
		self.bridge_object = CvBridge() 
		self.state_sub = rospy.Subscriber("/vbmbot/joint_states",JointState,self.jointstate_callback)
		self.cXr = 0
		self.cYr = 0
		self.cXb = 0
		self.cYb = 0
		self.cXg = 0
		self.cYg = 0
		self.cXp = 0
		self.cYp = 0
		self.s = 0
		self.s_star = 0
		self.j1 = 0
		self.j2 = 0
		self.cam_vel = 0
		self.flag = False
		self.red_sx_star = []
		self.red_sy_star = []
		self.blue_sx_star = []
		self.blue_sy_star = []
		self.green_sx_star = []
		self.green_sy_star = []
		self.purple_sx_star = []
		self.purple_sy_star = []
		 
		
	def talker(self):		
		pub_q1_pos = rospy.Publisher('/vbmbot/joint1_position_controller/command', Float64, queue_size=10)
		pub_q2_pos = rospy.Publisher('/vbmbot/joint2_position_controller/command', Float64, queue_size=10)
		# pub_q1_vel, pub_q2_vel - publish the joint velocity to the joints
		pub_q1_vel = rospy.Publisher('/vbmbot/joint1_velocity_controller/command', Float64, queue_size=10)
		pub_q2_vel = rospy.Publisher('/vbmbot/joint2_velocity_controller/command', Float64, queue_size=10)
		#cb = camera_callback()
		#initialize the node
		#at node initialization, the position controller is active 
		# so only position(joint angles) commands can be given to joints
		#rospy.init_node('joint_manip_talker', anonymous=True)
		rate = rospy.Rate(1)# meaning 1 message published in 1 sec
		rospy.sleep(5) 
		random.seed()
		q1_pos = 0
		q2_pos = -0.25
		pub_q1_pos.publish(q1_pos)
		pub_q2_pos.publish(q2_pos)
		rospy.sleep(5)
		self.s = np.array([self.cXr, self.cYr, self.cXb, self.cYb, self.cXg, self.cYg, self.cXp, self.cYp])
		print("printing S",self.s)
		#target position given to move the joints away from home position 

#		s = np.empty(8,1)
#		s_star = np.empty(8,1)   

#		s.append(fc.cXr, fc.cYr, fc.cXb, fc.cYb, fc.cXg, fc.cXg, fc.cXp, fc.cYp)
		q1_pos = 0.4
		q2_pos = -0.9 
		pub_q1_pos.publish(q1_pos)
		pub_q2_pos.publish(q2_pos)
		rospy.sleep(5)
		self.s_star = np.array([self.cXr, self.cYr, self.cXb, self.cYb, self.cXg, self.cYg, self.cXp, self.cYp])
		print("Printing s_star",self.s_star)
		print ("before switching")
		self.switch_controller()
		print ("After switching")
#		s_star.append(fc.cXr, fc.cYr, fc.cXb, fc.cYb, fc.cXg, fc.cXg, fc.cXp, fc.cYp)
		#once the joints have moved from home position, 
		# the position controller is stopped and velocity controller is started. 
		# We use ros inbuilt switch_controller service for that.
		
#		q1_dot = self.cam_vel[0]
#		q2_dot = self.cam_vel[1]
#		pub_q1_vel.publish(q1_dot)
#		pub_q2_vel.publish(q2_dot)
#		rospy.sleep(5)
#		rospy.wait_for_service('/vbmbot/controller_manager/switch_controller')
#		try:
#			sc_service = rospy.ServiceProxy('/vbmbot/controller_manager/switch_controller', SwitchController)
#			start_controllers = ['joint1_velocity_controller','joint2_velocity_controller']
#			stop_controllers = ['joint1_position_controller','joint2_position_controller']
#			strictness = 2
#			start_asap = False
#			timeout = 0.0
#			res = sc_service(start_controllers,stop_controllers, strictness, start_asap,timeout)
                       
                       
#		except rospy.ServiceException as e:
#			print("Service Call Failed")	


		
#		while not rospy.is_shutdown():				
			# once the controller switch is successful the joints will 
			# receive random velocity from the velocity controller 
			# and keep moving until ROS is shutdown
#			q1_vel = random.uniform(-0.5, 0.5)
#			q2_vel = random.uniform(-0.5, 0.5)

#			pub_q1_vel.publish(q1_vel)
#			pub_q2_vel.publish(q2_vel)

#			rate.sleep()		
	 
		
	def jointstate_callback(self, data):
		self.j1 = data.position[0]
		self.j2 = data.position[1]
			
	def csv_data(self):
	
  
		# The data assigned to the list 
		

		
#		self.red_sx_star.append(self.s_star[0])
#		self.red_sy_star.append(self.s_star[1])
#		self.blue_sx_star.append(self.s_star[2])
#		self.blue_sy_star.append(self.s_star[3])
#		self.green_sx_star.append(self.s_star[4])
#		self.green_sy_star.append(self.s_star[5])
#		self.purple_sx_star.append(self.s_star[6])
#		self.purple_sy_star.append(self.s_star[7])
		
		#plot the data and line 

#		plt.figure(figsize=(10, 8)) 
#		plt.figure(figsize=(10, 8))
#		plt.figure(figsize=(10, 8))
		#plt.figure(figsize=(10, 8))

#		plt.axes(xlim=(0, 300), ylim=(0, 300), autoscale_on=False) 
#		plt.axes(xlim=(0, 300), ylim=(0, 300), autoscale_on=False)
#		plt.axes(xlim=(0, 300), ylim=(0, 300), autoscale_on=False)
		#plt.axes(xlim=(0, 300), ylim=(0, 300), autoscale_on=False) 

#		plt.plot(red_sx_star, red_sy_star) 
#		plt.plot(blue_sx_star, blue_sy_star)
#		plt.plot(green_sx_star, green_sy_star)
		#plt.plot(red_sx_star, red_sy_star, blue_sx_star, blue_sy_star,green_sx_star, green_sy_star, purple_sx_star, purple_sy_star)

		#plt.xlabel("x", fontsize=14) 

		#plt.ylabel("y", fontsize=14) 
		
#		plt.xlabel("blue_x", fontsize=14) 

#		plt.ylabel("blue_y", fontsize=14)
		
#		plt.xlabel("green_x", fontsize=14) 

#		plt.ylabel("green_y", fontsize=14)
		
#		plt.xlabel("purple_x", fontsize=14) 

#		plt.ylabel("purple_y", fontsize=14)

		#plt.title("Graph of cordinates", fontsize=16) 
#		plt.show()
#		plt.show()
#		plt.show()
		#plt.show()
		list_data_s_star = []
		list_data_s_star.append(self.s_star[0])
		list_data_s_star.append(self.s_star[1])
		list_data_s_star.append(self.s_star[2])
		list_data_s_star.append(self.s_star[3])
		list_data_s_star.append(self.s_star[4])
		list_data_s_star.append(self.s_star[5])
		list_data_s_star.append(self.s_star[6])
		list_data_s_star.append(self.s_star[7])
		print(list_data_s_star)
		file_n = "s_star-final_450X.csv"
  
		# Pre-requisite - The CSV file should be manually closed before running this code.

		# First, open the old CSV file in append mode, hence mentioned as 'a'
		# Then, for the CSV file, create a file object
		with open(file_n, 'a') as f_object:  
    			# Pass the CSV  file object to the writer() function
			writer_object = writer(f_object)
    			# Result - a writer object
    			# Pass the data in the list as an argument into the writerow() function
    			#writer_object.writerow(list_data_s)
			writer_object.writerow(list_data_s_star)
#    			print("csv file is printeduuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")
    			  
    			# Close the file object
			f_object.close()

	
	def visual_survoing(self):
		pub_q1_vel = rospy.Publisher('/vbmbot/joint1_velocity_controller/command', Float64, queue_size=10)
		pub_q2_vel = rospy.Publisher('/vbmbot/joint2_velocity_controller/command', Float64, queue_size=10)
		

		while self.flag:
			
                
#		s = np.array([287,71,14,13,229,176,125,114])
#		s_star = np.array([208,58,87,59,209,179,88,177])
                
#		s = np.empty(8,1)
#		s_star = np.empty(8,1)
                
#		s = np.array[(cXr, cYr, cXb, cYb, cXg, cXg, cXp, cYp)]
#		s_star = np.array[(cXr, cYr, cXb, cYb, cXg, cXg, cXp, cYp)]
                
			error = np.subtract(self.s, self.s_star)
			print("error",error)
                
			f = 1
			l = 0.0001
			z = 1
                
			Le = np.array([[-f/z, 0], [0, -f/z], [-f/z, 0], [0, -f/z], [-f/z, 0], [0, -f/z], [-f/z, 0], [0, -f/z]])
                
			lambda1 = -np.array([[l, 0], [0, l]])
                
			vc_inter = np.matmul(lambda1, np.linalg.pinv(Le))
                
			vc =  np.matmul(vc_inter, error) 
                
#			print ("its vc",vc)
		
		
		
			l1 = 0.5
			l2 = 0.5
		
			jacob_inv = np.linalg.pinv(np.array([[-l1*math.sin(self.j1)-l2*math.sin(self.j1+self.j2), l2*math.sin(self.j1+self.j2)], [l1*math.cos(self.j1)+l2*math.cos(self.j1+self.j2), l2*math.cos(self.j1+self.j2)]]))
#			print ('its jacobian', jacob_inv)
		
			self.cam_vel = np.matmul(jacob_inv, vc)
#			print ("Got cam_vel", self.cam_vel)
		
#			print(self.cam_vel[0])
#			print(self.cam_vel[1])
		
			q1_dot = self.cam_vel[0]
			q2_dot = self.cam_vel[1]
			pub_q1_vel.publish(q2_dot)
			pub_q2_vel.publish(q1_dot)
			
			self.csv_data()

			
			if self.flag == False:
				break
		
#			start_controllers = ['joint1_velocity_controller','joint2_velocity_controller']
#			stop_controllers = ['joint1_position_controller','joint2_position_controller']
#			strictness = 2
#			start_asap = False
#			timeout = 0.0
#			res = sc_service(start_controllers,stop_controllers, strictness, start_asap,timeout)
                       
                       

	def switch_controller(self):
			sc_service = rospy.ServiceProxy('/vbmbot/controller_manager/switch_controller', SwitchController)
			start_controllers = ['joint1_velocity_controller','joint2_velocity_controller']
			stop_controllers = ['joint1_position_controller','joint2_position_controller']
			strictness = 2
			start_asap = False
			timeout = 0.0
			res = sc_service(start_controllers, stop_controllers, strictness, start_asap, timeout)
			self.flag = True


	def camera_callback(self,data): 
		try: 
			cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8") 
		except CvBridgeError as e: 
			print(e) 
#		image_path = '/home/rbe450X-ros/src/gazebo_ros_demos/platform/Scripts/rosraw_image_object.png' 
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

		self.cXr = int(m["m10"] / m["m00"])
		self.cYr = int(m["m01"] / m["m00"])
        
		self.cXb = int(mb["m10"] / mb["m00"])
		self.cYb = int(mb["m01"] / mb["m00"])
	        
		self.cXg = int(mg["m10"] / mg["m00"])
		self.cYg = int(mg["m01"] / mg["m00"])
	        
		self.cXp = int(mp["m10"] / mp["m00"])
		self.cYp = int(mp["m01"] / mp["m00"])
		
		self.s_star = np.array([self.cXr, self.cYr, self.cXb, self.cYb, self.cXg, self.cYg, self.cXp, self.cYp])
	# put text and highlight the center

		cv2.circle(image2, (self.cXr, self.cYr), 3, (255, 255, 255), -1)
		cv2.circle(image2, (self.cXb, self.cYb), 3, (255, 255, 255), -1)
		cv2.circle(image2, (self.cXg, self.cYg), 3, (255, 255, 255), -1)
		cv2.circle(image2, (self.cXp, self.cYp), 3, (255, 255, 255), -1)

		cv2.putText(image2, "centroid", (self.cXr - 25, self.cYr - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		cv2.putText(image2, "centroid", (self.cXb - 25, self.cYb - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		cv2.putText(image2, "centroid", (self.cXg - 25, self.cYg - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		cv2.putText(image2, "centroid", (self.cXp - 25, self.cYp - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		cv2.imshow('Orignal',image2)

        
#		print(self.cXr, self.cYr) 
#		print(self.cXb, self.cYb)
#		print(self.cXg, self.cYg)
#		print(self.cXp, self.cYp)
 

		cv2.waitKey(1) 
   

def main():
	servo_object = servo()  
	rospy.init_node('center_detection_node', anonymous=True) 
	try: 
		#rospy.spin()
		servo_object.talker()
#	servo_object.csv_data()
		while servo_object.flag:
			servo_object.visual_survoing()
		#rospy.sleep(30)
	except KeyboardInterrupt: 
#		plt.figure(figsize=(10, 8))
#		plt.axes(xlim=(0, 300), ylim=(0, 300), autoscale_on=False) 
#		plt.plot(servo_object.red_sx_star, servo_object.red_sy_star,"-r", servo_object.blue_sx_star, servo_object.blue_sy_star,"-b",servo_object.green_sx_star, servo_object.green_sy_star, "-g",servo_object.purple_sx_star, servo_object.purple_sy_star,"-m")
#		plt.show()
		print("Shutting down") 
	cv2.destroyAllWindows() 
#	rospy.spin()
	      

if __name__ == '__main__':	
	main()
