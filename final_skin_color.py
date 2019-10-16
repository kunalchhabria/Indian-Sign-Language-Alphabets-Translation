from imutils.video import VideoStream
from imutils.video import FPS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import collections
import argparse
import imutils
import time
import cv2
import sys
from fastai import *
from fastai.vision import *
total_rectangle = 9
hand_rect_one_x = None
hand_rect_one_y = None

hand_rect_two_x = None
hand_rect_two_y = None

ans='' 
alphabet_array=[]

def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame

roi = np.zeros([360, 10, 3], dtype=np.uint8)

def hand_range_manual(roi):
	h=[]
	s=[]
	v=[]
	hand_min=[]
	hand_max=[]
	for i in roi:
		for j in i:
			h.append(j[0])
			s.append(j[1])
			v.append(j[2])
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(h,s,v)
	plt.title('Skin HSV scatter plot')
	plt.show()
	nph=np.array(h)
	nps=np.array(s)
	npv=np.array(v)
	nph_mean=nph.mean()
	nph_std=nph.std()
	nps_mean=nps.mean()
	nps_std=nps.std()
	npv_mean=npv.mean()
	npv_std=npv.std()
	print('HSV stands for Hue Saturation Value.')
	
	print('suggestions= h_min:',max(0,nph_mean-3*nph_std),' and h_max:',min(255,nph_mean+3*nph_std))
	plot_skin(h,'h')
	print('Do you want to change h_min and h_max values? Enter y or n.')
	temp=input()
	if temp=='n':
		hand_min.append(max(0,nph_mean-3*nph_std))
		hand_max.append(min(255,nph_mean+3*nph_std))
	else:
		print('enter h_min and h_max:')
		hand_min.append(int(input()))
		hand_max.append(int(input()))

	print('suggestions= s_min:',max(0,nps_mean-3*nps_std),' and s_max:',min(255,nps_mean+3*nps_std))
	plot_skin(s,'s')
	print('Do you want to change s_min and s_max values? Enter y or n.')
	temp=input()
	if temp=='n':
		hand_min.append(max(0,nps_mean-3*nps_std))
		hand_max.append(min(255,nps_mean+3*nps_std))
	else:
		print('enter s_min and s_max:')
		hand_min.append(int(input()))
		hand_max.append(int(input()))

	print('suggestions= v_min:',max(0,npv_mean-3*npv_std),' and v_max:',min(255,npv_mean+3*npv_std))
	plot_skin(v,'v')
	print('Do you want to change v_min and v_max values? Enter y or n.')
	temp=input()
	if temp=='n':
		hand_min.append(max(0,npv_mean-3*npv_std))
		hand_max.append(min(255,npv_mean+3*npv_std))
	else:
		print('enter s_min and s_max:')
		hand_min.append(int(input()))
		hand_max.append(int(input()))
	return hand_min,hand_max


def hand_range(roi1):

	h=[]
	s=[]
	v=[]
	for i in roi1:
		for j in i:
			h.append(j[0])
			s.append(j[1])
			v.append(j[2])
	plot_skin(h,'h')
	plot_skin(s,'s')
	plot_skin(v,'v')
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.scatter(h,s,v)
	plt.show()
	#h_max,h_min=max(h),min(h)
	#s_max,s_min=max(s),min(s)
	#v_max,v_min=max(v),min(s)
	c = collections.Counter(h)
	c = sorted(c.items())
	temp=[]
	for i in c:
		if i[1]>20:
			temp.append(int(i[0]))
	h_max=max(temp)
	h_min=min(temp)
	c = collections.Counter(s)
	c = sorted(c.items())
	temp=[]
	for i in c:
		if i[1]>20:
			temp.append(int(i[0]))
	s_max=max(temp)
	s_min=min(temp)
	c = collections.Counter(v)
	c = sorted(c.items())
	temp=[]
	for i in c:
		if i[1]>20:
			temp.append(int(i[0]))
	v_max=max(temp)
	v_min=min(temp)

	hand_min = np.array([h_min, s_min, v_min], dtype=np.uint8)
	hand_max = np.array([h_max,s_max,v_max], dtype=np.uint8)
	print('hand min',hand_min,'hand max',hand_max)
	return hand_min,hand_max
	'''
	hand_hist = cv2.calcHist([roi1], [0, 1], None, [180, 256], [0, 180, 0, 256])
	return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)
	'''
def plot_skin(data,name):
	c = collections.Counter(data)
	c = sorted(c.items())
	pixel_value = [i[0] for i in c]
	freq = [i[1] for i in c]

	#print('counter',c)
	f, ax = plt.subplots()


	plt.bar(pixel_value, freq)
	temp="Distribution of "+ name.upper()+ " in sample data"
	plt.title(temp)
	plt.xlabel("pixel val")
	plt.ylabel("Frequency")
	plt.show()




def combine_skin_pixels(frame,start_pos):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    

    for i in range(total_rectangle):
        i_temp=i+start_pos
        roi[i_temp * 10: i_temp * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

def extractSkin(image,hand_min,hand_max):
  # Taking a copy of the image
  img =  image.copy()
  # Converting from BGR Colours Space to HSV
  img =  cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
  
  # Defining HSV Threadholds
  #lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
  #upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
  lower_threshold=np.array(hand_min)
  upper_threshold=np.array(hand_max)
  #lower_threshold = np.array([0, 50, 0], dtype=np.uint8)
  #upper_threshold = np.array([120, 150, 255], dtype=np.uint8)
  # Single Channel mask,denoting presence of colours in the about threshold
  skinMask = cv2.inRange(img,lower_threshold,upper_threshold)
  #cv2.imshow('inrange',skinMask)
  # Cleaning up mask using Gaussian Filter
  #skinMask = cv2.GaussianBlur(skinMask,(3,3),0)
  
  # Extracting skin from the threshold mask
  skin  =  cv2.bitwise_and(img,img,mask=skinMask)
  
  # Return the Skin image
  return cv2.cvtColor(skin,cv2.COLOR_HSV2BGR)


def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)

    thresh = cv2.merge((thresh, thresh, thresh))

    skin= cv2.bitwise_and(frame, thresh)
    #return skin
    return cv2.cvtColor(skin,cv2.COLOR_HSV2BGR)

print('Hello, I hope you are having a good day.')
print("[INFO] loading model...")

path='C:\\Users\\kunal_bem86lt\\Desktop\\capstone test'
learn=load_learner(path,fname='isl_final_skin_color.pkl')

print("[INFO] model loaded...")
 

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)


counter=0


is_skin_colur_detected=False
global hist_hand

is_skin_colur_detected=False
flag1=False
flag2=False
flag3=False
flag4=False

while vs.isOpened():
	nothing,frame = vs.read()
	key = cv2.waitKey(1)
	if key & 0xFF ==ord('z'):
		print('detected z')
		flag1=True
		combine_skin_pixels(frame,0)
		cv2.imshow('skin_range z',roi)

	if key & 0xFF ==ord('x'):
		print('detected x')
		flag2=True
		combine_skin_pixels(frame,9)
		cv2.imshow('skin_range x',roi)

	if key & 0xFF ==ord('c'):
		print('detected c')
		flag3=True
		combine_skin_pixels(frame,18)
		cv2.imshow('skin_range c',roi)

	if key & 0xFF ==ord('v'):
		print('detected v')
		flag4=True
		combine_skin_pixels(frame,27)
		cv2.imshow('skin_range v',roi)
		hand_min,hand_max=hand_range_manual(roi)
		is_skin_colur_detected=True



	if is_skin_colur_detected:


	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
		top, right,bottom, left = 120-30, 350-30, 360+30, 590+30 #(590,10  ,  350,225) 120 360
		cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
		
		roi = frame[top:bottom, right:left]
		#print(frame.shape)
		#print(roi.shape)

		frame_skinned=extractSkin(roi,hand_min,hand_max)
		#frame = imutils.resize(frame, width=400)
	
		# grab the frame dimensions and convert it to a blob
		
		font                   = cv2.FONT_HERSHEY_SIMPLEX
		bottomLeftCornerOfText = (10,50)
		fontScale              = 1
		fontColor              = (255,255,255)
		lineType               = 2
		
		cv2.imwrite('temp.jpg',frame_skinned)
		
		temp=open_image('temp.jpg')
		
		values=np.array(learn.predict(temp))[2]
		
		temp=np.argmax(values)
		
		confidence=values[temp]
		alphabet=temp+97
		alphabet_char=chr(alphabet)
	
		
		char_flag=False
		
		if len(alphabet_array)>30:
			del alphabet_array[0]
			alphabet_array.append(alphabet_char)
		else:
			alphabet_array.append(alphabet_char)

		sorted_alphabet_array=sorted(alphabet_array)
		most_freq_char=sorted_alphabet_array[-1]
		most_freq_char_count=sorted_alphabet_array.count(most_freq_char)

		if most_freq_char_count>10 and len(alphabet_array)>29:
			ans+=most_freq_char
			alphabet_array=[]

		alphabet=chr(alphabet) +'  '+str(confidence)
		alphabet1=ans
		


		cv2.putText(frame,alphabet1, 
		    bottomLeftCornerOfText, 
		    font, 
		    fontScale,
		    fontColor,
		    lineType)
		
		cv2.imshow("Frame", frame)

		cv2.putText(frame_skinned,alphabet, 
		    bottomLeftCornerOfText, 
		    font, 
		    fontScale,
		    fontColor,
		    lineType)

		cv2.imshow("frame_skinned",frame_skinned)
		#cv2.imshow('Frame1',frame)
		counter+=1
			
		#key = cv2.waitKey(1) & 0xFF
	 
		# if the `q` key was pressed, break from the loop
		if key & 0xFF == ord("q"):
			#print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
			#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
			 
			# do a bit of cleanup
			cv2.destroyAllWindows()
			sys.exit(0)
	 
		# update the FPS counter
	


	else:
		frame=draw_rect(frame)
		cv2.imshow('detect skin',frame)

	if key & 0xFF ==ord('q'):
		#print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
		#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
		 
		# do a bit of cleanup
		cv2.destroyAllWindows()
		vs.release()
		sys.exit(0)