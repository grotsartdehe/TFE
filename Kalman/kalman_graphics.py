from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import numpy as np
import math
from kalman_tools import track, kalman_params
import time
#from main2 import kalman_queue
#import cv2
import copy
import datetime
import os
from meta_params import *

def spher2cart(x):

	phi = math.radians(x[0]) 	# azimuth
	theta = math.radians(90 - x[1]) # inclination
	r = x[2]

	X = r*math.sin(theta)*math.sin(phi)
	Y = r*math.sin(theta)*math.cos(phi)
	Z = r*math.cos(theta)

	v_x = x[5]*math.sin(theta)*math.sin(phi) + r*(x[3]*math.sin(theta)*math.cos(phi) - x[4]*math.cos(theta)*math.sin(phi))
	v_y = x[5]*math.sin(theta)*math.cos(phi) + r*(x[3]*math.sin(theta)*math.sin(phi) - x[4]*math.cos(theta)*math.cos(phi))
	v_z = x[5]*math.cos(theta) - r*x[4]*math.sin(theta)

	return [X, Y, Z, v_x, v_y, v_z]


class current_state():

	def __init__(self):
		self.colors = {"car": (0,128,0), "person":(255,0,0), "truck":(0,0,255), "suv":(255,0,255), "van":(255,255,0), "bicycle":(128,128,0), "motorbike":(128,0,0), "bus":(0,0,128), "scooter":(128,0,128), "ucl":(0,255,255), "trailer":(128,128,128), "dog":(0,0,0), "petit":(255,255,255), "moyen":(255,255,255), "grand":(255,255,255)}
		self.X = 0.0
		self.Y = 0.0
		self.color = (255,255,255)
	
	def update(self, track):
		self.X = spher2cart(track.X)[0]
		self.Y = spher2cart(track.X)[1]
		self.color = self.colors[track.classe]

def f(q):

	st = current_state()

	r = np.array(
	    [[1.0,0.0,0.0,0.0,0.0,0.0],\
	    [0.0,1.0,0.0,0.0,0.0,0.0],\
	    [0.0,0.0,4.0,0.0,0.0,0.0],\
	    [0.0,0.0,0.0,0.0,0.0,0.0],\
	    [0.0,0.0,0.0,0.0,0.0,0.0],\
	    [0.0,0.0,0.0,0.0,0.0,0.0]])

	tr0 = track(0)
	tr0.create([0.0, 0.0, 5.0, 0.0,0.0,0.0],"person",r)

	tr1 = track(1)
	tr1.create([0.0, 0.0, 10.0, 0.0,0.0,0.0],"person",r)


	tr2 = track(2)
	tr2.create([0.0, 0.0, 15.0, 0.0,0.0,0.0],"person",r)


	tr3 = track(3)
	tr3.create([0.0, 0.0, 20.0, 0.0,0.0,0.0],"person",r)


	tr4 = track(4)
	tr4.create([0.0, 0.0, 25.0, 0.0,0.0,0.0],"person",r)

	tracks = [tr0, tr1, tr2, tr3, tr4]

	while True:

		for tr in tracks:

			st.update(tr)
			q.put([st.X, st.Y])
			time.sleep(0.1)


def sort_tr(tracks):

	classes = ["car", "person", "truck", "suv", "van", "bicycle", "motorbike", "bus", "scooter", "ucl", "trailer", "dog", "petit", "moyen", "grand"]

	out = []
	for cl in classes:
		for tr in tracks:
			if(tr.classe == cl):
				out.append(tr)

	return out

def get_min_distance(tracks, ti):

	if(len(tracks) < 2):
		return []

	veh = ["car", "truck", "suv", "van", "bus", "ucl", "trailer"]

	ped = ["person", "bicycle", "scooter", "dog"]


	distances = []
		
	for tr in tracks:
		for t in tracks:
			x1 = tr.X
			x2 = t.X
			s = (x1[0] - x2[0])*(x1[0] - x2[0]) + (x1[1] - x2[1])*(x1[1] - x2[1])# + (x1[2] - x2[2])*(x1[2] - x2[2])
			d = math.sqrt(s)
			if(tr.classe in veh and t.classe in ped and d < danger_dist):
				distances.append((tr.classe, t.classe, tr.min_det, t.min_det, round(d,3), ti))
	

	return distances


colors = {"car": (0,128,0), "person":(255,0,0), "truck":(0,0,255), "suv":(255,0,255), "van":(255,255,0), "bicycle":(128,128,0), "motorbike":(128,0,0), "bus":(0,0,128), "scooter":(128,0,128), "ucl":(0,255,255), "trailer":(128,128,128), "dog":(0,0,0), "petit":(255,255,255), "moyen":(255,255,255), "grand":(255,255,255)}

if __name__ == '__main__':

	names = sorted(os.listdir(video_folders))

	folder_number = 0

	for folder in names:

		folder_number += 1

		text = "[INFO] treating folder " + folder + " (" + str(folder_number) + "/" + str(len(names)) + ")"
		print(text)

		sample = folder
		radar_folder = radar_folders + "/" + sample + "/kalman"
		camera_folder =  video_folders + "/" + sample + "/kalman"

		rad_files = os.listdir(radar_folders + "/" + sample)
		for file in rad_files:
			if file.endswith("annot.mp4"):
				rad_file = radar_folders + "/" + sample + "/" + file
			elif file.endswith("times.pickle"):
				rad_times = radar_folders + "/" + sample + "/" + file

		vid_files = os.listdir(video_folders + "/" + sample)

		for file in vid_files:
			if file.endswith("annot.mp4"):
				cam_file = video_folders + "/" + sample + "/" + file
			elif file.endswith("times.pickle"):
				cam_times = video_folders + "/" + sample + "/" + file

		save = True

		t_start = 0.0
		t_stop = 500.0
		video_fps = 50.0

		if(save):
			out_name = save_folder + "/" + sample + ".mp4"
			resol_w = 1000
			resol_h = 680
			output_vid = cv2.VideoWriter(out_name,0x7634706d, video_fps, (resol_w,resol_h))

		q = Queue(maxsize=10)
		p = Process(target=kalman_queue, args=(q,radar_folder,camera_folder,cam_file,rad_file,rad_times, cam_times,))
		p.start()
		
		X = [0.0]
		Y = [0.0]
		C = [(255/255.0,255/255.0,255/255.0)]
		D = [2]
		t = 0.0
		t_rec = 0.0
		cam = False
		rad = False

		frame = cv2.imread("background_4.0.png",cv2.IMREAD_COLOR)

		vid_frame = 255*np.ones((720,1280,3), np.uint8)
		rad_frame = 255*np.ones((512,512,3), np.uint8)
		camera = 255*np.ones((360,640,3), np.uint8)
		radar = 255*np.ones((360,360,3), np.uint8)
		kalm = 255*np.ones((320,360,3), np.uint8)

		params = kalman_params()

		dt, gate, lamb, max_invisible = params.get_params()

		q1 = Queue(maxsize=25)

		X = []
		Y = []
		C = []
		D = []

		distances_cr = []

		while True:

			new = False

			back = copy.copy(frame)
			#back = frame

			X = []
			Y = []
			C = []
			D = []

			#if not q.empty():
			new = True
			t, cam, rad, vid_frame, rad_frame, tracks = q.get(True, None)
			if(t == 'end'):
				break
			M = cv2.getRotationMatrix2D((256,256), 90, 1.0)
			rad_frame = cv2.warpAffine(rad_frame, M, (512, 512)) 
			for tr in tracks:
				if((tr.min_det > min_det_affichage or (tr.multi and display_multi)) and tr.disper < max_disper):#200000):
					vec = spher2cart(tr.X)
					if(vec[1] > 23.2):
						X.append(vec[0])
						#X.append(0.0)
						Y.append(vec[1])
						C.append(colors[tr.classe])
						#D.append(math.exp(min(tr.disper,200)))
						D.append(tr.disper/1.0 + 2.0)

			if(q1.full()):
				q1.get()
			q1.put((X,Y,C))

			for x, y, c, d in zip(X, Y, C, D):
				x = (513.0/48.0)*(x + 24) + 60
				y = (558.0/85.0)*(85 - y) + 14
				cv2.circle(back, (int(x),int(y)), min(int(d),10), color=c, thickness=2)
				#cv2.circle(back, (int(x),int(y)), 2, color=c, thickness=2)

			i = q1.qsize()
			for k in range(0,i):
				X, Y, C = q1.get()
				if q1.full() and k == 0:
					pass
				else:
					q1.put((X, Y, C ))

				for x, y, c in zip(X, Y, C):
					x = (513.0/48.0)*(x + 24) + 60
					y = (558.0/85.0)*(85 - y) + 14
					cv2.circle(back, (int(x),int(y)), 1, color=c, thickness=1)
			
			#kalm = cv2.resize(back[200:430,:200:430], (360,320), interpolation = cv2.INTER_AREA) 
			
			

			mask = np.zeros((680,1000,3), np.uint8)

			if(cam):
				camera = cv2.resize(vid_frame,None,fx=0.5,fy=0.5)
			mask[0:360,0:640] = camera
			if(rad):
				radar = cv2.resize(rad_frame, (360,360), interpolation = cv2.INTER_AREA)
			mask[0:360,640:1000] = radar
			if(True):
				kalm = cv2.resize(back[200:445,80:-80], (560,320), interpolation = cv2.INTER_AREA) 
			mask[360:680,440:1000] = kalm

			if(True):

				blank = 255*np.ones((320,440,3), np.uint8)

				font                   = cv2.FONT_ITALIC #cv2.FONT_HERSHEY_SIMPLEX
				bottomLeftCornerOfText = (520,300)
				fontScale              = 0.35
				fontColor              = (0,0,0)
				lineType               = 1

				i = 0

				tracks = sort_tr(tracks)

				min_distances = get_min_distance(tracks, str(datetime.timedelta(seconds=t))[2:10])

				for tr in tracks:
					#vec_x = spher2cart(tr.X)
					vec_x = tr.X
					v = 3.6*math.sqrt(vec_x[5]*vec_x[5])# + vec_x[4]*vec_x[4] + vec_x[5]*vec_x[5])
					if(tr.min_det > min_det_write or (tr.multi and display_multi)):
						C = colors[tr.classe]
						cv2.putText(blank,"-", (5,20*(1+i)), font, fontScale, C, lineType)
						cv2.putText(blank,tr.classe, (25,20*(1+i)), font, fontScale, C, lineType)
						#cv2.putText(blank,"epsilon : " + str(round(vec_x[3],1)), (120,20*(1+i)), font, fontScale, C, lineType)
						cv2.putText(blank,str(round(tr.height,1)), (120,20*(1+i)), font, fontScale, C, lineType)
						#cv2.putText(blank,"vitesse radiale : " +str(round(v*3.6,2)) + "  km/h", (100,20*(1+i)), font, fontScale, C, lineType)
						#cv2.putText(blank,str(round(tr.disper,2)), (310,20*(1+i)), font, fontScale, C, lineType)
						
						if(tr.obstruction):
							C = colors[tr.classe]
							cv2.putText(blank,"obstruction", (220,20*(1+i)), font, fontScale, C, lineType)
						i += 1

				if(len(min_distances) > 0):
					distances_cr.extend(min_distances)
					#print(distances_cr)
				#	cv2.putText(blank, 'distance critique = ' + str(round(min_distance,2)), (35,200), font, fontScale, fontColor, lineType)

				pad = 355
				cv2.putText(blank, 'dt = ' + str(round(dt,2)), (pad,20), font, fontScale, fontColor, lineType)
				cv2.putText(blank, 'lambda = ' + str(int(lamb)), (pad,40), font, fontScale, fontColor, lineType)
				cv2.putText(blank, 'beta = ' + str(round(gate,2)), (pad,60), font, fontScale, fontColor, lineType)
				cv2.putText(blank, 'Imax = ' + str(max_invisible), (pad,80), font, fontScale, fontColor, lineType)
				cv2.putText(blank,str(datetime.timedelta(seconds=t))[2:10], (340,300), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, fontColor, lineType)
				cv2.putText(blank,"vitesse 0.5x", (260,300), font, fontScale, fontColor, lineType)
				mask[360:680,0:440] = blank


			if (save and t > t_start and t < t_stop and new):
				output_vid.write(mask)
				t_rec = t

			elif(save and t > t_stop):
				output_vid.release()

			if cv2.waitKey(10) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				p.terminate()
				break

		f = open(min_distances_file, 'a')
		for d in distances_cr:
			line = '{:<4} {:<15} {:<15} {:<4} {:<4} {:<6} {:<10} {:<3}'.format(sample, d[0], d[1], d[2], d[3], d[4], d[5],'\n').lstrip()
			f.write(line)

		f.write('\n')
		f.close()

		try:
			cv2.destroyAllWindows()
			output_vid.release()
			p.terminate()
		except:
			pass
	
