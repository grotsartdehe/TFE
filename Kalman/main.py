#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:23:17 2021

@author: kdesousa
"""
import numpy as np
import math
from kalman_tools import track, detection, kalman_params, frame
#from kalman_graphics import kalman_draw
import os
import time
import datetime
import cv2
import pickle
from meta_params import *

params = kalman_params()
"""
def IQ_imbalance(fr):
    
    non utilisé car pas de delai entre flux video et radar
"""
	
def check_classe(d_c, t_c):
"""
malus = dico[t_c][d_c]
"""
        high = 0.7
    	middle = 0.6
    	low = 0.05
    
    	if(t_c == "init"):
    		return 10.0
    
    dic_car = {"car":0.0, "person":high, "truck":high, "suv":low, "van":low, "bicycle":high, "motorbike":high, "bus":high, "scooter":high, "ucl":middle, "trailer":high, "dog":high, "petit":middle, "moyen":0.0, "grand":middle}
    malus = dico[t_c][d_c]	
        return malus


def predict(tracks):

	s = 'radar'
	F, H, Q, R = params.get_matrices(s)

	for tr in tracks:

		X_pred = np.dot(F,tr.X)		# x_{k+1} = F x_{k} + q
		P_pred = np.dot(F, tr.P)	# P_{k+1} = F P_{k} F^T + Q
		P_pred = np.dot(P_pred, F.T)
		P_pred = np.add(P_pred, Q)

		tr.predict(X_pred, P_pred)

	return tracks
def associate(tracks, detections):

	dt, gate, lamb, max_invisible = params.get_params()

	try:
		F, H, Q, R = params.get_matrices(detections[0].get_sensor()) # toutes les detections d'une frames sont du même capteur

	except TypeError:

		return (tracks, detections, [], [])
		
	distances = [(0,0,0.0) for x in range(0,(len(tracks)*len(detections)))]
	i = 0
	k = 0
	for tr in tracks:
		j = 0

		S = np.dot(H,tr.P)		# S = H P_{k+1} H^T + R
		S = np.dot(S,H.T)
		S = np.add(S,R)

		K = np.dot(tr.P,H.T)		# K = P_{k+1} H^T S^-1
		try:
			K = np.dot(K,np.linalg.inv(S))
		except np.linalg.LinAlgError as e:
			print("K ", K)
			print("S ", S)
			print('\n\n')

		tr.set_KS(K, S)

		for det in detections:

			v = det.get_X() - np.dot(H, tr.X)	# v = z - H x
			d = np.dot(v.T,np.linalg.inv(tr.S))	# d = v^T S^-1 v
			d = np.dot(d,v)

			d = max(0.0,d)

			d += check_classe(det.get_class(), tr.classe)

			#if not tr.multi:# and len(detections) == 1:
			#	d -= 0.8
			#	d = max(0.0,d)

			distances[k] = (i, j, d)
			k += 1

			j += 1
		i += 1

	distances.sort(key=lambda tup: tup[2])

	m = []
	n = []

	for elem in distances:
		if(elem[2] < gate):
			m.append(elem)
		else:
			#print(elem[2])
			#print(tracks[elem[0]].classe)
			#print(detections[elem[1]].c0)
			n.append(elem)
			
	l1 = []
	l2 = []
	associated = []
	for e in m:
		if not e[1] in l1 and (not e[0] in l2 or not associate_det_one):
			l1.append(e[1])
			l2.append(e[0])
			associated.append(e)

	new = []
	for e in n:
		if not e[1] in l1:
			l1.append(e[1])
			new.append(e)

	return (tracks, detections, associated, new)

def update(tracks, detections, associated, new):

	dt, gate, lamb, max_invisible = params.get_params()

	try:
		F, H, Q, R = params.get_matrices(detections[0].get_sensor()) # toutes les detections d'une frames sont du même capteur

	except TypeError:

		return  (tracks, new, detections)

	sum_p = np.array(len(tracks)*[0.0])
	for e in associated: 
		sum_p[e[0]] += math.exp(-0.5*e[2])	# sum_{l=1}^{m'} exp(0.5*(d{i,l})^2)

	prob = []
	classes = [] 
	for tr in tracks:
		classes.append((0.0, tr.classe))
		
	prob_0 = [(x,0,1.0) for x in range(0,len(tracks))] 
	for e in associated:
		sum_i = sum_p[e[0]]
		classes[e[0]] = get_max_class(classes[e[0]], detections[e[1]])
		if(sum_i != 0.0):
			p_d = detections[e[1]].get_confidence()
			S = tracks[e[0]].S
			k = (lamb)*((1-p_d)/p_d)*math.sqrt(np.linalg.det(S))
			p = math.exp(-0.5*e[2])/(k+sum_i)
			prob.append((e[0],e[1],p))

			p_0 = k/(k+sum_i)
			prob_0[e[0]] = ((e[0], -1, p_0))
		else:
			prob_0[e[0]] = ((e[0], -1, 1.0))


	s_pv = np.array(len(tracks)*[6*[0.0]])
	for p in prob:
		b = (detections[p[1]].get_X() - np.dot(H, tracks[p[0]].X))
		s_pv[p[0]] = np.add(s_pv[p[0]], p[2]*b) 	# sum_{}^{} p_{i,j}v_{i,j}

	s_pvt = len(tracks)*[6*[0.0]]
	for p in prob:
		a = np.dot((detections[p[1]].get_X() - np.dot(H, tracks[p[0]].X)), (detections[p[1]].get_X() - np.dot(H, tracks[p[0]].X)).T)
		s_pvt[p[0]] = np.add(s_pvt[p[0]], p[2]*a) 	# sum_{}^{} p_{i,j}v_{i,j}v_{i,j}^T

	U = np.array(len(tracks)*[0.0])
	i = 0
	for p0 in prob_0:
		u = np.dot(s_pv[i],s_pv[i].T) 	# U_i = (1 - p_{i,0})*(sum*sum^T)
		u = (1-p0[2])*u
		U[i] = u

		i += 1

	i = 0
	for tr in tracks:
		a = np.dot(tr.K,tr.S)
		a = np.dot(a,(tr.K).T)
		a = (1 - prob_0[i][2])*a
		
		b = np.dot(tr.K,(s_pvt[i] - U[i])*(tr.K).T)

		P = tr.P - a + b

		X = tr.X + np.dot(tr.K,s_pv[i])
	
		found = False
		if(sum_p[i] != 0.0):
			found = True

		tr.update(X, P, classes[i][1], found)

		i += 1

	for elem in associated:
		tracks[elem[0]].update_sensor(detections[elem[1]].get_sensor())


	return (tracks, new, detections)
def housekeep(tracks, detections, new = 0):

	# Create new tracks with unassociated detections

	if not new == 0 and len(new) > 0:

		#new_thresh = min_distance_new
		new_tracks = []
		for n in new:

			det = detections[n[1]]

			new_thresh = new_thresh_classes[det.get_class()]

			if(n[2] > new_thresh):

				new_t = track(0)
				F, H, Q, R = params.get_matrices(det.get_sensor())
				R_0 = params.get_init_matrice()
				new_t.create(det.get_X(), det.get_class(), R_0, det.get_sensor())
				new_tracks.append(new_t)

		# Keep only good tracks
		
		tracks.extend(new_tracks)

	if(len(tracks) == 0 and len(detections) != 0):

		new_tracks = []

		for det in detections:
			if det is not None:
				new_t = track(0)
				try:
					F, H, Q, R = params.get_matrices(det.get_sensor())
				except:	
					print(det.get_sensor())
					F, H, Q, R = params.get_matrices(det.get_sensor())
				new_t.create(det.get_X(), det.get_class(), R, det.get_sensor())
				new_tracks.append(new_t)

		tracks.extend(new_tracks)

	dt, gate, lamb, max_invisible = params.get_params()

	keep_tracks = []
	for tr in tracks:
	
		#thresh = obstruction_thresh

		obs = False

		for t in tracks:
			thresh = 0.8*math.degrees(math.atan(objects_obs_size[t.classe]/(2*t.X[2])))
			if(abs(tr.X[0] - t.X[0]) < thresh and (tr.X[2] > t.X[2]) and tr.min_det > min_det_obstruction):
				if(t.min_det > min_obs_front):
					tr.obstruction += 1
					obs = True

		if(obs == False):
			tr.obstruction = 0
	
			#else:
			#	tr.obstruction = 0

			#tr.obstruction = max(0, tr.obstruction)

		if(tr.invisible < max_invisible_obs):
			if((tr.invisible < max_invisible_no_obs) or (tr.obstruction > 0)):
				keep_tracks.append(tr)	

	return keep_tracks

def kalman_estimate(tracks, detections):

	# Si il n'y a aucune piste, passer l'assoc et l'update 
	if(len(tracks) == 0):
		return housekeep(tracks, detections)
	
	# Prédiction de l'état suivant	
	tracks = predict(tracks)


	# Si pas de détection, passer l'assoc
	if(len(detections) == 0):
		for tr in tracks:
			tr.invisible += 1
		return housekeep(tracks, detections)

	# Association 
	tracks, detections, associated, new = associate(tracks, detections)

	# Update
	tracks, new, detections = update(tracks, detections, associated, new)

	# Housekeep
	filtered_tracks = housekeep(tracks, detections, new)

	return filtered_tracks

def kalman_save(tracks, file):
	pass

def load_data(radar_folder, camera_folder, radar_times, camera_times, delta_t):
	"""
	f = open(radar_times, 'rb')
	rad_times = pickle.load(f)
	f.close()
    """
	f = open(camera_times, 'rb')
	cam_times = pickle.load(f)
	f.close()

	r_n = len(os.listdir(radar_folder))

	print("Number of radar files ", r_n)

	radar_frames = []

	for i in range(0, r_n):

		fr = frame("radar")

		#ret, image = cap_rad.read()

		fr.set_time("rad", float(str(rad_times[i])[5:]))

		name = radar_folder + "/predictions_" + str(i+1) + ".txt"

		pos =  False

		f = open(name,'r')
		for l in f:
			elems = l.split()
			if(len(elems)>1 and elems[0] != 'X'):
				fr.pos = True
				det = detection(elems[0], elems[1], elems[2], elems[3], elems[4], elems[5], elems[6], elems[7], elems[8], elems[9], elems[10], elems[11], elems[12])
				#det = detection(x, y, w, h, p0, c0, p1, c1, wi, hi, t, f, s)
				fr.add_detection(det)			

		radar_frames.append(fr)


	c_n = len(os.listdir(camera_folder))

	print("Number of video files ", c_n)

	cam_frames = []

	z = 0
	for i in range(0, c_n):

		fr = frame("cam")

		#ret, image = cap_vid.read()

		fr.set_time("cam", float(str(cam_times[z])[5:]))

		z += 2

		name = camera_folder + "/predictions_" + str(i+1) + ".txt"

		pos = False

		f = open(name,'r')
		for l in f:
			elems = l.split()
			if(len(elems)>1 and elems[0] != 'X'):
				fr.pos = True
				det = detection(elems[0], elems[1], elems[2], elems[3], elems[4], elems[5], elems[6], elems[7], elems[8], elems[9], elems[10], elems[11], elems[12])
				#det = detection(x, y, w, h, p0, c0, p1, c1, wi, hi, t, f, s)
				fr.add_detection(det)
			

		cam_frames.append(fr)

		

	frames= []

	current_t = min(radar_frames[0].t, cam_frames[0].t) - delta_t

	i = 0
	j = 0
	l = 0
	i_max = len(radar_frames)-1
	j_max = len(cam_frames)-1

	while (i<(i_max) or j<(j_max)):

		a = radar_frames[i]
		b = cam_frames[j]

		if(a.t < b.t and a.t < (current_t + delta_t) and i<i_max):
			#print("Radar ", a.t)
			current = a
			i += 1
			l = 0

		elif(j == j_max and a.t < (current_t + delta_t)):
			current = a
			i += 1
			l = 0
	
		elif(b.t < (current_t + delta_t) and i == i_max):
			#print("Video ", b.t)
			current = b
			j += 1
			l = 0
		elif(b.t < (current_t + delta_t) and j<j_max):
			current = b
			j += 1
			l = 0
		else:
			#print("Empty ", delta_t)	
			current = frame("empty")
			l += 1

			#print(i,j)
		if(l <= 10):
			frames.append(current)

		current_t += delta_t

	frames.append(frame("end"))

	print("Radar frames with detection ", i+1)
	print("Camera frames with detection ", j+1)
	print("Number of frames ", len(frames))

	filtered_frames = []
	for fr in frames:
		filtered_frames.append(IQ_imbalance(fr))
		

	return filtered_frames


def kalman(radar_det, camera_det):

	dt, gate, lamb, max_invisible = params.get_params()

	frames = load_data(radar_det, camera_det, dt)

	tracks = []

	i = 0
	for fr in frames:
		tracks = kalman_estimate(tracks, fr.get_detections())
		#if len(tracks) > 0:
		#	print(tracks[0].X)
		i += 1
		#kalman_draw(tracks)
		#kalman_save(tracks, file)

def get_max_class(cl, detection):

	if(cl[0] > detection.get_confidence()):
		return cl

	else:
		return (detection.get_confidence(), detection.get_class())

def kalman_queue(q, radar_det, camera_det, cam_vid, rad_vid, rad_times, cam_times):

	cap_vid = cv2.VideoCapture(cam_vid)
	cap_rad = cv2.VideoCapture(rad_vid)

	dt, gate, lamb, max_invisible = params.get_params()

	frames = load_data(radar_det, camera_det, rad_times, cam_times, dt)

	tracks = []

	
	vid_frame = 255*np.ones((720,1280,3), np.uint8)

	rad_frame = 255*np.ones((512,512,3), np.uint8)

	i = 0
	i_vid = 0
	i_cam = 0

	for fr in frames:

		cam = False
		rad = False

		t_0 = time.time()

		tracks = kalman_estimate(tracks, fr.get_detections())

		if(fr.get_sensor() == 'cam'):
			cam = True
			ret, vid_frame = cap_vid.read()
		elif(fr.get_sensor() == 'radar'):
			#for i in range(0,fr.empty):
			rad = True
			ret, rad_frame = cap_rad.read()
		elif(fr.get_sensor() == 'end'):
			q.put(('end', 0, 0, 0, 0, 0), True, None)

		#if q.full():
		#	q.get()
		q.put((i*dt, cam, rad, vid_frame, rad_frame, tracks), True, None)

		step = str(datetime.timedelta(seconds=i*dt))
		print(step, end='\r')
		i += 1

		t_h = dt - (time.time() - t_0)
		t_h = (abs(t_h) + t_h)/2.0
		time.sleep(t_h)


if __name__ == '__main__':

	radar_folder = "/home/kdesousa/TFE/Kalman/data/radar"
	camera_folder = "/home/kdesousa/TFE/Kalman/data/cam"
	kalman(radar_folder,camera_folder)
	
