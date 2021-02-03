#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:05:38 2021

@author: kdesousa
"""
# importer d'Alexis

import math
import numpy as np


files_id = '49'

radar_folders = "/home/alexis/Bureau/gen_results_yolo/dataset_test/radar/compute_res"
video_folders = "/home/alexis/Bureau/gen_results_yolo/dataset_test/results_video"

save_folder = "/home/alexis/Bureau/kalman_results"

min_distances_file = "/home/alexis/Bureau/kalman_results/distances.txt"


danger_dist = 3.0

# display 

min_det_affichage = 5
max_disper = 1000000
display_multi = True

# write

min_det_write = 0

# class malus

high_malus = 2.8
middle_malus = 1.5
low_malus = 0.5

# I-Q imbalance 

I_Q_speed_thesh = 10 # 3.1 km/h
I_Q_dist_thresh = 15 # 2.06 m 

# kalman

associate_det_one = True
#min_distance_new = 3.0

# obstruction
objects_obs_size = {"car":3.0, "person":1.0, "truck":5.0, "suv":3.0 , "van":3.0, "bicycle":1.5, "motorbike":1.5, "bus":5.0, "scooter":1.5, "ucl":1.5, "trailer":2.5, "dog":1.0, "petit":1.0, "moyen":3.0, "grand":5.0}

#obstruction_thresh = 0.0
min_det_obstruction = 3
max_invisible_obs = 200
min_obs_front = 6


high_new = 4.0
middle_new = 3.1
low_new = 0.5

new_thresh_classes = {"car":middle_new , "person":low_new, "truck":high_new, "suv":middle_new , "van":middle_new, "bicycle":low_new, "motorbike":low_new, "bus":high_new, "scooter":low_new, "ucl":middle_new, "trailer":low_new, "dog":low_new, "petit":middle_new, "moyen":middle_new, "grand":high_new}

rad_delay = 0.245
dist_focal = 5695
q_alpha = - 1.0#1.2 

delta_time = 1/100
min_dist_gate = 3.0#0.28
lambda_fp = math.sqrt(2*math.pi)*4.0
max_invisible_no_obs = 10

# hauteurs 

classes_height = {"car":1.423+0.1, "person":1.6, "truck":3.0, "suv":1.423+0.25,"van":2.5, "bicycle":0.76+0.25, "motorbike":1.10, "bus":3.31, "scooter":1.17, "ucl":1.5,"trailer":0.8, "dog":0.5, "petit":0.0, "moyen":0.0, "grand":0.0, "init":0.0}


# Matrices 

vari_Q = np.array(
	 [[0.005,0.0,0.0,0.0,0.0,0.0],\
	 [0.0,0.001,0.0,0.0,0.0,0.0],\
	 [0.0,0.0,0.001,0.0,0.0,0.0],\
	 [0.0,0.0,0.0,0.05,0.0,0.0],\
	 [0.0,0.0,0.0,0.0,0.05,0.0],\
	 [0.0,0.0,0.0,0.0,0.0,0.02]])
vari_R_c = np.array(
	 [[0.05,0.0,0.0,0.0,0.0,0.0],\
	 [0.0,0.3,0.0,0.0,0.0,0.0],\
	 [0.0,0.0,100.0,0.0,0.0,0.0],\
	 [0.0,0.0,0.0,0.1,0.0,0.0],\
	 [0.0,0.0,0.0,0.0,0.1,0.0],\
	 [0.0,0.0,0.0,0.0,0.0,0.1]])

vari_R_r = np.array(
	 [[100.0,0.0,0.0,0.0,0.0,0.0],\
	 [0.0,100.0,0.0,0.0,0.0,0.0],\
	 [0.0,0.0,0.09,0.0,0.0,0.0],\
	 [0.0,0.0,0.0,0.01,0.0,0.0],\
	 [0.0,0.0,0.0,0.0,0.01,0.0],\
	 [0.0,0.0,0.0,0.0,0.0,0.08378]])

init_P = np.array(
	 [[50.0,0.0,0.0,0.0,0.0,0.0],\
	 [0.0,50.0,0.0,0.0,0.0,0.0],\
	 [0.0,0.0,200.0,0.0,0.0,0.0],\
	 [0.0,0.0,0.0,50.0,0.0,0.0],\
	 [0.0,0.0,0.0,0.0,50.0,0.0],\
	 [0.0,0.0,0.0,0.0,0.0,50.0]])

