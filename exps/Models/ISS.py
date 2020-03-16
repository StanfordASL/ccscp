#!/usr/bin/env python3

import sys, os
import time
import numpy as np
import matplotlib as plt

sys.path.append('../src/utils/')

# from polygonal_obstacles import *
from polygonal_obstacles import PolygonalObstacle as PolyObs
from viz import *

def get_ISS_zones():
	#### INSIDE polytopes
	btms_lft, tops_rgt = [np.zeros(3)] * 6, [np.zeros(3)] * 6
	btms_lft[0], tops_rgt[0] = np.array([ 5.9,-0.6, 4.2]), np.array([ 7.7, 0.6, 5.4])  # 3
	btms_lft[1], tops_rgt[1] = np.array([10.2, 1.2, 4.2]), np.array([11.6, 2.7, 5.5])  # 4
	btms_lft[2], tops_rgt[2] = np.array([ 9.6, 2.7, 3.8]), np.array([11.9, 7.3, 5.9])  # 5
	btms_lft[3], tops_rgt[3] = np.array([10.3,-2.7, 4.3]), np.array([11.6,-1.2, 5.4])  # 6
	btms_lft[4], tops_rgt[4] = np.array([ 7.7,-1.2, 3.7]), np.array([11.6, 1.2, 6.0])  # 8
	btms_lft[5], tops_rgt[5] = np.array([11.6,-0.8, 4.1]), np.array([12.0, 0.8, 5.5])  # 16
	keepin_zones = []
	for (btm, top) in zip(btms_lft, tops_rgt):
		center, width = (top+btm)/2., (top-btm)
		keepin_zones.append( PolyObs(center,width) ) 


	#### OUTSIDE
	btms_lft, tops_rgt = [np.zeros(3)] * 26, [np.zeros(3)] * 26
	btms_lft[0],  tops_rgt[0]  = np.array([ 3.0,-0.6, 4.2]), np.array([ 5.9, 0.6, 5.4])  # 1
	btms_lft[1],  tops_rgt[1]  = np.array([ 5.0, 0.6, 3.7]), np.array([ 7.7, 1.2, 6.0])  # 2
	btms_lft[2],  tops_rgt[2]  = np.array([ 5.9, 1.2, 3.7]), np.array([10.2, 2.7, 6.0])  # 3
	btms_lft[3],  tops_rgt[3]  = np.array([ 8.0, 2.7, 3.8]), np.array([ 9.6, 7.3, 5.9])  # 4
	btms_lft[4],  tops_rgt[4]  = np.array([ 9.6, 7.3, 3.8]), np.array([11.9, 9.0, 5.9])  # 5
	btms_lft[5],  tops_rgt[5]  = np.array([11.9, 2.7, 3.8]), np.array([13.0, 7.3, 5.9])  # 6
	btms_lft[6],  tops_rgt[6]  = np.array([11.6, 1.2, 3.8]), np.array([12.0, 2.7, 5.9])  # 7
	btms_lft[7],  tops_rgt[7]  = np.array([11.6, 0.8, 3.7]), np.array([12.0, 1.2, 6.0])  # 8
	btms_lft[8],  tops_rgt[8]  = np.array([12.0,-0.8, 3.7]), np.array([13.0, 0.8, 6.0])  # 9
	btms_lft[9],  tops_rgt[9]  = np.array([11.6,-1.2, 3.7]), np.array([12.0,-0.8, 6.0])  # 10
	btms_lft[10], tops_rgt[10] = np.array([11.6,-2.7, 4.3]), np.array([12.0,-1.2, 5.4])  # 11
	btms_lft[11], tops_rgt[11] = np.array([10.3,-4.0, 4.3]), np.array([11.6,-2.7, 5.4])  # 12
	btms_lft[12], tops_rgt[12] = np.array([ 5.9,-4.0, 0.0]), np.array([10.3,-1.2, 0.0])  # 13
	btms_lft[13], tops_rgt[13] = np.array([ 5.9,-1.2, 3.7]), np.array([ 7.7,-0.6, 6.0])  # 14
	btms_lft[14], tops_rgt[14] = np.array([ 5.9,-1.2, 5.4]), np.array([ 7.7, 1.2, 6.0])  # 15
	btms_lft[15], tops_rgt[15] = np.array([ 5.9,-1.2, 3.0]), np.array([ 7.7, 1.2, 4.2])  # 16
	btms_lft[16], tops_rgt[16] = np.array([10.2, 1.2, 5.5]), np.array([11.6, 2.7, 7.0])  # 17
	btms_lft[17], tops_rgt[17] = np.array([10.2, 1.2, 3.0]), np.array([11.6, 2.7, 4.2])  # 18
	btms_lft[18], tops_rgt[18] = np.array([ 9.6, 2.7, 3.0]), np.array([11.9, 7.3, 3.8])  # 19
	btms_lft[19], tops_rgt[19] = np.array([ 9.6, 2.7, 5.9]), np.array([11.9, 7.3, 7.0])  # 20
	btms_lft[20], tops_rgt[20] = np.array([10.3,-2.7, 5.4]), np.array([11.6,-1.2, 7.0])  # 21
	btms_lft[21], tops_rgt[21] = np.array([10.3,-2.7, 3.0]), np.array([11.6,-1.2, 4.3])  # 22
	btms_lft[22], tops_rgt[22] = np.array([ 7.7,-1.2, 3.0]), np.array([11.6, 1.2, 3.7])  # 23
	btms_lft[23], tops_rgt[23] = np.array([ 7.7,-1.2, 6.0]), np.array([11.6, 1.2, 7.0])  # 24
	btms_lft[24], tops_rgt[24] = np.array([11.6,-0.8, 3.0]), np.array([12.0, 0.8, 4.1])  # 25
	btms_lft[25], tops_rgt[25] = np.array([11.6,-0.8, 5.5]), np.array([12.0, 0.8, 7.0])  # 26
	keepout_zones = []
	for (btm, top) in zip(btms_lft, tops_rgt):
		center, width = (top+btm)/2., (top-btm)
		keepout_zones.append( PolyObs(center,width) ) 

	return keepin_zones, keepout_zones