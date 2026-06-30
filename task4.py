#!/usr/bin/env python
import os
import time
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from framework import *
from math import *
from jerk import *
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LinearRing
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

accuracy = 0.004

bottom_layer=__import__('bottom_layer_gripper2')


def update_points(num):
    global object_pose_trajectory, gripper_pose_trajectory, iteration
    if num<3:
        return line, vertex_edge, finger1, finger2
    if iteration<len(object_pose_trajectory):
        vertex_edge.set_data([j[0] for j in object_pose_trajectory[iteration]+[object_pose_trajectory[iteration][0]]], [j[1] for j in object_pose_trajectory[iteration]+[object_pose_trajectory[iteration][0]]])
        current_F1, current_F2, current_gripper_profile = bottom_layer.gripper_profile(gripper_pose_trajectory[iteration])
        F1_F1_F2_perp1 = current_gripper_profile.geoms[0].coords[:]
        F2_F1_F2_perp1 = current_gripper_profile.geoms[1].coords[:]
        F1_perp1_out = current_gripper_profile.geoms[2].coords[:]
        F2_perp1_out = current_gripper_profile.geoms[3].coords[:]
        gripper_frame1.set_data([j[0] for j in F1_F1_F2_perp1], [j[1] for j in F1_F1_F2_perp1])
        gripper_frame2.set_data([j[0] for j in F2_F1_F2_perp1], [j[1] for j in F2_F1_F2_perp1])
        gripper_frame3.set_data([j[0] for j in F1_perp1_out], [j[1] for j in F1_perp1_out])
        gripper_frame4.set_data([j[0] for j in F2_perp1_out], [j[1] for j in F2_perp1_out])
        finger1.set_data([current_F1[0]], [current_F1[1]])
        finger2.set_data([current_F2[0]], [current_F2[1]])
        iteration+=1
    return line, vertex_edge, finger1, finger2

if __name__ == '__main__':
    global object_pose_trajectory, total_finger_position_trajectory, iteration
    iteration = 0
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-0.1, 3.1), ylim=(-0.1, 3.1))
    ax.set_aspect('equal')
    #ax.grid(ls="--")
    environment = [[0,0.75], [0,0], [3,0], [3,0.75]]
    line, = ax.plot([i[0] for i in environment], [i[1] for i in environment], 'darkgoldenrod', lw=2)
    initial_object_vertex_position = [[1,1], [1.7,1], [1.7,2.5], [1,2.5]]
    initial_CoM_position = calculate_CoM(initial_object_vertex_position)
    vertex_edge, = ax.plot([i[0] for i in initial_object_vertex_position+[initial_object_vertex_position[0]]], [i[1] for i in initial_object_vertex_position+[initial_object_vertex_position[0]]], '-g', lw=3)
    initial_gripper_pose = (1.35, 2, 0.7, 90, '+')
    ini_F1, ini_F2, ini_gripper_profile = bottom_layer.gripper_profile(initial_gripper_pose)
    F1_F1_F2_perp1 = ini_gripper_profile.geoms[0].coords[:]
    F2_F1_F2_perp1 = ini_gripper_profile.geoms[1].coords[:]
    F1_perp1_out = ini_gripper_profile.geoms[2].coords[:]
    F2_perp1_out = ini_gripper_profile.geoms[3].coords[:]
    gripper_frame1, = ax.plot([j[0] for j in F1_F1_F2_perp1], [j[1] for j in F1_F1_F2_perp1], '#54C8F4', zorder=4, lw=2)
    gripper_frame2, = ax.plot([j[0] for j in F2_F1_F2_perp1], [j[1] for j in F2_F1_F2_perp1], '#54C8F4', zorder=4, lw=2)
    gripper_frame3, = ax.plot([j[0] for j in F1_perp1_out], [j[1] for j in F1_perp1_out], '#54C8F4', zorder=4, lw=2)
    gripper_frame4, = ax.plot([j[0] for j in F2_perp1_out], [j[1] for j in F2_perp1_out], '#54C8F4', zorder=4, lw=2)
    finger1, = ax.plot([ini_F1[0]], [ini_F1[1]], '-or', zorder=5, lw=2, markersize=5)
    finger2, = ax.plot([ini_F2[0]], [ini_F2[1]], '-or', zorder=5, lw=2, markersize=5)
    target_object_vertex_position = [[1,0.7], [1,0], [2.5,0], [2.5,0.7]]
    target_finger_position=None
    #try:
    temp_result = framework(initial_object_vertex_position, initial_CoM_position, initial_gripper_pose, target_object_vertex_position, target_finger_position, environment, 'Type 2', mu_gripper=0.6, mu_environment=0.03)
    #temp_result = True
    if temp_result != False:
        object_pose_trajectory, gripper_pose_trajectory = temp_result
        #object_pose_trajectory = [[[1+i,0.7], [1+i,0], [2.5+i,0], [2.5+i,0.7]] for i in [0.1,0.2,0.3,0.4]]
        #gripper_pose_trajectory = [(1.4+i, 2, 0.8, 90, '+') for i in [0.1, 0.2, 0.3, 0.4]]
        while True:
            if_continue = input("continue?")
            if if_continue != '':
                break
        ani = animation.FuncAnimation(fig, update_points, np.arange(0, 5000, 1), interval=200, blit=False) 
        plt.show() 
    else:
        print('Infeasible')


