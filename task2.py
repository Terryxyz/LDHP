#!/usr/bin/env python
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

bottom_layer=__import__('bottom_layer_gripper1')


def update_points(num):
    global object_pose_trajectory, gripper_pose_trajectory, iteration
    if num<60:
        return line, gripper, vertex_edge 
    if iteration<len(object_pose_trajectory):
        vertex_edge.set_data([j[0] for j in object_pose_trajectory[iteration]+[object_pose_trajectory[iteration][0]]], [j[1] for j in object_pose_trajectory[iteration]+[object_pose_trajectory[iteration][0]]])
        F1, F2, _ = bottom_layer.gripper_profile(gripper_pose_trajectory[iteration])
        gripper.set_data([F1[0],F2[0]], [F1[1],F2[1]])
        iteration+=1
    return line, gripper, vertex_edge

if __name__ == '__main__':
    global object_pose_trajectory, total_finger_position_trajectory, iteration
    iteration = 0
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-0.1, 3.1), ylim=(-0.1, 3.1))
    ax.set_aspect('equal')
    #ax.grid(ls="--")
    environment = [[0,0], [3,0]]
    line, = ax.plot([i[0] for i in environment], [i[1] for i in environment], 'darkgoldenrod', lw=2)
    initial_object_vertex_position = [[1.078446275633149, 1.6893654271085456], [1.2, 1.0], [2.677211629518312, 1.2604722665003951], [2.5556579051514614, 1.9498376936089408]]
    initial_CoM_position = calculate_CoM(initial_object_vertex_position)
    vertex_edge, = ax.plot([i[0] for i in initial_object_vertex_position+[initial_object_vertex_position[0]]], [i[1] for i in initial_object_vertex_position+[initial_object_vertex_position[0]]], '-g', lw=3)
    initial_gripper_pose = (2.2678, 1.5437, 0.8, 160.75)
    ini_F1, ini_F2, _0 = bottom_layer.gripper_profile(initial_gripper_pose)
    gripper, = ax.plot([ini_F1[0],ini_F2[0]], [ini_F1[1],ini_F2[1]], '-or', lw=2, markersize=5)
    target_object_vertex_position = [[1,0], [1.7,0], [1.7,1.5], [1,1.5]]
    target_finger_position=None
    #try:
    temp_result = framework(initial_object_vertex_position, initial_CoM_position, initial_gripper_pose, target_object_vertex_position, target_finger_position, environment, 'Type 1', mu_gripper=0.2, mu_environment=0.21)
    if temp_result != False:
        object_pose_trajectory, gripper_pose_trajectory = temp_result
        while True:
            if_continue = input("continue?")
            if if_continue != '':
                break
        ani = animation.FuncAnimation(fig, update_points, np.arange(0, 5000, 1), interval=200, blit=False) 
        plt.show() 
    else:
        print('Infeasible')


