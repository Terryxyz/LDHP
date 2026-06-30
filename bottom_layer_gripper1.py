#!/usr/bin/env python
import time
import re
import numpy as np
from general_functions import *
from top_and_middle_layer import *
from math import *
from jerk import *
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LinearRing
from shapely.affinity import rotate
from shapely.geometry import MultiPoint
from shapely.geometry import MultiLineString
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def Feasible_Grasp(current_vertex_position, F1_position, F2_position, profile_environment, accuracy = 0.002):
    if MultiPoint([F1_position, F2_position]).distance(LineString(profile_environment))<0.1:
        return False
    if Polygon(current_vertex_position).buffer(-accuracy).intersects(MultiPoint([F1_position, F2_position]))==True:
        return False
    if MultiPoint([F1_position, F2_position]).distance(MultiPoint(current_vertex_position))<0.1:
        return False
    return True

def from_real_position_to_index(current_vertex_position, finger_position):
    if Point(finger_position).distance(Polygon(current_vertex_position))<0.01:
        point_index_temp = finger_position2index(current_vertex_position, finger_position)
        circumference0 = circumference(current_vertex_position)
        return ceil(point_index_temp/(circumference0/100.0))
    else:
        return 0

def from_index_to_real_position(current_vertex_position, finger_index):
    if finger_index <= 0:
        return None
    elif finger_index>100:
        return current_vertex_position[finger_index/100-1]
    else:
        point_index_temp = (finger_index-0.5)*(circumference(current_vertex_position)/100.0)
        point = finger_index2position(current_vertex_position, point_index_temp)
        return point

def whether_finger_contact_pair_feasible(ini_vertex_position, F1_index, F2_index):
    F1_position = from_index_to_real_position(ini_vertex_position, F1_index)
    F2_position = from_index_to_real_position(ini_vertex_position, F2_index)
    if Point(F1_position).distance(Point(F2_position))<0.78:
        return False
    if Point(F1_position).distance(Point(F2_position))>0.82:
        return False
    if MultiPoint([F1_position, F2_position]).distance(MultiPoint(ini_vertex_position))<0.1:
        return False
    return True

def feasible_finger_contact_pair_set(ini_vertex_position):
    grasp_index_list = [(i,j) for i in range(1,101) for j in range(1,101)]
    temp_list = grasp_index_list[:]
    for grasp_index in temp_list:
        if whether_finger_contact_pair_feasible(ini_vertex_position, grasp_index[0], grasp_index[1])==False:
            grasp_index_list.remove(grasp_index)
    return grasp_index_list

def feasible_fcp_avoid_env_collision(current_vertex_position, feasible_fcp_set, profile_environment, accuracy = 0.002):
    avoid_index_set = []
    for index in range(1,101):
        finger_position = from_index_to_real_position(current_vertex_position, index)
        if Point(finger_position).distance(LineString(profile_environment))<accuracy:
            avoid_index_set.append(index)
    determine_condition = False
    feasible_fcp_set0 = np.array(feasible_fcp_set)
    for avoid_index in avoid_index_set:
        determine_condition = determine_condition | (feasible_fcp_set0[:,0] == avoid_index) | (feasible_fcp_set0[:,1] == avoid_index)
    delete_index = np.argwhere(determine_condition)
    feasible_fcp_set1 = np.delete(feasible_fcp_set0, delete_index, axis = 0)
    feasible_fcp_set2 = feasible_fcp_set1
    feasible_gripper_pose_set = []
    for fcp in feasible_fcp_set2:
        gripper_pose_set = from_index_to_gripper_pose_set(current_vertex_position, fcp[0], fcp[1])
        if gripper_pose_set == []:
            feasible_fcp_set1.remove(fcp)
        feasible_gripper_pose_set+=gripper_pose_set
    return list(xtuple(feasible_fcp_set1.tolist())), feasible_gripper_pose_set

def obtain_feasible_fcp_set_of_each_pose(feasible_fcp_set, object_pose_trajectory, ini_vertex_position, profile_environment, fcp_set_index_memory = dict(), gripper_pose_set_index_memory=dict()):
    feasible_fcp_set_list = []
    feasible_gripper_pose_set_list = []
    for i in range(len(object_pose_trajectory)):
        object_pose_discrete = [[(round(vertex[0]*100)/100), (round(vertex[1]*100)/100)] for vertex in object_pose_trajectory[i]]
        if xtuple(object_pose_discrete) in fcp_set_index_memory:
            feasible_fcp_set_list.append(fcp_set_index_memory[xtuple(object_pose_discrete)])
            feasible_gripper_pose_set_list.append(gripper_pose_set_index_memory[xtuple(object_pose_discrete)])
        else:
            fcp_index_list_temp = feasible_fcp_set[:]
            feasible_fcp_set1, feasible_gripper_pose_set1 = feasible_fcp_avoid_env_collision(object_pose_trajectory[i], fcp_index_list_temp, profile_environment)
            feasible_fcp_set_list.append(feasible_fcp_set1)
            feasible_gripper_pose_set_list.append(feasible_gripper_pose_set1)
            fcp_set_index_memory[xtuple(object_pose_discrete)]=feasible_fcp_set1
            gripper_pose_set_index_memory[xtuple(object_pose_discrete)]=feasible_gripper_pose_set1
    return feasible_fcp_set_list, fcp_set_index_memory, feasible_gripper_pose_set_list, gripper_pose_set_index_memory

def from_index_to_gripper_pose_set(current_vertex_position, F1_index, F2_index):
    F1_position = from_index_to_real_position(current_vertex_position, F1_index)
    F2_position = from_index_to_real_position(current_vertex_position, F2_index)
    aperture = Point(F1_position).distance(Point(F2_position))
    center_position = [(F1_position[0]+F2_position[0])/2, (F1_position[1]+F2_position[1])/2]
    angle = inclination_angle(F1_position,F2_position)*180/pi
    return [(center_position[0], center_position[1], aperture, angle+90)]

def system_config_list_tip(current_vertex_position, current_F1, current_F2, pole, direction, rotate_angle, tip_step = 5):
    object_pose_list = []
    F1_list = []
    F2_list = []
    for angle in np.arange(0, rotate_angle, tip_step):
        if direction == 'CCW':
            object_pose = [point_position_after_rotation(vertex, pole, angle) for vertex in current_vertex_position]
            object_pose_list.append(object_pose)
            F1 = point_position_after_rotation(current_F1, pole, angle)
            F1_list.append(F1)
            F2 = point_position_after_rotation(current_F2, pole, angle)
            F2_list.append(F2)
        else:
            object_pose = [point_position_after_rotation(vertex, pole, -angle) for vertex in current_vertex_position]
            object_pose_list.append(object_pose)
            F1 = point_position_after_rotation(current_F1, pole, -angle)
            F1_list.append(F1)
            F2 = point_position_after_rotation(current_F2, pole, -angle)
            F2_list.append(F2)
    if direction == 'CCW':
        object_pose = [point_position_after_rotation(vertex, pole, rotate_angle) for vertex in current_vertex_position]
        object_pose_list.append(object_pose)
        F1 = point_position_after_rotation(current_F1, pole, rotate_angle)
        F1_list.append(F1)
        F2 = point_position_after_rotation(current_F2, pole, rotate_angle)
        F2_list.append(F2)
    else:
        object_pose = [point_position_after_rotation(vertex, pole, -rotate_angle) for vertex in current_vertex_position]
        object_pose_list.append(object_pose)
        F1 = point_position_after_rotation(current_F1, pole, -rotate_angle)
        F1_list.append(F1)
        F2 = point_position_after_rotation(current_F2, pole, -rotate_angle)
        F2_list.append(F2) 
    return object_pose_list, F1_list, F2_list

def system_config_list_push(current_vertex_position, current_F1, current_F2, direction, distance, translate_step = 0.1):
    object_pose_list = []
    F1_list = []
    F2_list = []
    for translate_step in np.arange(0, distance, (distance-0.1)/20):
        object_pose = [[vertex[0]+translate_step*direction[0], vertex[1]+translate_step*direction[1]] for vertex in current_vertex_position]
        object_pose_list.append(object_pose)
        F1 = [current_F1[0]+translate_step*direction[0], current_F1[1]+translate_step*direction[1]]
        F1_list.append(F1)
        F2 = [current_F2[0]+translate_step*direction[0], current_F2[1]+translate_step*direction[1]]
        F2_list.append(F2)
    object_pose = [[vertex[0]+distance*direction[0], vertex[1]+distance*direction[1]] for vertex in current_vertex_position]
    object_pose_list.append(object_pose)
    F1 = [current_F1[0]+distance*direction[0], current_F1[1]+distance*direction[1]]
    F1_list.append(F1)
    F2 = [current_F2[0]+distance*direction[0], current_F2[1]+distance*direction[1]]
    F2_list.append(F2) 
    return object_pose_list, F1_list, F2_list

def system_config_list_move_in_air(current_vertex_position, current_F1, current_F2, direction, distance, translate_step = 0.1):
    object_pose_list = []
    F1_list = []
    F2_list = []
    for translate_step in np.arange(0, distance, (distance-0.1)/20):
        object_pose = [[vertex[0]+translate_step*direction[0], vertex[1]+translate_step*direction[1]] for vertex in current_vertex_position]
        object_pose_list.append(object_pose)
        F1 = [current_F1[0]+translate_step*direction[0], current_F1[1]+translate_step*direction[1]]
        F1_list.append(F1)
        F2 = [current_F2[0]+translate_step*direction[0], current_F2[1]+translate_step*direction[1]]
        F2_list.append(F2)
    object_pose = [[vertex[0]+distance*direction[0], vertex[1]+distance*direction[1]] for vertex in current_vertex_position]
    object_pose_list.append(object_pose)
    F1 = [current_F1[0]+distance*direction[0], current_F1[1]+distance*direction[1]]
    F1_list.append(F1)
    F2 = [current_F2[0]+distance*direction[0], current_F2[1]+distance*direction[1]]
    F2_list.append(F2) 
    return object_pose_list, F1_list, F2_list

def system_config_list_tilting_slide(current_vertex_position, current_F1, current_F2, target_contact_state, total_angle, profile_environment):
    object_pose_list = []
    F1_list = []
    F2_list = []
    current_contact_state = obtain_contact_state(current_vertex_position, profile_environment)
    contact_edge_point = dict()
    contact_edge_point=Tilting_slide_contact_edge_point(current_vertex_position, target_contact_state, profile_environment)
    if total_angle>0:
        rotate_step = 0.5
    else:
        rotate_step = -0.5
    for angle in np.append(np.arange(0, total_angle, rotate_step), total_angle):
        new_vertex_position = Tilting_slide_calculate_next_vertex_position_simplified(contact_edge_point, angle, current_vertex_position, profile_environment)
        object_pose_list.append(new_vertex_position)
        [new_F1, new_F2] = Tilting_slide_calculate_next_vertex_position_simplified(contact_edge_point, angle, [current_F1, current_F2], profile_environment)
        F1_list.append(new_F1)
        F2_list.append(new_F2)
    return object_pose_list, F1_list, F2_list

def system_config_list(current_vertex_position, current_F1, current_F2, target_action, action_parameter, profile_environment):
    if target_action == 'Tip':
        [rotation_pole, rotation_direction, rotation_angle] = action_parameter
        return system_config_list_tip(current_vertex_position, current_F1, current_F2, rotation_pole, rotation_direction, rotation_angle)
    elif target_action == 'Push':
        [push_direction, push_distance] = action_parameter
        return system_config_list_push(current_vertex_position, current_F1, current_F2, push_direction, push_distance)
    elif target_action == 'Tilting-slide':
        [target_contact_state, A_r]=action_parameter
        return system_config_list_tilting_slide(current_vertex_position, current_F1, current_F2, target_contact_state, A_r, profile_environment)
    elif target_action == 'Move-in-air':
        [move_direction, move_distance] = action_parameter
        return system_config_list_move_in_air(current_vertex_position, current_F1, current_F2, move_direction, move_distance, profile_environment)
    else:
        return 'Wrong input'

def SuitableGrasp_tip_instantaneous(current_vertex_position, F1_index, F2_index, ini_vertex_position, ini_CoM_position, pole, direction, profile_environment, accuracy = 0.002):
    current_F1 = from_index_to_real_position(current_vertex_position, F1_index)
    current_F2 = from_index_to_real_position(current_vertex_position, F2_index)   
    contact_index, contact_list, contact_normal_list, contact_local_tangent_list = contact_points_and_normal(current_vertex_position, current_F1, current_F2, profile_environment, ini_vertex_position, F1_index, F2_index)
    contact_index_E = []
    contact_list_E = []
    contact_normal_list_E = []
    for i in range(len(contact_index)):
        if re.match('E', contact_index[i])!=None:
            contact_index_E.append(contact_index[i])
            contact_list_E.append(contact_list[i])
            contact_normal_list_E.append(contact_normal_list[i])
    contact_mode = {'F1': 'R', 'F2': 'R'}
    for i in range(len(contact_index_E)):
        radius = [contact_list_E[i][0]-pole[0], contact_list[i][1]-pole[1]]
        velocity = [-radius[1], radius[0]]
        if np.cross(radius, velocity)<0 and direction=='CCW':
            velocity = [-velocity[0], -velocity[1]]
        elif np.cross(radius, velocity)>0 and direction=='CW':
            velocity = [-velocity[0], -velocity[1]]
        if abs(sqrt(velocity[0]**2+velocity[1]**2))<0.001:
            contact_mode[contact_index_E[i]] = 'R'
        elif len((np.array(contact_normal_list_E[i])).shape)==1 and abs(np.dot(velocity, contact_normal_list_E[i]))<0.00001:
            if np.cross(velocity, contact_normal_list_E[i])>0:
                contact_mode[contact_index_E[i]] = 'S-'
            else:
                contact_mode[contact_index_E[i]] = 'S+'
        elif len((np.array(contact_normal_list_E[i])).shape)>1:
            for j in contact_normal_list_E[i]:
                if abs(np.dot(velocity, j))<0.00001:
                    if np.cross(velocity, j)>0:
                        contact_mode[contact_index_E[i]] = 'S-'
                    else:
                        contact_mode[contact_index_E[i]] = 'S+'
            else:
                contact_mode[contact_index_E[i]] = 'B'
        else:
            contact_mode[contact_index_E[i]] = 'B'
    current_CoM = calculate_current_CoM(ini_vertex_position, current_vertex_position, ini_CoM_position)
    IsStable, angle_to_normal_rad = whether_stable(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode)
    if IsStable==False:
        return False, None
    else:
        angle_to_normal_rad=max(angle_to_normal_rad.values())
        return True, angle_to_normal_rad

def Suitable_Grasp_tip(current_vertex_position, F1_index, F2_index, ini_vertex_position, ini_CoM_position, pole, direction, rotate_angle, profile_environment):
    current_F1 = from_index_to_real_position(current_vertex_position, F1_index)
    current_F2 = from_index_to_real_position(current_vertex_position, F2_index)  
    object_pose_list, F1_list, F2_list = system_config_list_tip(current_vertex_position, current_F1, current_F2, pole, direction, rotate_angle)
    cost = 0
    for i in range(len(object_pose_list)):
        IsStable_instantaneous, angle_to_normal_rad = SuitableGrasp_tip_instantaneous(object_pose_list[i], F1_index, F2_index, ini_vertex_position, ini_CoM_position, pole, direction, profile_environment)
        if IsStable_instantaneous==False:
            return False, None
        else:
            cost+=angle_to_normal_rad/(pi/2.0)
    return True, cost/len(object_pose_list)

def SuitableGrasp_push_instantaneous(current_vertex_position, F1_index, F2_index, ini_vertex_position, ini_CoM_position, direction, profile_environment):
    current_F1 = from_index_to_real_position(current_vertex_position, F1_index)
    current_F2 = from_index_to_real_position(current_vertex_position, F2_index)   
    contact_index, contact_list, contact_normal_list, contact_local_tangent_list = contact_points_and_normal(current_vertex_position, current_F1, current_F2, profile_environment, ini_vertex_position, F1_index, F2_index)
    contact_index_E = []
    contact_list_E = []
    contact_normal_list_E = []
    for i in range(len(contact_index)):
        if re.match('E', contact_index[i])!=None:
            contact_index_E.append(contact_index[i])
            contact_list_E.append(contact_list[i])
            contact_normal_list_E.append(contact_normal_list[i])
    contact_mode = {'F1': 'R', 'F2': 'R'}
    for i in range(len(contact_index_E)):
        velocity = direction
        if abs(sqrt(velocity[0]**2+velocity[1]**2))<0.00001:
            contact_mode[contact_index_E[i]] = 'R'
        elif len((np.array(contact_normal_list_E[i])).shape)==1 and abs(np.dot(velocity, contact_normal_list_E[i]))<0.00001:
            if np.cross(velocity, contact_normal_list_E[i])>0:
                contact_mode[contact_index_E[i]] = 'S-'
            else:
                contact_mode[contact_index_E[i]] = 'S+'
        elif len((np.array(contact_normal_list_E[i])).shape)>1:
            for j in contact_normal_list_E[i]:
                if abs(np.dot(velocity, j))<0.00001:
                    if np.cross(velocity, j)>0:
                        contact_mode[contact_index_E[i]] = 'S-'
                    else:
                        contact_mode[contact_index_E[i]] = 'S+'
        else:
            contact_mode[contact_index_E[i]] = 'B'
    current_CoM = calculate_current_CoM(ini_vertex_position, current_vertex_position, ini_CoM_position)
    IsStable, angle_to_normal_rad = whether_stable(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode)
    if IsStable==False:
        return False, None
    else:
        angle_to_normal_rad=max(angle_to_normal_rad.values())
        return True, angle_to_normal_rad

def Suitable_Grasp_push(current_vertex_position, F1_index, F2_index, ini_vertex_position, ini_CoM_position, direction, distance, profile_environment):
    current_F1 = from_index_to_real_position(current_vertex_position, F1_index)
    current_F2 = from_index_to_real_position(current_vertex_position, F2_index)  
    object_pose_list, F1_list, F2_list = system_config_list_push(current_vertex_position, current_F1, current_F2, direction, distance)
    cost=0
    for i in np.arange(0,len(object_pose_list)-2,len(object_pose_list)-3):
        IsStable_instantaneous, angle_to_normal_rad = SuitableGrasp_push_instantaneous(object_pose_list[i], F1_index, F2_index, ini_vertex_position, ini_CoM_position, direction, profile_environment)
        if IsStable_instantaneous==False:
            return False, None
        else:
            cost+=angle_to_normal_rad/(pi/2.0)
        return True, cost/len(object_pose_list)

def SuitableGrasp_move_in_air_instantaneous(current_vertex_position, F1_index, F2_index, ini_vertex_position, ini_CoM_position, direction, profile_environment):
    current_F1 = from_index_to_real_position(current_vertex_position, F1_index)
    current_F2 = from_index_to_real_position(current_vertex_position, F2_index) 
    contact_index, contact_list, contact_normal_list, contact_local_tangent_list = contact_points_and_normal(current_vertex_position, current_F1, current_F2, profile_environment, ini_vertex_position, F1_index, F2_index)
    contact_index_E = []
    contact_list_E = []
    contact_normal_list_E = []
    for i in range(len(contact_index)):
        if re.match('E', contact_index[i])!=None:
            contact_index_E.append(contact_index[i])
            contact_list_E.append(contact_list[i])
            contact_normal_list_E.append(contact_normal_list[i])
    contact_mode = {'F1': 'R', 'F2': 'R'}
    for i in range(len(contact_index_E)):
        contact_mode[contact_index_E[i]] = 'B'
    current_CoM = calculate_current_CoM(ini_vertex_position, current_vertex_position, ini_CoM_position)
    IsStable, angle_to_normal_rad = whether_stable(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode)
    if IsStable==False:
        return False, None
    else:
        angle_to_normal_rad=max(angle_to_normal_rad.values())
        return True, angle_to_normal_rad

def Suitable_Grasp_move_in_air(current_vertex_position, F1_index, F2_index, ini_vertex_position, ini_CoM_position, direction, distance, profile_environment):
    current_F1 = from_index_to_real_position(current_vertex_position, F1_index)
    current_F2 = from_index_to_real_position(current_vertex_position, F2_index)  
    object_pose_list, F1_list, F2_list = system_config_list_push(current_vertex_position, current_F1, current_F2, direction, distance)
    cost = 0
    for i in np.arange(0,len(object_pose_list)-2,len(object_pose_list)-3):
        IsStable_instantaneous, angle_to_normal_rad = SuitableGrasp_move_in_air_instantaneous(object_pose_list[i], F1_index, F2_index, ini_vertex_position, ini_CoM_position, direction, profile_environment)
        if IsStable_instantaneous==False:
            return False, None
        else:
            cost+=angle_to_normal_rad/(pi/2.0)
        return True, cost/len(object_pose_list)

def SuitableGrasp_tilting_slide_instantaneous(current_vertex_position, F1_index, F2_index, ini_vertex_position, ini_CoM_position, target_contact_state, rotate_angle_direction, profile_environment):
    current_F1 = from_index_to_real_position(current_vertex_position, F1_index)
    current_F2 = from_index_to_real_position(current_vertex_position, F2_index) 
    contact_index, contact_list, contact_normal_list, contact_local_tangent_list = contact_points_and_normal(current_vertex_position, current_F1, current_F2, profile_environment, ini_vertex_position, F1_index, F2_index)
    contact_index_E = []
    contact_list_E = []
    contact_normal_list_E = []
    for i in range(len(contact_index)):
        if re.match('E', contact_index[i])!=None:
            contact_index_E.append(contact_index[i])
            contact_list_E.append(contact_list[i])
            contact_normal_list_E.append(contact_normal_list[i])
    contact_mode = {'F1': 'R', 'F2': 'R'}   
    contact_edge_point = dict()
    contact_edge_point=Tilting_slide_contact_edge_point(current_vertex_position, target_contact_state, profile_environment) 
    for i in range(len(contact_index_E)):
        new_point = Tilting_slide_calculate_next_vertex_position_simplified(contact_edge_point, rotate_angle_direction, [contact_list_E[i]], profile_environment)[0]
        velocity = [new_point[0]-contact_list_E[i][0], new_point[1]-contact_list_E[i][1]]
        if abs(sqrt(velocity[0]**2+velocity[1]**2))<0.00001:
            contact_mode[contact_index_E[i]] = 'R'
        elif len((np.array(contact_normal_list_E[i])).shape)==1 and abs(np.dot(velocity, contact_normal_list_E[i]))<0.00001:
            if np.cross(velocity, contact_normal_list_E[i])>0:
                contact_mode[contact_index_E[i]] = 'S-'
            else:
                contact_mode[contact_index_E[i]] = 'S+'
        elif len((np.array(contact_normal_list_E[i])).shape)>1:
            for j in contact_normal_list_E[i]:
                if abs(np.dot(velocity, j))<0.00001:
                    if np.cross(velocity, j)>0:
                        contact_mode[contact_index_E[i]] = 'S-'
                    else:
                        contact_mode[contact_index_E[i]] = 'S+'
        else:
            contact_mode[contact_index_E[i]] = 'B'
    current_CoM = calculate_current_CoM(ini_vertex_position, current_vertex_position, ini_CoM_position)
    IsStable, angle_to_normal_rad = whether_stable(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode)
    if IsStable==False:
        return False, None
    else:
        angle_to_normal_rad=max(angle_to_normal_rad.values())
        return True, angle_to_normal_rad

def Suitable_Grasp_tilting_slide(current_vertex_position, F1_index, F2_index, ini_vertex_position, ini_CoM_position, target_contact_state, A_r, profile_environment):
    current_F1 = from_index_to_real_position(current_vertex_position, F1_index)
    current_F2 = from_index_to_real_position(current_vertex_position, F2_index) 
    object_pose_list, F1_list, F2_list = system_config_list_tilting_slide(current_vertex_position, current_F1, current_F2, target_contact_state, A_r, profile_environment)  
    cost = 0
    if A_r > 0:
        rotate_angle_direction = 1
    else:
        rotate_angle_direction = -1
    for i in np.arange(1,len(object_pose_list)-2,8):
        IsStable_instantaneous, angle_to_normal_rad = SuitableGrasp_tilting_slide_instantaneous(object_pose_list[i], F1_index, F2_index, ini_vertex_position, ini_CoM_position, target_contact_state, rotate_angle_direction, profile_environment)
        if IsStable_instantaneous==False:
            return False, None
        else:
            cost+=angle_to_normal_rad/(pi/2.0)
        return True, cost/len(object_pose_list)

def object_trajectory_moveobject(current_vertex_position, action, parameter, profile_environment):
    trajectory = [current_vertex_position]
    if action=='Tip':
        for angle in np.hstack((np.arange(0, parameter[2], 5*np.sign(parameter[2])),parameter[2])): 
            if parameter[1] == 'CW':
                new_vertex_position = [point_position_after_rotation(point, parameter[0], -angle) for point in current_vertex_position]
            else:
                new_vertex_position = [point_position_after_rotation(point, parameter[0], angle) for point in current_vertex_position]
            trajectory.append(new_vertex_position)
    elif action=='Push' or action=='Move-in-air':
        for distance in np.hstack((np.arange(0, parameter[1], 0.1*np.sign(parameter[1])),parameter[1])): 
            new_vertex_position = [[point[0]+distance*parameter[0][0], point[1]+distance*parameter[0][1]] for point in current_vertex_position]
            trajectory.append(new_vertex_position)
    elif action=='Tilting-slide':
        contact_edge_point=Tilting_slide_contact_edge_point(current_vertex_position, parameter[0], profile_environment)
        for angle in np.hstack((np.arange(0, parameter[1], 2*np.sign(parameter[1])),parameter[1])): 
            trajectory.append(Tilting_slide_calculate_next_vertex_position_simplified(contact_edge_point, angle, current_vertex_position, profile_environment))
    return trajectory

#print object_trajectory_moveobject([[0.0,0], [0.7,0],[0.7,1.5],[0,1.5]], 'Tilting-slide', [['V1E2', 'V2E1', 'double_contact'], -20], [[0,3], [0,0], [3,0]])
# need check

def gripper_trajectory_moveobject(current_vertex_position, current_gripper_pose, action, parameter, profile_environment):
    trajectory = [current_gripper_pose]
    _0, _1, current_gripper_profile = gripper_profile(current_gripper_pose)
    if action=='Tip':
        for angle in np.hstack((np.arange(0, parameter[2], 5*np.sign(parameter[2])),parameter[2])): 
            if parameter[1] == 'CW':
                new_gripper_profile = rotate(current_gripper_profile, -angle, origin=Point(parameter[0]))
            else:
                new_gripper_profile = rotate(current_gripper_profile, angle, origin=Point(parameter[0]))
            trajectory.append(gripper_pose_from_profile(new_gripper_profile))
    elif action=='Push' or action=='Move-in-air':
        for distance in np.hstack((np.arange(0, parameter[1], 0.1*np.sign(parameter[1])),parameter[1])): 
            new_gripper_profile = translate(current_gripper_profile, xoff=distance*parameter[0][0], yoff=distance*parameter[0][1])
            trajectory.append(gripper_pose_from_profile(new_gripper_profile))
    elif action=='Tilting-slide':
        contact_edge_point=Tilting_slide_contact_edge_point(current_vertex_position, parameter[0], profile_environment)
        for angle in np.hstack((np.arange(0, parameter[1], 2*np.sign(parameter[1])),parameter[1])): 
            new_object_vertex_position = Tilting_slide_calculate_next_vertex_position_simplified(contact_edge_point, angle, current_vertex_position, profile_environment)   
            delta_translate, D_r, A_r = calculate_parameter_translate_and_rotate_moveA(current_vertex_position[0], current_vertex_position[1], new_object_vertex_position[0], new_object_vertex_position[1])
            new_gripper_profile=rotate(current_gripper_profile, A_r, origin=Point(current_vertex_position[0]), use_radians=True)
            new_gripper_profile=translate(new_gripper_profile, xoff=delta_translate[0], yoff=delta_translate[1])
            trajectory.append(gripper_pose_from_profile(new_gripper_profile))
    return trajectory

def whether_fcp(current_vertex_position, F1_index, F2_index, ini_vertex_position, ini_CoM_position, target_action, action_parameter, profile_environment):
    F1_position = from_index_to_real_position(current_vertex_position, F1_index)
    F2_position = from_index_to_real_position(current_vertex_position, F2_index)
    Is_force_closure=whether_force_closure(current_vertex_position, F1_position, F2_position)
    if Is_force_closure == True:
        cost = 0.5
    else:
        cost = 1
    if target_action == 'Tip':
        [rotation_pole, rotation_direction, rotation_angle] = action_parameter
        whether_suitable, cost_add=Suitable_Grasp_tip(current_vertex_position, F1_index, F2_index, ini_vertex_position, ini_CoM_position, rotation_pole, rotation_direction, rotation_angle, profile_environment)
        if whether_suitable==True:
            return True, cost+cost_add
        else:
            return False, None
    elif target_action == 'Push':
        [push_direction, push_distance] = action_parameter
        whether_suitable, cost_add=Suitable_Grasp_push(current_vertex_position, F1_index, F2_index, ini_vertex_position, ini_CoM_position, push_direction, push_distance, profile_environment)
        if whether_suitable==True:
            return True, cost+cost_add
        else:
            return False, None
    elif target_action == 'Tilting-slide':
        [target_contact_state, A_r]=action_parameter
        whether_suitable, cost_add=Suitable_Grasp_tilting_slide(current_vertex_position, F1_index, F2_index, ini_vertex_position, ini_CoM_position, target_contact_state, A_r, profile_environment)
        if whether_suitable==True:
            return True, cost+cost_add
        else:
            return False, None
    elif target_action == 'Move-in-air':
        [move_direction, move_distance] = action_parameter
        whether_suitable, cost_add=Suitable_Grasp_move_in_air(current_vertex_position, F1_index, F2_index, ini_vertex_position, ini_CoM_position, move_direction, move_distance, profile_environment)
        if whether_suitable==True:
            return True, cost+cost_add
        else:
            return False, None
    else:
        return False, None

def obtain_suitable_fcp_set_of_each_action(action_sequence, action_parameter_sequence, object_pose_trajectory_plus_ini, feasible_grasp_set_list_plus_ini, ini_vertex_position, ini_CoM_position, profile_environment):
    suitable_grasp_list_sequence = []
    suitable_grasp_cost_list_sequence = []
    for i in range(len(action_sequence)):
        suitable_grasp_list=[]
        suitable_grasp_cost_list=[]
        current_feasible_grasp_set = list(set(feasible_grasp_set_list_plus_ini[i]).intersection(set(feasible_grasp_set_list_plus_ini[i+1])))
        for grasp in current_feasible_grasp_set:
            if grasp[0]==0 or grasp[1]==0:
                continue
            whether_suitable, cost = whether_fcp(object_pose_trajectory_plus_ini[i], grasp[0], grasp[1], ini_vertex_position, ini_CoM_position, action_sequence[i], action_parameter_sequence[i], profile_environment)
            if whether_suitable==True:
                suitable_grasp_list.append(grasp)
                suitable_grasp_cost_list.append(cost)
        suitable_grasp_list_sequence.append(suitable_grasp_list)
        suitable_grasp_cost_list_sequence.append(suitable_grasp_cost_list)
    return suitable_grasp_list_sequence, suitable_grasp_cost_list_sequence

def gripper_pose_map_moveobject(action_sequence, action_parameter_sequence, object_pose_trajectory_plus_ini, suitable_fcp_list_sequence, profile_environment):
    gripper_pose_map_moveobject_link_set = []
    for i in range(len(action_sequence)):
        for j in range(len(suitable_fcp_list_sequence[i])):
            gripper_pose_set_begin = from_index_to_gripper_pose_set(object_pose_trajectory_plus_ini[i], suitable_fcp_list_sequence[i][j][0], suitable_fcp_list_sequence[i][j][1])
            for k in range(len(gripper_pose_set_begin)):
                action_gripper_pose_trajectory = gripper_trajectory_moveobject(object_pose_trajectory_plus_ini[i], gripper_pose_set_begin[k], action_sequence[i], action_parameter_sequence[i], profile_environment)
                for l in range(len(action_gripper_pose_trajectory)):
                    _0, _1, gripper_profile0 = gripper_profile(action_gripper_pose_trajectory[l])
                    if gripper_profile0.distance(LineString(profile_environment))<0.01:
                        break
                else:
                    gripper_pose_map_moveobject_link_set.append([[i, gripper_pose_set_begin[k]], [i+1, action_gripper_pose_trajectory[-1]], 0.1, action_sequence[i], action_parameter_sequence[i]])
    return gripper_pose_map_moveobject_link_set

def gripper_pose_map_adjustgrasp(object_pose_trajectory_plus_ini, feasible_fcp_set_list, feasible_gripper_pose_set_list, gripper_pose_map_moveobject_link_set, ini_gripper_pose, profile_environment, ini_vertex_position, ini_CoM_position):
    gripper_pose_map_adjustgrasp_link_set = []
    for i in range(len(object_pose_trajectory_plus_ini)):
        print('i', i)
        current_CoM = calculate_current_CoM(ini_vertex_position, object_pose_trajectory_plus_ini[i], ini_CoM_position)   
        if i == 0:
            moveobj_link_end_equal_i_gripper_pose = feasible_gripper_pose_set_list[i]
        else:
            moveobj_link_end_equal_i_index = []
            for k in range(len(gripper_pose_map_moveobject_link_set)):
                if gripper_pose_map_moveobject_link_set[k][1][0]==i:
                    moveobj_link_end_equal_i_index.append(k)
            moveobj_link_end_equal_i_gripper_pose = []
            for i0 in moveobj_link_end_equal_i_index:
                moveobj_link_end_equal_i_gripper_pose.append(gripper_pose_map_moveobject_link_set[i0][1][1])
        if ini_gripper_pose != None:
            ini_F1, ini_F2, ini_ =gripper_profile(ini_gripper_pose)  
            contact_index, contact_list, contact_normal_list, contact_local_tangent_list = contact_points_and_normal(object_pose_trajectory_plus_ini[0], ini_F1, ini_F2, profile_environment)
        if i==0 and ini_gripper_pose != None and whether_AdjustGraspAction_feasible(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, 'MoveGripper', ini_F1, ini_F2)==True:   # need change for gripper 2
            moveobj_link_start_equal_i_index = []
            for k in range(len(gripper_pose_map_moveobject_link_set)):
                if gripper_pose_map_moveobject_link_set[k][0][0]==i:
                    moveobj_link_start_equal_i_index.append(k)
            moveobj_link_start_equal_i_gripper_pose = []
            for i0 in moveobj_link_start_equal_i_index:
                moveobj_link_start_equal_i_gripper_pose.append(gripper_pose_map_moveobject_link_set[i0][0][1])
            for feasible_gripper_pose in moveobj_link_start_equal_i_gripper_pose:
                gripper_pose_map_adjustgrasp_link_set.append([[i, ini_gripper_pose], [i, feasible_gripper_pose], 0.005, ['MoveGripper', feasible_gripper_pose]])
            gripper_pose_reachable_i = moveobj_link_start_equal_i_gripper_pose
        elif i != len(object_pose_trajectory_plus_ini)-1:
            moveobj_link_start_equal_i_index = []
            for k in range(len(gripper_pose_map_moveobject_link_set)):
                if gripper_pose_map_moveobject_link_set[k][0][0]==i:
                    moveobj_link_start_equal_i_index.append(k)
            feasible_gripper_pose_subset_i = []
            moveobj_link_start_equal_i_gripper_pose = []
            for i0 in moveobj_link_start_equal_i_index:
                moveobj_link_start_equal_i_gripper_pose.append(gripper_pose_map_moveobject_link_set[i0][0][1])
            temp_contact_index, temp_contact_list, temp_contact_normal_list, temp_contact_local_tangent_list = contact_points_and_normal(object_pose_trajectory_plus_ini[i], None, None, profile_environment)
            if i>=1:
                gripper_pose_reachable_last_i = gripper_pose_reachable_i
            gripper_pose_reachable_i = []
            if i>=1: 
                gripper_pose_reachable_current_i = []
                for item0 in gripper_pose_reachable_last_i:
                    for item1 in gripper_pose_map_moveobject_link_set:
                        if item1[0][0]==i-1 and item1[0][1]==item0 and item1[1][0]==i:
                            gripper_pose_reachable_current_i.append(item1[1][1])
            for j in range(len(moveobj_link_end_equal_i_gripper_pose)):
                #print 'i', i, 'j0', j, len(moveobj_link_end_equal_i_gripper_pose)
                if i>=1 and moveobj_link_end_equal_i_gripper_pose[j] not in gripper_pose_reachable_current_i:
                    continue
                for feasible_gripper_pose in moveobj_link_start_equal_i_gripper_pose:
                    if abs(moveobj_link_end_equal_i_gripper_pose[j][0]-feasible_gripper_pose[0])<0.005 and abs(moveobj_link_end_equal_i_gripper_pose[j][1]-feasible_gripper_pose[1])<0.005 and abs(moveobj_link_end_equal_i_gripper_pose[j][2]-feasible_gripper_pose[2])<0.01 and abs(moveobj_link_end_equal_i_gripper_pose[j][3]-feasible_gripper_pose[3])<1:
                        gripper_pose_map_adjustgrasp_link_set.append([[i, moveobj_link_end_equal_i_gripper_pose[j]], [i, feasible_gripper_pose], 0.00001, ['MoveGripper', feasible_gripper_pose]])
                        gripper_pose_reachable_i.append(feasible_gripper_pose)                
                current_F1, current_F2, _ =gripper_profile(moveobj_link_end_equal_i_gripper_pose[j])
                temp_contact_index_finger, temp_contact_list_finger, temp_contact_normal_list_finger, temp_contact_local_tangent_list_finger = contact_points_and_normal_finger(object_pose_trajectory_plus_ini[i], current_F1, current_F2) # need change for gripper 2
                contact_index = temp_contact_index+temp_contact_index_finger
                contact_list = temp_contact_list+temp_contact_list_finger
                contact_normal_list = temp_contact_normal_list + temp_contact_normal_list_finger
                contact_local_tangent_list = temp_contact_local_tangent_list + temp_contact_local_tangent_list_finger             
                feasible_action_list = AdjustGraspAction_feasible_list(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, current_F1, current_F2)  # need change for gripper 2
                for action in feasible_action_list:
                    if action == 'MoveGripper':
                        for feasible_gripper_pose in moveobj_link_start_equal_i_gripper_pose:
                            gripper_pose_map_adjustgrasp_link_set.append([[i, moveobj_link_end_equal_i_gripper_pose[j]], [i, feasible_gripper_pose], 0.005, ['MoveGripper', feasible_gripper_pose]])
                            gripper_pose_reachable_i.append(feasible_gripper_pose)

                        break
                    else:
                        target_gripper_pose_set = finger_position_after_action(object_pose_trajectory_plus_ini[i], moveobj_link_end_equal_i_gripper_pose[j], action, profile_environment, ini_vertex_position, ini_CoM_position, accuracy=0.01)
                        for item in target_gripper_pose_set:
                            for feasible_gripper_pose in feasible_gripper_pose_set_list[i]:
                                if abs(item[0][0]-feasible_gripper_pose[0])<0.005 and abs(item[0][1]-feasible_gripper_pose[1])<0.005 and abs(item[0][2]-feasible_gripper_pose[2])<0.01 and abs(item[0][3]-feasible_gripper_pose[3])<1:
                                    gripper_pose_map_adjustgrasp_link_set.append([[i, moveobj_link_end_equal_i_gripper_pose[j]], [i, feasible_gripper_pose], item[1], item[2]])
                                    feasible_gripper_pose_subset_i.append(feasible_gripper_pose)
                            for feasible_gripper_pose in moveobj_link_start_equal_i_gripper_pose:
                                if abs(item[0][0]-feasible_gripper_pose[0])<0.005 and abs(item[0][1]-feasible_gripper_pose[1])<0.005 and abs(item[0][2]-feasible_gripper_pose[2])<0.01 and abs(item[0][3]-feasible_gripper_pose[3])<1:
                                    gripper_pose_map_adjustgrasp_link_set.append([[i, moveobj_link_end_equal_i_gripper_pose[j]], [i, feasible_gripper_pose], item[1], item[2]])
                                    gripper_pose_reachable_i.append(feasible_gripper_pose)
            for j in range(len(feasible_gripper_pose_subset_i)):
                #print 'i', i, 'j1', j, len(feasible_gripper_pose_subset_i)
                current_F1, current_F2, _ =gripper_profile(feasible_gripper_pose_subset_i[j])
                temp_contact_index_finger, temp_contact_list_finger, temp_contact_normal_list_finger, temp_contact_local_tangent_list_finger = contact_points_and_normal_finger(object_pose_trajectory_plus_ini[i], current_F1, current_F2) # need change for gripper 2
                contact_index = temp_contact_index+temp_contact_index_finger
                contact_list = temp_contact_list+temp_contact_list_finger
                contact_normal_list = temp_contact_normal_list + temp_contact_normal_list_finger
                contact_local_tangent_list = temp_contact_local_tangent_list + temp_contact_local_tangent_list_finger 
                feasible_action_list = AdjustGraspAction_feasible_list(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, current_F1, current_F2)
                for action in feasible_action_list:
                    target_gripper_pose_set = finger_position_after_action(object_pose_trajectory_plus_ini[i], feasible_gripper_pose_subset_i[j], action, profile_environment, ini_vertex_position, ini_CoM_position, accuracy=0.01)
                    for item in target_gripper_pose_set:
                        for feasible_gripper_pose in moveobj_link_start_equal_i_gripper_pose:
                            if abs(item[0][0]-feasible_gripper_pose[0])<0.005 and abs(item[0][1]-feasible_gripper_pose[1])<0.005 and abs(item[0][2]-feasible_gripper_pose[2])<0.01 and abs(item[0][3]-feasible_gripper_pose[3])<1:
                                gripper_pose_map_adjustgrasp_link_set.append([[i, feasible_gripper_pose_subset_i[j]], [i, feasible_gripper_pose], item[1], item[2]])
                                gripper_pose_reachable_i.append(feasible_gripper_pose)
    achievable_index = 0
    for link in gripper_pose_map_adjustgrasp_link_set:
        if link[1][0]>achievable_index:
            achievable_index = link[1][0]
    if achievable_index != len(object_pose_trajectory_plus_ini)-2:
        return []
    return gripper_pose_map_adjustgrasp_link_set

def A_star_find_action_path(ini_gripper_pose, gripper_pose_map_moveobject_link_set, gripper_pose_map_adjustgrasp_link_set, object_pose_trajectory_plus_ini, feasible_fcp_set_list, initial_CoM_position, profile_environment):
    ini_feasible_gripper_pose_subset=[]
    total_obj_pose_num=len(object_pose_trajectory_plus_ini)
    map_link_set = gripper_pose_map_moveobject_link_set+gripper_pose_map_adjustgrasp_link_set
    ini_F1, ini_F2, ini_ =gripper_profile(ini_gripper_pose)
    contact_index, contact_list, contact_normal_list, contact_local_tangent_list = contact_points_and_normal(object_pose_trajectory_plus_ini[0], ini_F1, ini_F2, profile_environment)
    if whether_AdjustGraspAction_feasible(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, initial_CoM_position, 'MoveGripper', ini_F1, ini_F2)==False:
        for j in range(len(feasible_fcp_set_list[0])):
            ini_feasible_gripper_pose_subset+=from_index_to_gripper_pose_set(object_pose_trajectory_plus_ini[0], feasible_fcp_set_list[0][j][0], feasible_fcp_set_list[0][j][1])
        for discrete_gripper_pose in ini_feasible_gripper_pose_subset:
            if abs(ini_gripper_pose[0]-discrete_gripper_pose[0])<0.01 and abs(ini_gripper_pose[1]-discrete_gripper_pose[1])<0.01 and abs(ini_gripper_pose[2]-discrete_gripper_pose[2])<0.01 and abs(ini_gripper_pose[3]-discrete_gripper_pose[3])<1:
                ini_gripper_pose = discrete_gripper_pose
                break
        else:
            print('Error! Initial gripper pose not valid!')
            return False
    initial_node = (0, ini_gripper_pose)
    OPEN = {initial_node}
    CLOSED = set()
    past_cost=dict()
    past_cost[initial_node]=0
    parent_node = dict()
    action_parameter_from_parent_to_current=dict()
    while OPEN != set():
        min_past_cost = past_cost[list(OPEN)[0]]+(total_obj_pose_num-list(OPEN)[0][0])
        first_node = list(OPEN)[0]
        for node in list(OPEN):
            if past_cost[node]+(total_obj_pose_num-node[0])<min_past_cost:
                min_past_cost = past_cost[node]+(total_obj_pose_num-node[0])
                first_node=node
        print(first_node)
        OPEN.remove(first_node)
        CLOSED.add(first_node)    
        if first_node[0]==total_obj_pose_num-1:
            final_node = first_node
            break
        for link in map_link_set:
            if tuple(link[0]) == first_node and tuple(link[1]) not in CLOSED:
                tentative_past_cost = past_cost[first_node]+link[2]
                if tuple(link[1]) not in past_cost or tentative_past_cost<past_cost[tuple(link[1])]:
                    past_cost[tuple(link[1])]=tentative_past_cost
                    parent_node[tuple(link[1])]=first_node
                    if link[1][0] != link[0][0]:
                        action_parameter_from_parent_to_current[tuple(link[1])]=['MoveObject', link[3], link[4]]
                    else:
                        action_parameter_from_parent_to_current[tuple(link[1])]=['AdjustGrasp', link[3]]
                    OPEN.add(tuple(link[1]))
    else:
        print('Error! There is no feasible path.')
        return False
    
    final_node_temp = final_node
    action_and_parameter_sequence = []
    gripper_trajectory_node_sequence = []
    while final_node != initial_node:
        action_and_parameter_sequence.append(action_parameter_from_parent_to_current[final_node])
        gripper_trajectory_node_sequence.append(final_node)
        final_node = parent_node[final_node]
    gripper_trajectory_node_sequence.append(initial_node)
    return gripper_trajectory_node_sequence[::-1], action_and_parameter_sequence[::-1]                 

# need change, smaller cost better

#print from_real_position_to_index([[0.5,0], [1.2,0], [1.2,1.5], [0.5,1.5]], [0.5,1.5])

#print whether_fcp([[0,0], [0.7,0], [0.7,1.5], [0,1.5]], 32, 66, [[0,0], [0.7,0], [0.7,1.5], [0,1.5]], [0.35,0.75], 'Tilting-slide', [['V1E2', 'V2E1', 'double_contact'], -10], [[0,3], [0,0], [3,0]])

def AdjustGraspAction_feasible_list(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, current_F1_position, current_F2_position):
    feasible_action_list = []
    for action in ['MoveGripper','PivotF1CW','PivotF1CCW','PivotF2CW','PivotF2CCW','SlideF1+', 'SlideF1-','SlideF2+', 'SlideF2-', 'Slide+PivotF1CW', 'Slide+PivotF1CCW', 'Slide-PivotF1CW', 'Slide-PivotF1CCW','Slide+PivotF2CW', 'Slide+PivotF2CCW', 'Slide-PivotF2CW', 'Slide-PivotF2CCW']:   # another action: move to target grasp
        if whether_AdjustGraspAction_feasible(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, action, current_F1_position, current_F2_position)==True:
            feasible_action_list.append(action)
    return feasible_action_list

def whether_AdjustGraspAction_feasible(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, action, current_F1_position, current_F2_position):
    temp_contact_index = contact_index[:]
    if 'F1' in temp_contact_index and Point(contact_list[temp_contact_index.index('F1')]).distance(Point(current_F2_position))<0.01:
        contact_index[temp_contact_index.index('F1')]='F2'
    if 'F2' in temp_contact_index and Point(contact_list[temp_contact_index.index('F2')]).distance(Point(current_F1_position))<0.01:
        contact_index[temp_contact_index.index('F2')]='F1'
    contact_index_E = []
    for i in range(len(contact_index)):
        if re.match('E', contact_index[i])!=None:
            contact_index_E.append(contact_index[i])
    contact_finger_number = len(contact_index)-len(contact_index_E)
    if contact_finger_number==0:
        if action == 'MoveGripper':
            return True
    elif contact_finger_number==1:
        if 'F1' in contact_index:
            if action == 'MoveGripper':
                contact_mode1 = dict({'F1':'B'}, **dict.fromkeys(contact_index_E, 'R'))
                if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode1)==True:
                    return True
            elif action == 'PivotF1CW' or action == 'PivotF1CCW':
                contact_mode1 = dict({'F1':'R'}, **dict.fromkeys(contact_index_E, 'R'))
                if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode1)==True:
                    return True
            elif action == 'Slide+PivotF1CW' or action == 'Slide+PivotF1CCW':
                contact_mode2 = dict({'F1':'S+'}, **dict.fromkeys(contact_index_E, 'R'))
                if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode2)==True:
                    return True
            elif action == 'Slide-PivotF1CW' or action == 'Slide-PivotF1CCW':
                contact_mode3 = dict({'F1':'S-'}, **dict.fromkeys(contact_index_E, 'R'))
                if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode3)==True:
                    return True
        elif 'F2' in contact_index:
            if action == 'MoveGripper':
                contact_mode1 = dict({'F2':'B'}, **dict.fromkeys(contact_index_E, 'R'))
                if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode1)==True:
                    return True
            elif action == 'PivotF2CW' or action == 'PivotF2CCW':
                contact_mode1 = dict({'F2':'R'}, **dict.fromkeys(contact_index_E, 'R'))
                if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode1)==True:
                    return True
            elif action == 'Slide+PivotF2CW' or action == 'Slide+PivotF2CCW':
                contact_mode2 = dict({'F2':'S+'}, **dict.fromkeys(contact_index_E, 'R'))
                if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode2)==True:
                    return True
            elif action == 'Slide-PivotF2CW' or action == 'Slide-PivotF2CCW':
                contact_mode3 = dict({'F2':'S-'}, **dict.fromkeys(contact_index_E, 'R'))
                if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode3)==True:
                    return True
    elif contact_finger_number==2:
        if action == 'MoveGripper':
            contact_mode1 = dict({'F1':'B', 'F2': 'B'}, **dict.fromkeys(contact_index_E, 'R'))
            if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode1)==True:
                return True
        elif action == 'PivotF1CW' or action == 'PivotF1CCW':
            contact_mode1 = dict({'F1':'R', 'F2': 'B'}, **dict.fromkeys(contact_index_E, 'R'))
            if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode1)==True:
                return True
        elif action == 'PivotF2CW' or action == 'PivotF2CCW':
            contact_mode2 = dict({'F1':'B', 'F2': 'R'}, **dict.fromkeys(contact_index_E, 'R'))
            if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode2)==True:
                return True
        elif action == 'SlideF1+' or action == 'SlideF1-' or action == 'SlideF2+' or action == 'SlideF2-':
            F1_contact_normal = contact_normal_list[contact_index.index('F1')]
            F2_contact_normal = contact_normal_list[contact_index.index('F2')]
            if abs(F1_contact_normal[0] + F2_contact_normal[0])<0.0001 and abs(F1_contact_normal[1] + F2_contact_normal[1])<0.0001:
                if action == 'SlideF1+' or action == 'SlideF2-':
                    contact_mode3 = dict({'F1':'S+', 'F2': 'S-'}, **dict.fromkeys(contact_index_E, 'R'))
                    if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode3)==True:
                        return True
                elif action == 'SlideF1-' or action == 'SlideF2+':
                    contact_mode3 = dict({'F1':'S-', 'F2': 'S+'}, **dict.fromkeys(contact_index_E, 'R'))
                    if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode3)==True:
                        return True
            else:
                if action == 'SlideF1+':
                    contact_mode3 = dict({'F1':'S+', 'F2': 'B'}, **dict.fromkeys(contact_index_E, 'R'))
                    if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode3)==True:
                        return True
                elif action == 'SlideF1+':
                    contact_mode3 = dict({'F1':'S-', 'F2': 'B'}, **dict.fromkeys(contact_index_E, 'R'))
                    if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode3)==True:
                        return True
                elif action == 'SlideF2+':
                    contact_mode3 = dict({'F1':'B', 'F2': 'S+'}, **dict.fromkeys(contact_index_E, 'R'))
                    if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode3)==True:
                        return True
                elif action == 'SlideF2-':
                    contact_mode3 = dict({'F1':'B', 'F2': 'S-'}, **dict.fromkeys(contact_index_E, 'R'))
                    if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode3)==True:
                        return True
        elif action == 'Slide+PivotF1CW' or action == 'Slide+PivotF1CCW' or action == 'Slide-PivotF1CW' or action == 'Slide-PivotF1CCW' or action == 'Slide+PivotF2CW' or action == 'Slide+PivotF2CCW' or action == 'Slide-PivotF2CW' or action == 'Slide-PivotF2CCW':
            F1_contact_normal = contact_normal_list[contact_index.index('F1')]
            F2_contact_normal = contact_normal_list[contact_index.index('F2')]
            if action == 'Slide+PivotF1CW' or action == 'Slide+PivotF1CCW':
                contact_mode3 = dict({'F1':'S+', 'F2': 'B'}, **dict.fromkeys(contact_index_E, 'R'))
                if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode3)==True:
                    return True
            elif action == 'Slide-PivotF1CW' or action == 'Slide-PivotF1CCW':
                contact_mode3 = dict({'F1':'S-', 'F2': 'B'}, **dict.fromkeys(contact_index_E, 'R'))
                if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode3)==True:
                    return True
            elif action == 'Slide+PivotF2CW' or action == 'Slide+PivotF2CCW':
                contact_mode3 = dict({'F1':'B', 'F2': 'S+'}, **dict.fromkeys(contact_index_E, 'R'))
                if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode3)==True:
                    return True
            elif action == 'Slide-PivotF2CW' or action == 'Slide-PivotF2CCW':
                contact_mode3 = dict({'F1':'B', 'F2': 'S-'}, **dict.fromkeys(contact_index_E, 'R'))
                if whether_stable_simple(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, contact_mode3)==True:
                    return True
    return False

def finger_position_after_action(current_vertex_position, current_gripper_pose, action, profile_environment, ini_vertex_position, ini_CoM_position, accuracy=0.01):
    current_F1 = [current_gripper_pose[0]-current_gripper_pose[2]/2.0*cos((current_gripper_pose[3]-90)*pi/180), current_gripper_pose[1]-current_gripper_pose[2]/2.0*sin((current_gripper_pose[3]-90)*pi/180)]
    current_F2 = [current_gripper_pose[0]+current_gripper_pose[2]/2.0*cos((current_gripper_pose[3]-90)*pi/180), current_gripper_pose[1]+current_gripper_pose[2]/2.0*sin((current_gripper_pose[3]-90)*pi/180)]
    target_gripper_pose_set=[]
    action_feasible=True       
    if action == 'PivotF1CCW' or action == 'PivotF1CW' or action == 'PivotF2CCW' or action == 'PivotF2CW':
        if 'F1' in action:
            pole = current_F1
        else:
            pole = current_F2
        if 'CCW' in action:
            rotate_direction = 'CCW'
        else:
            rotate_direction = 'CW'
        if 'F1CCW' in action:
            if Polygon(current_vertex_position).distance(Point(point_position_after_rotation(current_F2, current_F1, 1)))==0:
                return []
        elif 'F1CW' in action:
            if Polygon(current_vertex_position).distance(Point(point_position_after_rotation(current_F2, current_F1, -1)))==0:
                return []
        elif 'F2CCW' in action:
            if Polygon(current_vertex_position).distance(Point(point_position_after_rotation(current_F1, current_F2, 1)))==0:
                return []
        elif 'F2CW' in action:
            if Polygon(current_vertex_position).distance(Point(point_position_after_rotation(current_F1, current_F2, -1)))==0:
                return []
        if 'F1' in action:
            target_F1 = current_F1
            current_angle = inclination_angle(current_F1,current_F2)
        elif 'F2' in action:
            target_F2 = current_F2
            current_angle = inclination_angle(current_F2,current_F1)
        F1_F2_length = LineString([current_F1, current_F2]).length
        if 'F1' in action:
            intersection_circle_object = Point(current_F1).buffer(F1_F2_length).boundary.intersection(LinearRing(current_vertex_position))
        elif 'F2' in action:
            intersection_circle_object = Point(current_F2).buffer(F1_F2_length).boundary.intersection(LinearRing(current_vertex_position))
        intersection_circle_object = [list(point.coords[0]) for point in intersection_circle_object.geoms]
        intersection_circle_object_copy = intersection_circle_object[:]
        if 'F1' in action:
            for point in intersection_circle_object_copy:
                if Point(point).distance(Point(current_F2))<0.02:
                    intersection_circle_object.remove(point)
        elif 'F2' in action:
            for point in intersection_circle_object_copy:
                if Point(point).distance(Point(current_F1))<0.02:
                    intersection_circle_object.remove(point)
        if 'F1CCW' in action:
            target_rotation_angle_set = [(inclination_angle(current_F1,point)-current_angle)%(2*pi) for point in intersection_circle_object]
            target_rotation_angle = min(target_rotation_angle_set)
            target_F2 = intersection_circle_object[target_rotation_angle_set.index(target_rotation_angle)]
            effector_trajectory = point_position_after_rotation_batch(current_F2, current_F1, np.arange(1*pi/180, target_rotation_angle-0.2*pi/180, 1*pi/180)*180/pi)
        elif 'F1CW' in action:
            target_rotation_angle_set = [(inclination_angle(current_F1,point)-current_angle)%(-2*pi) for point in intersection_circle_object]
            target_rotation_angle = max(target_rotation_angle_set)
            target_F2 = intersection_circle_object[target_rotation_angle_set.index(target_rotation_angle)]
            effector_trajectory = point_position_after_rotation_batch(current_F2, current_F1, np.arange(-1*pi/180, target_rotation_angle+0.2*pi/180, -1*pi/180)*180/pi)
        elif 'F2CCW' in action:
            target_rotation_angle_set = [(inclination_angle(current_F2,point)-current_angle)%(2*pi) for point in intersection_circle_object]
            target_rotation_angle = min(target_rotation_angle_set)
            target_F1 = intersection_circle_object[target_rotation_angle_set.index(target_rotation_angle)]
            effector_trajectory = point_position_after_rotation_batch(current_F1, current_F2, np.arange(1*pi/180, target_rotation_angle-0.2*pi/180, 1*pi/180)*180/pi)
        elif 'F2CW' in action:
            target_rotation_angle_set = [(inclination_angle(current_F2,point)-current_angle)%(-2*pi) for point in intersection_circle_object]
            target_rotation_angle = max(target_rotation_angle_set)
            target_F1 = intersection_circle_object[target_rotation_angle_set.index(target_rotation_angle)]
            effector_trajectory = point_position_after_rotation_batch(current_F1, current_F2, np.arange(-1*pi/180, target_rotation_angle+0.2*pi/180, -1*pi/180)*180/pi)
        if MultiPoint(effector_trajectory).distance(Polygon(current_vertex_position))==0 or MultiPoint(effector_trajectory).distance(LineString(profile_environment))<0.02:
            return []
        cost = LineString([current_F1, current_F2]).length*abs(target_rotation_angle)+1
        aperture = Point(target_F1).distance(Point(target_F2))
        center_position = [(target_F1[0]+target_F2[0])/2, (target_F1[1]+target_F2[1])/2]
        angle = inclination_angle(target_F1,target_F2)*180/pi+90
        target_gripper_pose_set.append([(center_position[0], center_position[1], aperture, angle), cost, [['rotate', pole, rotate_direction, abs(target_rotation_angle*180/pi)]]]) 
    elif action == 'SlideF1+' or action == 'SlideF1-' or action == 'SlideF2+' or action == 'SlideF2-':
        min_distance = float('inf')
        temp_contact_index, temp_contact_list, temp_contact_normal_list, temp_contact_local_tangent_list = contact_points_and_normal(current_vertex_position, current_F1, current_F2, profile_environment)
        if action == 'SlideF1+' or action == 'SlideF1-':
            for i in range(len(current_vertex_position)):
                if Point(current_F1).distance(LineString([current_vertex_position[i],current_vertex_position[(i+1)%(len(current_vertex_position))]]))<min_distance:
                    contact_edge = [current_vertex_position[i], current_vertex_position[(i+1)%(len(current_vertex_position))]]
                    min_distance = Point(current_F1).distance(LineString([current_vertex_position[i],current_vertex_position[(i+1)%(len(current_vertex_position))]]))
            current_F1 = footCtoAB(contact_edge[0],contact_edge[1],current_F1)
            contact_edge_length = LineString(contact_edge).length
            slide_direction = [(contact_edge[1][0]-contact_edge[0][0])/contact_edge_length, (contact_edge[1][1]-contact_edge[0][1])/contact_edge_length]
            contact_normal_F1 = temp_contact_normal_list[temp_contact_index.index('F1')]
            if action == 'SlideF1+' and np.cross(contact_normal_F1, slide_direction)>0:
                slide_direction = [-slide_direction[0], -slide_direction[1]]
            elif action == 'SlideF1-' and np.cross(contact_normal_F1, slide_direction)<0:
                slide_direction = [-slide_direction[0], -slide_direction[1]]
        else:
            for i in range(len(current_vertex_position)):
                if Point(current_F2).distance(LineString([current_vertex_position[i],current_vertex_position[(i+1)%(len(current_vertex_position))]]))<min_distance:
                    contact_edge = [current_vertex_position[i], current_vertex_position[(i+1)%(len(current_vertex_position))]]
                    min_distance = Point(current_F2).distance(LineString([current_vertex_position[i],current_vertex_position[(i+1)%(len(current_vertex_position))]]))
            current_F2 = footCtoAB(contact_edge[0],contact_edge[1],current_F2)
            contact_edge_length = LineString(contact_edge).length
            slide_direction = [(contact_edge[1][0]-contact_edge[0][0])/contact_edge_length, (contact_edge[1][1]-contact_edge[0][1])/contact_edge_length]
            contact_normal_F2 = temp_contact_normal_list[temp_contact_index.index('F2')]
            if action == 'SlideF2+' and np.cross(contact_normal_F2, slide_direction)>0:
                slide_direction = [-slide_direction[0], -slide_direction[1]]
            elif action == 'SlideF2-' and np.cross(contact_normal_F2, slide_direction)<0:
                slide_direction = [-slide_direction[0], -slide_direction[1]]
        temp0_contact_index = temp_contact_index[:]
        temp0_contact_list = temp_contact_list[:]
        temp0_contact_normal_list = temp_contact_normal_list[:]
        temp0_contact_local_tangent_list = temp_contact_local_tangent_list[:]
        for i in range(len(temp0_contact_index)):
            if temp0_contact_index[i] == 'F1' or temp0_contact_index[i] == 'F2':
                temp_contact_index.remove(temp0_contact_index[i])
                temp_contact_list.remove(temp0_contact_list[i])
                temp_contact_normal_list.remove(temp0_contact_normal_list[i])
                temp_contact_local_tangent_list.remove(temp0_contact_local_tangent_list[i])

        max_sliding_distance = 1.6
        min_sliding_distance = 0
        i = 0
        while i < 5:
            target_F1 = [current_F1[0]+max_sliding_distance*slide_direction[0], current_F1[1]+max_sliding_distance*slide_direction[1]]
            target_F2 = [current_F2[0]+max_sliding_distance*slide_direction[0], current_F2[1]+max_sliding_distance*slide_direction[1]]
            temp_contact_index_finger, temp_contact_list_finger, temp_contact_normal_list_finger, temp_contact_local_tangent_list_finger = contact_points_and_normal_finger(current_vertex_position, target_F1, target_F2)
            contact_index = temp_contact_index+temp_contact_index_finger
            contact_list = temp_contact_list+temp_contact_list_finger
            contact_normal_list = temp_contact_normal_list + temp_contact_normal_list_finger
            contact_local_tangent_list = temp_contact_local_tangent_list + temp_contact_local_tangent_list_finger
            current_CoM = calculate_current_CoM(ini_vertex_position, current_vertex_position, ini_CoM_position)
            if whether_AdjustGraspAction_feasible(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, action, target_F1, target_F2)==True:
                if max_sliding_distance == 1.6:
                    break
                delta_distance = max_sliding_distance-min_sliding_distance
                min_sliding_distance = max_sliding_distance
                max_sliding_distance = max_sliding_distance+delta_distance/2
            else:
                delta_distance = max_sliding_distance-min_sliding_distance
                max_sliding_distance = min_sliding_distance+delta_distance/2
            i+=1
        for sliding_length in np.arange(0.05, max_sliding_distance, 0.05):
            target_F1 = [current_F1[0]+sliding_length*slide_direction[0], current_F1[1]+sliding_length*slide_direction[1]]
            target_F2 = [current_F2[0]+sliding_length*slide_direction[0], current_F2[1]+sliding_length*slide_direction[1]]
            cost = sliding_length+1
            if Feasible_Grasp(current_vertex_position, target_F1, target_F2, profile_environment)==False:
                break
            aperture = Point(target_F1).distance(Point(target_F2))
            center_position = [(target_F1[0]+target_F2[0])/2, (target_F1[1]+target_F2[1])/2]
            angle = inclination_angle(target_F1,target_F2)*180/pi+90
            target_gripper_pose_set.append([(center_position[0], center_position[1], aperture, angle), cost, [['slide', slide_direction, sliding_length]]]) 
    elif action == 'Slide+PivotF1CW' or action == 'Slide+PivotF1CCW' or action == 'Slide-PivotF1CW' or action == 'Slide-PivotF1CCW' or action == 'Slide+PivotF2CW' or action == 'Slide+PivotF2CCW' or action == 'Slide-PivotF2CW' or action == 'Slide-PivotF2CCW':
        min_distance = float('inf')
        temp_contact_index, temp_contact_list, temp_contact_normal_list, temp_contact_local_tangent_list = contact_points_and_normal(current_vertex_position, current_F1, current_F2, profile_environment)
        if ('+' in action and 'F1' in action) or ('-' in action and 'F1' in action):
            for i in range(len(current_vertex_position)):
                if Point(current_F1).distance(LineString([current_vertex_position[i],current_vertex_position[(i+1)%(len(current_vertex_position))]]))<min_distance:
                    contact_edge = [current_vertex_position[i], current_vertex_position[(i+1)%(len(current_vertex_position))]]
                    min_distance = Point(current_F1).distance(LineString([current_vertex_position[i],current_vertex_position[(i+1)%(len(current_vertex_position))]]))
            current_F1 = footCtoAB(contact_edge[0],contact_edge[1],current_F1)
            contact_edge_length = LineString(contact_edge).length
            slide_direction = [(contact_edge[1][0]-contact_edge[0][0])/contact_edge_length, (contact_edge[1][1]-contact_edge[0][1])/contact_edge_length]
            contact_normal_F1 = temp_contact_normal_list[temp_contact_index.index('F1')]
            if ('+' in action and 'F1' in action) and np.cross(contact_normal_F1, slide_direction)>0:
                slide_direction = [-slide_direction[0], -slide_direction[1]]
            elif ('-' in action and 'F1' in action) and np.cross(contact_normal_F1, slide_direction)<0:
                slide_direction = [-slide_direction[0], -slide_direction[1]]
        else:
            for i in range(len(current_vertex_position)):
                if Point(current_F2).distance(LineString([current_vertex_position[i],current_vertex_position[(i+1)%(len(current_vertex_position))]]))<min_distance:
                    contact_edge = [current_vertex_position[i], current_vertex_position[(i+1)%(len(current_vertex_position))]]
                    min_distance = Point(current_F2).distance(LineString([current_vertex_position[i],current_vertex_position[(i+1)%(len(current_vertex_position))]]))
            current_F2 = footCtoAB(contact_edge[0],contact_edge[1],current_F2)
            contact_edge_length = LineString(contact_edge).length
            slide_direction = [(contact_edge[1][0]-contact_edge[0][0])/contact_edge_length, (contact_edge[1][1]-contact_edge[0][1])/contact_edge_length]
            contact_normal_F2 = temp_contact_normal_list[temp_contact_index.index('F2')]
            if ('+' in action and 'F2' in action) and np.cross(contact_normal_F2, slide_direction)>0:
                slide_direction = [-slide_direction[0], -slide_direction[1]]
            elif ('-' in action and 'F2' in action) and np.cross(contact_normal_F2, slide_direction)<0:
                slide_direction = [-slide_direction[0], -slide_direction[1]]
        temp0_contact_index = temp_contact_index[:]
        temp0_contact_list = temp_contact_list[:]
        temp0_contact_normal_list = temp_contact_normal_list[:]
        temp0_contact_local_tangent_list = temp_contact_local_tangent_list[:]
        for i in range(len(temp0_contact_index)):
            if temp0_contact_index[i] == 'F1' or temp0_contact_index[i] == 'F2':
                temp_contact_index.remove(temp0_contact_index[i])
                temp_contact_list.remove(temp0_contact_list[i])
                temp_contact_normal_list.remove(temp0_contact_normal_list[i])
                temp_contact_local_tangent_list.remove(temp0_contact_local_tangent_list[i])

        max_sliding_distance = 1.6
        min_sliding_distance = 0
        i = 0
        while i < 5:
            target_F1 = [current_F1[0]+max_sliding_distance*slide_direction[0], current_F1[1]+max_sliding_distance*slide_direction[1]]
            target_F2 = [current_F2[0]+max_sliding_distance*slide_direction[0], current_F2[1]+max_sliding_distance*slide_direction[1]]
            temp_contact_index_finger, temp_contact_list_finger, temp_contact_normal_list_finger, temp_contact_local_tangent_list_finger = contact_points_and_normal_finger(current_vertex_position, target_F1, target_F2)
            contact_index = temp_contact_index+temp_contact_index_finger
            contact_list = temp_contact_list+temp_contact_list_finger
            contact_normal_list = temp_contact_normal_list + temp_contact_normal_list_finger
            contact_local_tangent_list = temp_contact_local_tangent_list + temp_contact_local_tangent_list_finger
            current_CoM = calculate_current_CoM(ini_vertex_position, current_vertex_position, ini_CoM_position)
            if whether_AdjustGraspAction_feasible(contact_index, contact_list, contact_normal_list, contact_local_tangent_list, current_CoM, action, target_F1, target_F2)==True:
                if max_sliding_distance == 1.6:
                    break
                delta_distance = max_sliding_distance-min_sliding_distance
                min_sliding_distance = max_sliding_distance
                max_sliding_distance = max_sliding_distance+delta_distance/2
            else:
                delta_distance = max_sliding_distance-min_sliding_distance
                max_sliding_distance = min_sliding_distance+delta_distance/2
            i+=1
        for sliding_length in np.arange(0.05, max_sliding_distance, 0.05):
            current_F1_new = [current_F1[0]+sliding_length*slide_direction[0], current_F1[1]+sliding_length*slide_direction[1]]
            current_F2_new = [current_F2[0]+sliding_length*slide_direction[0], current_F2[1]+sliding_length*slide_direction[1]]
            if 'F1' in action:
                pole = current_F1_new
            else:
                pole = current_F2_new
            if 'CCW' in action:
                rotate_direction = 'CCW'
            else:
                rotate_direction = 'CW'
            if Feasible_Grasp(current_vertex_position, current_F1_new, current_F2_new, profile_environment)==False:
                break
            if 'F1CCW' in action:
                if Polygon(current_vertex_position).distance(Point(point_position_after_rotation(current_F2_new, current_F1_new, 1)))==0:
                    continue
            elif 'F1CW' in action:
                if Polygon(current_vertex_position).distance(Point(point_position_after_rotation(current_F2_new, current_F1_new, -1)))==0:
                    continue
            elif 'F2CCW' in action:
                if Polygon(current_vertex_position).distance(Point(point_position_after_rotation(current_F1_new, current_F2_new, 1)))==0:
                    continue
            elif 'F2CW' in action:
                if Polygon(current_vertex_position).distance(Point(point_position_after_rotation(current_F1_new, current_F2_new, -1)))==0:
                    continue
            if 'F1' in action:
                target_F1 = current_F1_new
                current_angle = inclination_angle(current_F1_new,current_F2_new)
            elif 'F2' in action:
                target_F2 = current_F2_new
                current_angle = inclination_angle(current_F2_new,current_F1_new)
            F1_F2_length = LineString([current_F1_new, current_F2_new]).length
            if 'F1' in action:
                intersection_circle_object = Point(current_F1_new).buffer(F1_F2_length).boundary.intersection(LinearRing(current_vertex_position))
            elif 'F2' in action:
                intersection_circle_object = Point(current_F2_new).buffer(F1_F2_length).boundary.intersection(LinearRing(current_vertex_position))
            intersection_circle_object = [list(point.coords[0]) for point in intersection_circle_object.geoms]
            intersection_circle_object_copy = intersection_circle_object[:]
            if 'F1' in action:
                for point in intersection_circle_object_copy:
                    if Point(point).distance(Point(current_F2_new))<0.02:
                        intersection_circle_object.remove(point)
            elif 'F2' in action:
                for point in intersection_circle_object_copy:
                    if Point(point).distance(Point(current_F1_new))<0.02:
                        intersection_circle_object.remove(point)
            if 'F1CCW' in action:
                target_rotation_angle_set = [(inclination_angle(current_F1_new,point)-current_angle)%(2*pi) for point in intersection_circle_object]
                target_rotation_angle = min(target_rotation_angle_set)
                target_F2 = intersection_circle_object[target_rotation_angle_set.index(target_rotation_angle)]
                effector_trajectory = point_position_after_rotation_batch(current_F2_new, current_F1_new, np.arange(1*pi/180, target_rotation_angle-0.2*pi/180, 1*pi/180)*180/pi)
            elif 'F1CW' in action:
                target_rotation_angle_set = [(inclination_angle(current_F1_new,point)-current_angle)%(-2*pi) for point in intersection_circle_object]
                target_rotation_angle = max(target_rotation_angle_set)
                target_F2 = intersection_circle_object[target_rotation_angle_set.index(target_rotation_angle)]
                effector_trajectory = point_position_after_rotation_batch(current_F2_new, current_F1_new, np.arange(-1*pi/180, target_rotation_angle+0.2*pi/180, -1*pi/180)*180/pi)
            elif 'F2CCW' in action:
                target_rotation_angle_set = [(inclination_angle(current_F2_new,point)-current_angle)%(2*pi) for point in intersection_circle_object]
                target_rotation_angle = min(target_rotation_angle_set)
                target_F1 = intersection_circle_object[target_rotation_angle_set.index(target_rotation_angle)]
                effector_trajectory = point_position_after_rotation_batch(current_F1_new, current_F2_new, np.arange(1*pi/180, target_rotation_angle-0.2*pi/180, 1*pi/180)*180/pi)
            elif 'F2CW' in action:
                target_rotation_angle_set = [(inclination_angle(current_F2_new,point)-current_angle)%(-2*pi) for point in intersection_circle_object]
                target_rotation_angle = max(target_rotation_angle_set)
                target_F1 = intersection_circle_object[target_rotation_angle_set.index(target_rotation_angle)]
                effector_trajectory = point_position_after_rotation_batch(current_F1_new, current_F2_new, np.arange(-1*pi/180, target_rotation_angle+0.2*pi/180, -1*pi/180)*180/pi)
            if MultiPoint(effector_trajectory).distance(Polygon(current_vertex_position))==0 or MultiPoint(effector_trajectory).distance(LineString(profile_environment))<0.02:
                continue
            cost = sliding_length+LineString([current_F1_new, current_F2_new]).length*abs(target_rotation_angle)+2
            aperture = Point(target_F1).distance(Point(target_F2))
            center_position = [(target_F1[0]+target_F2[0])/2, (target_F1[1]+target_F2[1])/2]
            angle = inclination_angle(target_F1,target_F2)*180/pi+90
            target_gripper_pose_set.append([(center_position[0], center_position[1], aperture, angle), cost, [['slide', slide_direction, sliding_length], ['rotate', pole, rotate_direction, abs(target_rotation_angle*180/pi)]]]) 
    return target_gripper_pose_set  


#time0=time.time()
#print finger_position_after_action([[0.5,0], [1.2,0], [1.2,1.5], [0.5,1.5]], (0.85, 0.9935, 0.7998556119700605, 118.93632518370461), 'PivotF1CW', [[0,0], [3,0]], [[0,0], [0.7,0], [0.7,1.5], [0,1.5]], [0.35,0.75], accuracy=0.01)
#print time.time()-time0       

'''
target_F1=[0.5,0.8]
target_F2=[1.2, 1.187]
aperture = Point(target_F1).distance(Point(target_F2))
center_position = [(target_F1[0]+target_F2[0])/2, (target_F1[1]+target_F2[1])/2]
angle = inclination_angle(target_F1,target_F2)*180/pi+90
print center_position[0], center_position[1], aperture, angle
'''

def gripper_profile(current_gripper_pose):
    current_F1 = [current_gripper_pose[0]-current_gripper_pose[2]/2.0*cos((current_gripper_pose[3]-90)*pi/180), current_gripper_pose[1]-current_gripper_pose[2]/2.0*sin((current_gripper_pose[3]-90)*pi/180)]
    current_F2 = [current_gripper_pose[0]+current_gripper_pose[2]/2.0*cos((current_gripper_pose[3]-90)*pi/180), current_gripper_pose[1]+current_gripper_pose[2]/2.0*sin((current_gripper_pose[3]-90)*pi/180)]
    return current_F1, current_F2, LineString([current_F1, current_F2])

def gripper_pose_from_profile(gripper_profile):
    F1_position = gripper_profile.coords[:][0]
    F2_position = gripper_profile.coords[:][1]
    aperture = Point(F1_position).distance(Point(F2_position))
    center_position = [(F1_position[0]+F2_position[0])/2, (F1_position[1]+F2_position[1])/2]
    angle = inclination_angle(F1_position,F2_position)*180/pi+90
    return (center_position[0], center_position[1], aperture, angle)

def action_trajectory(current_gripper_pose, parameter_set):
    trajectory = [current_gripper_pose]
    _0, _1, current_gripper_profile = gripper_profile(current_gripper_pose)
    for parameter in parameter_set:
        if parameter[0]=='rotate':
            for angle in np.hstack((np.arange(0, parameter[3], 5),parameter[3])): 
                if parameter[2] == 'CW':
                    new_gripper_profile = rotate(current_gripper_profile, -angle, origin=Point(parameter[1]))
                else:
                    new_gripper_profile = rotate(current_gripper_profile, angle, origin=Point(parameter[1]))
                trajectory.append(gripper_pose_from_profile(new_gripper_profile))
            current_gripper_profile = new_gripper_profile
        elif parameter[0]=='slide':
            for distance in np.hstack((np.arange(0, parameter[2], 0.1),parameter[2])): 
                new_gripper_profile = translate(current_gripper_profile, xoff=distance*parameter[1][0], yoff=distance*parameter[1][1])
                trajectory.append(gripper_pose_from_profile(new_gripper_profile))
            current_gripper_profile = new_gripper_profile
    return trajectory
