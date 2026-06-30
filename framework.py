#!/usr/bin/env python
import os
import sys
import time
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import gol
from general_functions import *
from top_and_middle_layer import *
from math import *
from jerk import *
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import LinearRing
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

accuracy = 0.004

def object_pose_plan_consider_move_in_air(initial_object_vertex_position, target_object_vertex_position, action_sequence, contact_state_trajectory, environment, repeating_time):
    initial_contact_state = obtain_contact_state(initial_object_vertex_position, environment)
    target_contact_state = obtain_contact_state(target_object_vertex_position, environment)
    if initial_contact_state == ['no_contact']:
        for direction in [[0,-1], [-1,0], [0,1], [1,0]]:
            for i in np.arange(0,5,0.01):
                temp_new_vertex_position = Move_in_air_next_vertex_position(initial_object_vertex_position, [direction, i], environment)
                new_contact_state = obtain_contact_state(temp_new_vertex_position, environment)
                if new_contact_state != ['no_contact']:
                    contact_state_after_ini = new_contact_state
                    action_achieve_after_ini = 'Move-in-air'
                    action_parameter_after_ini = [direction, i]
                    object_pose_after_ini = temp_new_vertex_position
                    break
            else:
                continue
            break
    if target_contact_state == ['no_contact']:
        for direction in [[0,-1], [-1,0], [0,1], [1,0]]:
            for i in np.arange(0,5,0.01):
                temp_new_vertex_position = Move_in_air_next_vertex_position(target_object_vertex_position, [direction, i], environment)
                new_contact_state = obtain_contact_state(temp_new_vertex_position, environment)
                if new_contact_state != ['no_contact']:
                    contact_state_before_end= new_contact_state
                    action_achieve_before_end = 'Move-in-air'
                    action_parameter_before_end = [[-direction[0], -direction[1]], i]
                    object_pose_before_end = temp_new_vertex_position
                    break
            else:
                continue
            break
    if initial_contact_state != ['no_contact'] and target_contact_state != ['no_contact']:
        action_parameter_sequence, object_pose_trajectory = object_pose_plan(initial_object_vertex_position, None, action_sequence, contact_state_trajectory, environment, repeating_time)
        last_action_sequence, last_contact_state_trajectory, last_action_parameter_sequence, last_object_pose_trajectory = object_pose_plan_within_contact_state(object_pose_trajectory[-1], target_object_vertex_position, environment)
        action_sequence+=last_action_sequence
        contact_state_trajectory+=last_contact_state_trajectory
        action_parameter_sequence+=last_action_parameter_sequence
        object_pose_trajectory+= last_object_pose_trajectory
    elif initial_contact_state == ['no_contact'] and target_contact_state == ['no_contact']: 
        action_parameter_sequence, object_pose_trajectory = object_pose_plan(object_pose_after_ini, None, action_sequence, contact_state_trajectory, environment, repeating_time)
        last_action_sequence, last_contact_state_trajectory, last_action_parameter_sequence, last_object_pose_trajectory = object_pose_plan_within_contact_state(object_pose_trajectory[-1], object_pose_before_end, environment)
        action_sequence+=last_action_sequence
        contact_state_trajectory+=last_contact_state_trajectory
        action_parameter_sequence+=last_action_parameter_sequence
        object_pose_trajectory+= last_object_pose_trajectory  
        action_sequence = ['Move-in-air']+action_sequence+['Move-in-air']
        contact_state_trajectory = [contact_state_after_ini]+contact_state_trajectory+[['no_contact']]
        object_pose_trajectory = [object_pose_after_ini] + object_pose_trajectory + [target_object_vertex_position]
        action_parameter_sequence = [action_parameter_after_ini] + action_parameter_sequence + [action_parameter_before_end]   
    elif initial_contact_state == ['no_contact'] and target_contact_state != ['no_contact']: 
        action_parameter_sequence, object_pose_trajectory = object_pose_plan(object_pose_after_ini, None, action_sequence, contact_state_trajectory, environment, repeating_time)
        last_action_sequence, last_contact_state_trajectory, last_action_parameter_sequence, last_object_pose_trajectory = object_pose_plan_within_contact_state(object_pose_trajectory[-1], target_object_vertex_position, environment)
        action_sequence+=last_action_sequence
        contact_state_trajectory+=last_contact_state_trajectory
        action_parameter_sequence+=last_action_parameter_sequence
        object_pose_trajectory+= last_object_pose_trajectory 
        action_sequence = ['Move-in-air']+action_sequence
        contact_state_trajectory = [contact_state_after_ini]+contact_state_trajectory 
        object_pose_trajectory = [object_pose_after_ini] + object_pose_trajectory
        action_parameter_sequence = [action_parameter_after_ini] + action_parameter_sequence     
    elif initial_contact_state != ['no_contact'] and target_contact_state == ['no_contact']: 
        action_parameter_sequence, object_pose_trajectory = object_pose_plan(initial_object_vertex_position, None, action_sequence, contact_state_trajectory, environment, repeating_time)
        last_action_sequence, last_contact_state_trajectory, last_action_parameter_sequence, last_object_pose_trajectory = object_pose_plan_within_contact_state(object_pose_trajectory[-1], object_pose_before_end, environment)
        action_sequence+=last_action_sequence
        contact_state_trajectory+=last_contact_state_trajectory
        action_parameter_sequence+=last_action_parameter_sequence
        object_pose_trajectory+= last_object_pose_trajectory
        action_sequence = action_sequence+['Move-in-air']
        contact_state_trajectory = contact_state_trajectory+[['no_contact']]  
        object_pose_trajectory = object_pose_trajectory + [target_object_vertex_position]
        action_parameter_sequence = action_parameter_sequence + [action_parameter_before_end]   
    return action_sequence, contact_state_trajectory, action_parameter_sequence, object_pose_trajectory

def framework(initial_object_vertex_position, initial_CoM_position, ini_gripper_pose, target_object_vertex_position, target_finger_position, environment, gripper_type, mu_gripper, mu_environment):
    if gripper_type == 'Type 1':
        bottom_layer=__import__('bottom_layer_gripper1')
    else:
        bottom_layer=__import__('bottom_layer_gripper2')
    gol.set_value('mu_environment', mu_environment)
    gol.set_value('mu_gripper', mu_gripper)
    feasible_fcp_set = bottom_layer.feasible_finger_contact_pair_set(initial_object_vertex_position)
    action_sequence_memory = []
    contact_state_trajectory_memory = []
    initial_contact_state = obtain_contact_state(initial_object_vertex_position, environment)
    target_contact_state = obtain_contact_state(target_object_vertex_position, environment)
    fcp_set_index_memory = dict()
    gripper_pose_set_index_memory=dict()
    if initial_contact_state == ['no_contact']:
        for direction in [[0,-1], [-1,0], [0,1], [1,0]]:
            for i in np.arange(0,5,0.01):
                temp_new_vertex_position = Move_in_air_next_vertex_position(initial_object_vertex_position, [direction, i], environment)
                new_contact_state = obtain_contact_state(temp_new_vertex_position, environment)
                if new_contact_state != ['no_contact']:
                    contact_state_after_ini = new_contact_state
                    action_achieve_after_ini = 'Move-in-air'
                    action_parameter_after_ini = [direction, i]
                    object_pose_after_ini = temp_new_vertex_position
                    break
            else:
                continue
            break
    if target_contact_state == ['no_contact']:
        for direction in [[0,-1], [-1,0], [0,1], [1,0]]:
            for i in np.arange(0,5,0.01):
                temp_new_vertex_position = Move_in_air_next_vertex_position(target_object_vertex_position, [direction, i], environment)
                new_contact_state = obtain_contact_state(temp_new_vertex_position, environment)
                if new_contact_state != ['no_contact']:
                    contact_state_before_end= new_contact_state
                    action_achieve_before_end = 'Move-in-air'
                    action_parameter_before_end = [[-direction[0], -direction[1]], i]
                    object_pose_before_end = temp_new_vertex_position
                    break
            else:
                continue
            break
                    
    top_layer_iteration = 0
    time_x=time.time()
    contact_state0 = contact_state(environment, initial_object_vertex_position)
    print(time.time()-time_x)
    while top_layer_iteration<30:
        if top_layer_iteration != 0:
            contact_state_trajectory_temp = [initial_contact_state]+contact_state_trajectory
            for i in range(len(contact_state_trajectory_temp)-1):
                for contact_state_key, contact_state_cost in contact_state0.action_cost.items():
                    if tuple(sorted(contact_state_trajectory_temp[i]))==tuple(sorted(contact_state_key)):
                        for j in range(len(contact_state_cost)):
                            if tuple(sorted(contact_state0.next_contact_state[contact_state_key][j]))==tuple(sorted(contact_state_trajectory_temp[i+1])):
                                contact_state0.action_cost[contact_state_key][j]+=0.1
        if initial_contact_state != ['no_contact'] and target_contact_state != ['no_contact']:
            action_sequence, contact_state_trajectory = contact_state0.contact_state_plan(initial_contact_state, target_contact_state)
        elif initial_contact_state == ['no_contact'] and target_contact_state == ['no_contact']:
            action_sequence, contact_state_trajectory = contact_state0.contact_state_plan(contact_state_after_ini, contact_state_before_end)
        elif initial_contact_state == ['no_contact'] and target_contact_state != ['no_contact']:
            action_sequence, contact_state_trajectory = contact_state0.contact_state_plan(contact_state_after_ini, target_contact_state)
        elif initial_contact_state != ['no_contact'] and target_contact_state == ['no_contact']:
            action_sequence, contact_state_trajectory = contact_state0.contact_state_plan(initial_contact_state, contact_state_before_end)
        if action_sequence in action_sequence_memory and contact_state_trajectory in contact_state_trajectory_memory:
            continue
        #action_sequence=['Push', 'Tilting-slide', 'Push']
        #contact_state_trajectory=[['double_contact', 'l1E3', 'l2E2'], ['V1E3', 'V2E2', 'double_contact'], ['V2E2', 'single_contact']]

        last_big_circ_action_sequence=action_sequence[:]
        action_sequence_memory.append(action_sequence[:])
        last_big_circ_contact_state_trajectory=contact_state_trajectory[:]
        contact_state_trajectory_memory.append(contact_state_trajectory[:])
        
        print(action_sequence)
        print(contact_state_trajectory)

        repeating_time_kind_A = 0                              #repeating_time_kind_A = 0
        repeating_time_kind_B = 0
        while repeating_time_kind_A<2:   #4
            print('repeating_time_kind_A', repeating_time_kind_A)
            print('repeating_time_kind_B', repeating_time_kind_B)
            time0 = time.time()
            if repeating_time_kind_A>0:
                action_sequence = last_big_circ_action_sequence[:]
                contact_state_trajectory = last_big_circ_contact_state_trajectory[:]
            #try:
            try:
                if repeating_time_kind_B==0:
                    action_sequence, contact_state_trajectory, action_parameter_sequence, object_pose_trajectory = object_pose_plan_consider_move_in_air(initial_object_vertex_position, target_object_vertex_position, action_sequence, contact_state_trajectory, environment, repeating_time_kind_A)
                    old_action_sequence=action_sequence[:]
                    old_contact_state_trajectory=contact_state_trajectory[:]
                    old_action_parameter_sequence=action_parameter_sequence[:]
                    old_object_pose_trajectory=object_pose_trajectory[:]                   
                elif repeating_time_kind_B<3: 
                    old_action_sequence=action_sequence[:]
                    old_contact_state_trajectory=contact_state_trajectory[:]
                    old_action_parameter_sequence=action_parameter_sequence[:]
                    old_object_pose_trajectory=object_pose_trajectory[:] 
                    action_sequence = old_action_sequence[0:unsuitable_action_index]+[old_action_sequence[unsuitable_action_index]]+[old_action_sequence[unsuitable_action_index]]+old_action_sequence[(unsuitable_action_index+1)::]
                    contact_state_trajectory = old_contact_state_trajectory[0:unsuitable_action_index]+[old_contact_state_trajectory[unsuitable_action_index]]+[old_contact_state_trajectory[unsuitable_action_index]]+old_contact_state_trajectory[(unsuitable_action_index+1)::]
                    temp_action_parameter_sequence = old_action_parameter_sequence[:]
                    temp_object_pose_trajectory = old_object_pose_trajectory[:]
                    if old_action_sequence[unsuitable_action_index]=='Tip':
                        action_parameter_sequence = temp_action_parameter_sequence[0:unsuitable_action_index]+[[temp_action_parameter_sequence[unsuitable_action_index][0], temp_action_parameter_sequence[unsuitable_action_index][1], temp_action_parameter_sequence[unsuitable_action_index][2]/2.0]]+[[temp_action_parameter_sequence[unsuitable_action_index][0], temp_action_parameter_sequence[unsuitable_action_index][1], temp_action_parameter_sequence[unsuitable_action_index][2]/2.0]]+temp_action_parameter_sequence[(unsuitable_action_index+1)::]
                        object_pose_trajectory = temp_object_pose_trajectory[0:unsuitable_action_index]+[Tip_next_vertex_position(temp_object_pose_trajectory[unsuitable_action_index-1], action_parameter_sequence[unsuitable_action_index], environment)]+temp_object_pose_trajectory[(unsuitable_action_index)::]
                    elif old_action_sequence[unsuitable_action_index]=='Push':
                        action_parameter_sequence = temp_action_parameter_sequence[0:unsuitable_action_index]+[[temp_action_parameter_sequence[unsuitable_action_index][0], temp_action_parameter_sequence[unsuitable_action_index][1]/2.0]]+[[temp_action_parameter_sequence[unsuitable_action_index][0], temp_action_parameter_sequence[unsuitable_action_index][1]/2.0]]+temp_action_parameter_sequence[(unsuitable_action_index+1)::]
                        object_pose_trajectory = temp_object_pose_trajectory[0:unsuitable_action_index]+[Push_next_vertex_position(temp_object_pose_trajectory[unsuitable_action_index-1], action_parameter_sequence[unsuitable_action_index], environment)]+temp_object_pose_trajectory[(unsuitable_action_index)::]
                    elif old_action_sequence[unsuitable_action_index]=='Tilting-slide':
                        action_parameter_sequence = temp_action_parameter_sequence[0:unsuitable_action_index]+[[temp_action_parameter_sequence[unsuitable_action_index][0], temp_action_parameter_sequence[unsuitable_action_index][1]/2.0]]+[[temp_action_parameter_sequence[unsuitable_action_index][0], temp_action_parameter_sequence[unsuitable_action_index][1]/2.0]]+temp_action_parameter_sequence[(unsuitable_action_index+1)::]
                        object_pose_trajectory = temp_object_pose_trajectory[0:unsuitable_action_index]+[Tilting_slide_next_vertex_position(temp_object_pose_trajectory[unsuitable_action_index-1], action_parameter_sequence[unsuitable_action_index], environment)]+temp_object_pose_trajectory[(unsuitable_action_index)::]
                else:
                    repeating_time_kind_B=0
                    repeating_time_kind_A+=1  
                    continue          
            except:
                repeating_time_kind_B=0
                repeating_time_kind_A+=1
                continue 
            print(action_sequence, contact_state_trajectory, action_parameter_sequence, object_pose_trajectory)
            object_pose_trajectory_plus_ini = [initial_object_vertex_position]+object_pose_trajectory
            feasible_fcp_set_list, fcp_set_index_memory, feasible_gripper_pose_set_list, gripper_pose_set_index_memory  = bottom_layer.obtain_feasible_fcp_set_of_each_pose(feasible_fcp_set, object_pose_trajectory_plus_ini, object_pose_trajectory_plus_ini[0], environment, fcp_set_index_memory, gripper_pose_set_index_memory)
            suitable_fcp_list_sequence, suitable_fcp_cost_list_sequence = bottom_layer.obtain_suitable_fcp_set_of_each_action(action_sequence, action_parameter_sequence, object_pose_trajectory_plus_ini, feasible_fcp_set_list, initial_object_vertex_position, initial_CoM_position, environment)
            print(time.time()-time0)

            unsuitable_action_index_set = []
            for i in range(len(action_sequence)):
                if suitable_fcp_list_sequence[i] == []:
                    unsuitable_action_index_set.append(i)
                elif len(suitable_fcp_list_sequence[i]) <20:
                    unsuitable_action_index_set.append(i)
            unsuitable_grasp_dichotomy_not_need = False
            for j in unsuitable_action_index_set:
                if action_sequence[j]=='Move-in-air' or action_sequence[j]=='Push':
                    unsuitable_grasp_dichotomy_not_need = True
                    break
            if unsuitable_grasp_dichotomy_not_need ==True:
                repeating_time_kind_B=0
                repeating_time_kind_A+=1
                continue    
            elif unsuitable_action_index_set !=[]:  
                unsuitable_action_index = min(unsuitable_action_index_set)  
                repeating_time_kind_B+=1
                continue
            gripper_pose_map_moveobject_link_set = bottom_layer.gripper_pose_map_moveobject(action_sequence, action_parameter_sequence, object_pose_trajectory_plus_ini, suitable_fcp_list_sequence, environment)
            gripper_pose_map_adjustgrasp_link_set = bottom_layer.gripper_pose_map_adjustgrasp(object_pose_trajectory_plus_ini, feasible_fcp_set_list, feasible_gripper_pose_set_list, gripper_pose_map_moveobject_link_set, ini_gripper_pose, environment, initial_object_vertex_position, initial_CoM_position)
            result_temp = bottom_layer.A_star_find_action_path(ini_gripper_pose, gripper_pose_map_moveobject_link_set, gripper_pose_map_adjustgrasp_link_set, object_pose_trajectory_plus_ini, feasible_fcp_set_list, initial_CoM_position, environment)
            if result_temp != False:
                gripper_trajectory_node_sequence, action_and_parameter_sequence = result_temp
                print(gripper_trajectory_node_sequence, action_and_parameter_sequence)
                object_pose_trajectory = [initial_object_vertex_position]
                gripper_pose_trajectory = [ini_gripper_pose]
                for i in range(len(action_and_parameter_sequence)):
                    if action_and_parameter_sequence[i][0]=='MoveObject':
                        old_object_pose = object_pose_trajectory[-1]
                        object_pose_trajectory+=bottom_layer.object_trajectory_moveobject(old_object_pose, action_and_parameter_sequence[i][1], action_and_parameter_sequence[i][2], environment)
                        gripper_pose_trajectory+=bottom_layer.gripper_trajectory_moveobject(old_object_pose, gripper_pose_trajectory[-1], action_and_parameter_sequence[i][1], action_and_parameter_sequence[i][2], environment)
                        print(object_pose_trajectory[-1], gripper_pose_trajectory[-1])
                    else:
                        if action_and_parameter_sequence[i][1][0]=='MoveGripper':
                            object_pose_trajectory.append(object_pose_trajectory[-1])
                            gripper_pose_trajectory.append(action_and_parameter_sequence[i][1][1])
                            print(object_pose_trajectory[-1], gripper_pose_trajectory[-1])
                        else:
                            new_gripper_trajectory = bottom_layer.action_trajectory(gripper_pose_trajectory[-1], action_and_parameter_sequence[i][1])
                            object_pose_trajectory+=[object_pose_trajectory[-1]]*len(new_gripper_trajectory)
                            gripper_pose_trajectory+=new_gripper_trajectory
                            print(object_pose_trajectory[-1], gripper_pose_trajectory[-1])
                print(len(object_pose_trajectory), len(gripper_pose_trajectory))
                #print 'object_pose_trajectory', object_pose_trajectory
                #print 'gripper_pose_trajectory', gripper_pose_trajectory
                return object_pose_trajectory, gripper_pose_trajectory
            else:
                repeating_time_kind_B=0
                repeating_time_kind_A+=1
                continue 
            continue         
        top_layer_iteration+=1
    return False           
