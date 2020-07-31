'''
This code is licensed and documented by Teuvonet Technologies. 
Any use of this code, proprietary or personal, needs approval from the company.

'''


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math
import time
import imp
from pandas import *
import pyopencl as cl
import pyopencl.algorithm as algo
import pyopencl.array as pycl_array
import sklearn.metrics as met
from collections import defaultdict
import copy
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def Confidence_KNN(Input_Data, New_Feature_Order, NET_LATTICE, NUM_CLASSES, 
active_centers_orig, active_centers_total_count, active_dominant_count, active_centers_per_class, map_data, radius_map, 
LEARNING_RATES_PHASE_2, target_arr, list_num_feature_featurespaces, total_num_features_all_featurespaces, cumulative_no_features_featurespace, platforms, context, queue):

    #Intial Parameters
    NUM_DATAPOINTS = Input_Data.shape[0] #Number of tuples/datapoints
    NUM_NEURONS_PER_NET = [ i * i for i in NET_LATTICE] # [9,16,25]
    TOTAL_NUM_NEURONS = sum(NUM_NEURONS_PER_NET) # 3*3 + 4*4 + 5*5 = 50 neurons
    NUM_NETS = len(NET_LATTICE) #[3,4,5] = 3
    NUM_LEARNING_RATES = len(LEARNING_RATES_PHASE_2) # Number of Learning Rates
    TOTAL_NUM_FEATURES = len(New_Feature_Order) #Total Number of Features
    NUM_PARTITIONS = len(list_num_feature_featurespaces)
    print("Now doing Phase 3 Details Below\n")
    print("Number of Datapoints :"+str(NUM_DATAPOINTS))
    print("TOTAL_NUM_FEATURES : " + str(TOTAL_NUM_FEATURES))
    print("Total Number of learning rates : " + str(NUM_LEARNING_RATES))
    print("Number of Partitions : "+str(NUM_PARTITIONS))
    print("cumulative_no_features_featurespace : "+str(cumulative_no_features_featurespace))
    
    

    #Reading the kernel File
    cl_filename = "./confidence_trail.cl" #Reading the file
    with open(cl_filename, 'r') as fd:
	clstr = fd.read()
    #context = cl.create_some_context()
    #queue = cl.CommandQueue(context)
    mem_flags = cl.mem_flags
    myFloatType = np.float32 #Global Float Type for Kernel
    myIntType = np.int32 #Global int Type for Kernel
    program = cl.Program(context, clstr).build()
    # KERNEL BUFFER ARRAYS



    #KERNEL BUFFER ARRAYS
    #----------------------------------------------------------------------------------------------------------------------
    #Input Data
    Input_Data = Input_Data.iloc[:,New_Feature_Order]
    Input_Data = Input_Data.values.ravel() #Taking Input in the order of shuffled columns
    Input_Data = np.array(Input_Data, dtype = myFloatType) #Changing to a single dimensional array
    Input_Data_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = Input_Data)#read buffer

    #Buffer Array to store cumulative sum of distances from each neuron to its feature
    distance_map = np.zeros((NUM_LEARNING_RATES, NUM_PARTITIONS, TOTAL_NUM_NEURONS), dtype = myFloatType)

    #Converting Target array to int
    target_arr_int = np.array(target_arr.values.ravel(), dtype=np.int32)

    #Cumulative Array of features in each partition array
    cumulative_no_features_featurespace = np.array(cumulative_no_features_featurespace, dtype= myIntType)
    cumulative_no_features_featurespace_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = cumulative_no_features_featurespace)#read buffer

    #Number of features in each partition
    NUM_FEATURES_PER_PARTITION = np.array(list_num_feature_featurespaces, dtype = myIntType)
    NUM_FEATURES_PER_PARTITION_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = NUM_FEATURES_PER_PARTITION)#read buffer
    #Array to store minimum distance neuron for each net.
    min_pos = np.array(np.full((NUM_LEARNING_RATES, NUM_PARTITIONS), sys.maxsize), dtype = myIntType)
    min_pos_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = min_pos)#read buffer
    #Array to store minimum distance position of a neuron in each net
    min_array = np.array(np.full((NUM_LEARNING_RATES, NUM_PARTITIONS), sys.maxsize), dtype = myFloatType)
    min_array_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = min_array)#read buffer
    #Buffer Array to store cumulative sum of distances from each neuron to its feature
    distance_map = np.zeros((NUM_LEARNING_RATES, NUM_PARTITIONS, TOTAL_NUM_NEURONS), dtype = myFloatType)
    distance_map_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = distance_map)#read buffer

    #Conversion of int argumnets into fp32
    NUM_DATAPOINTS = np.array(NUM_DATAPOINTS, dtype = myIntType)
    TOTAL_NUM_FEATURES = np.array(TOTAL_NUM_FEATURES, dtype = myIntType)
    TOTAL_NUM_NEURONS = np.array(TOTAL_NUM_NEURONS, dtype = myIntType)
    total_num_features_all_featurespaces = np.array(total_num_features_all_featurespaces, dtype = myIntType)
    NUM_NETS= np.array(NUM_NETS, dtype = myIntType)
    NUM_PARTITIONS = np.array(NUM_PARTITIONS, dtype = myIntType)
    NUM_CLASSES = np.array(NUM_CLASSES, dtype = myIntType)
   
    #Buffer for input argumnets
    
    active_centers_orig_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = active_centers_orig)#read buffer
    active_centers_total_count_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = active_centers_total_count)#read buffer
    active_dominant_count_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = active_dominant_count)#read buffer
    active_centers_per_class_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = active_centers_per_class)#read buffer
    map_data_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = map_data)#read buffer
    radius_map_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = radius_map)#read buffer
    active_dominant_count_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = active_dominant_count)#read buffer
    
    #


    KNN_Winners_ClassWise_Count = np.zeros((NUM_LEARNING_RATES, NUM_PARTITIONS, NUM_CLASSES), dtype = myIntType)
    KNN_Winners_ClassWise_Count_buf_1 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf = KNN_Winners_ClassWise_Count)#read buffer

    
    
    #Confidence per each datapoint
    confidence_array_neuron_KNN = np.array(np.zeros((NUM_LEARNING_RATES * NUM_PARTITIONS * NUM_DATAPOINTS,)).T, dtype = myFloatType)
    confidence_array_neuron_KNN_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = confidence_array_neuron_KNN)#write buffer
    #Prediction per partition for each data point
    prediction_per_part_array_neuron_KNN = np.array(np.zeros((NUM_LEARNING_RATES * NUM_PARTITIONS * NUM_DATAPOINTS ,)).T, dtype = myIntType)
    prediction_per_part_array_neuron_KNN[prediction_per_part_array_neuron_KNN == 0] = '3' 
    prediction_per_part_array_neuron_KNN_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = prediction_per_part_array_neuron_KNN)#write buffer
#---------------------------------------------------------------------------------------------------------------------------------------------------------------
    summ= np.array(np.zeros(NUM_DATAPOINTS), dtype = myFloatType )
    summ_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = summ)#write buffer

    rad_map = np.array(NUM_DATAPOINTS, dtype = myFloatType)
    rad_map_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = rad_map)#write buffer

    dominant_count_first = np.array(np.zeros(NUM_DATAPOINTS), dtype = myIntType)
    dominant_count_first_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = dominant_count_first)#write buffer

    dominant_count_second = np.array(np.zeros(NUM_DATAPOINTS), dtype = myIntType)
    dominant_count_second_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = dominant_count_second)#write buffer

    dominant_count_third = np.array(np.zeros(NUM_DATAPOINTS), dtype = myIntType)
    dominant_count_third_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = dominant_count_third)#write buffer

    confidence_phase_3= np.array(np.zeros((NUM_LEARNING_RATES * NUM_PARTITIONS * TOTAL_NUM_NEURONS * NUM_CLASSES,)).T, dtype = myFloatType) 
    confidence_phase_3_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = confidence_phase_3)#write buffer

    confidence_phase_3= np.true_divide(active_dominant_count,active_centers_total_count+1)
    confidence_phase_3_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = confidence_phase_3)#write buffer

    #print("confidence",confidence)


    #print("shape of winner array",np.shape(dominant_count_first))
    dist_winner =  np.array(np.zeros(NUM_DATAPOINTS), dtype = myFloatType)
    dist_winner_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = dist_winner)#write buffer

    dist_second =  np.array(np.zeros(NUM_DATAPOINTS), dtype = myFloatType)
    dist_second_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = dist_second )#write buffer

    dist_third =  np.array(np.zeros(NUM_DATAPOINTS), dtype = myFloatType)
    dist_third_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = dist_third)#write buffer
    
    maxconfidenceclass =  np.array(np.zeros(NUM_DATAPOINTS), dtype = myIntType)
    maxconfidenceclass_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = maxconfidenceclass)#write buffer

    total_count_first =  np.array(np.zeros(NUM_DATAPOINTS), dtype = myIntType)
    total_count_first_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = total_count_first)#write buffer

    total_count_second =  np.array(np.zeros(NUM_DATAPOINTS), dtype = myIntType)
    total_count_second_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = total_count_second)#write buffer

    total_count_third =  np.array(np.zeros(NUM_DATAPOINTS), dtype = myIntType)
    total_count_third_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = total_count_third)#write buffer
    
    confidence_winner =  np.array(np.zeros(NUM_DATAPOINTS), dtype = myFloatType)
    confidence_winner_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = confidence_winner)#write buffer

    confidence_second =  np.array(np.zeros(NUM_DATAPOINTS), dtype = myFloatType)
    confidence_second_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = confidence_second)#write buffer

    confidence_third =  np.array(np.zeros(NUM_DATAPOINTS), dtype = myFloatType)
    confidence_third_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = confidence_third )#write buffer

    dominant_class_winner =np.array(np.zeros(NUM_DATAPOINTS), dtype = myIntType)
    dominant_class_winner_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = dominant_class_winner)#write buffer

    dominant_class_second =np.array(np.zeros(NUM_DATAPOINTS), dtype = myIntType)
    dominant_class_second_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = dominant_class_second)#write buffer

    dominant_class_third =np.array(np.zeros(NUM_DATAPOINTS), dtype = myIntType)
    dominant_class_third_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = dominant_class_third)#write buffer
    
    winner_neuron =np.array(np.zeros(NUM_DATAPOINTS), dtype = myIntType)
    winner_neuron_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = winner_neuron)#write buffer

    second_neuron =np.array(np.zeros(NUM_DATAPOINTS), dtype = myIntType)
    second_neuron_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = second_neuron)#write buffer

    third_neuron =np.array(np.zeros(NUM_DATAPOINTS), dtype = myIntType)
    third_neuron_buf_1 = cl.Buffer(context, mem_flags.READ_WRITE | mem_flags.COPY_HOST_PTR, hostbuf = third_neuron)#write buffer

    grid_X, grid_Y = NUM_PARTITIONS *TOTAL_NUM_NEURONS , NUM_LEARNING_RATES
    #noofThreads = int(TOTAL_NUM_NEURONS) #CUDA Doesn't allow np.int64 variables so doing a typecase to int
    #workUnits_X, workUnits_Y, workUnits_Z = noofThreads, 1, 1
    print("grid_x",grid_X)
    print("type grid",type(grid_X))
    confidenceScoreTime = time.time()


    program.Confidence_Score(queue, (grid_X, grid_Y), None,
        NUM_DATAPOINTS,
        TOTAL_NUM_FEATURES,
        TOTAL_NUM_NEURONS,
        total_num_features_all_featurespaces,
        NUM_NETS,
        NUM_PARTITIONS,
        NUM_CLASSES,
        NUM_FEATURES_PER_PARTITION_buf_1,
        active_centers_orig_buf_1,
        active_centers_total_count_buf_1,
        active_dominant_count_buf_1,
        active_centers_per_class_buf_1,
        min_pos_buf_1,
        prediction_per_part_array_neuron_KNN_buf_1,
        cumulative_no_features_featurespace_buf_1,
        Input_Data_buf_1,
        map_data_buf_1,
        distance_map_buf_1,
        min_array_buf_1,
        radius_map_buf_1,
        confidence_array_neuron_KNN_buf_1,
        KNN_Winners_ClassWise_Count_buf_1,
        summ_buf_1,
        rad_map_buf_1,
        dominant_count_first_buf_1,
        dominant_count_second_buf_1,
        dominant_count_third_buf_1,
        confidence_phase_3_buf_1,
        dist_winner_buf_1,
        dist_second_buf_1,
        dist_third_buf_1,
        total_count_first_buf_1,
        total_count_second_buf_1,
        total_count_third_buf_1,
        maxconfidenceclass_buf_1,
        confidence_winner_buf_1,
        confidence_second_buf_1,
        confidence_third_buf_1,
        dominant_class_winner_buf_1,
        dominant_class_second_buf_1,
        dominant_class_third_buf_1,
        winner_neuron_buf_1,
        second_neuron_buf_1,
        third_neuron_buf_1)
        #block=(workUnits_X, workUnits_Y, workUnits_Z), grid=(grid_X,grid_Y,grid_Z))
    cl.enqueue_copy(queue, prediction_per_part_array_neuron_KNN, prediction_per_part_array_neuron_KNN_buf_1).wait()
    cl.enqueue_copy(queue, confidence_array_neuron_KNN, confidence_array_neuron_KNN_buf_1).wait()
    cl.enqueue_copy(queue, distance_map, distance_map_buf_1).wait()
    cl.enqueue_copy(queue, radius_map, radius_map_buf_1).wait()
    print("\nTime for Confidence and Prediction Calculation : {0:.5f} secs".format(time.time()-confidenceScoreTime))
    #cl.enqueue_copy(queue, prediction_per_part_array_neuron_KNN, prediction_per_part_array_neuron_KNN_buf_1).wait()
    #cl.enqueue_copy(queue, confidence_array_neuron_KNN, confidence_array_neuron_KNN_buf_1).wait()
    #cl.enqueue_copy(queue, distance_map, distance_map_buf_1).wait()
    #cl.enqueue_copy(queue, radius_map, radius_map_buf_1).wait()
    #cl.enqueue_copy(queue, dominant_count_first, dominant_count_first_buf_1).wait()
    #cl.enqueue_copy(queue, dominant_count_second, dominant_count_second_buf_1).wait()
    #cl.enqueue_copy(queue, dominant_count_third, dominant_count_third_buf_1).wait()
    #cl.enqueue_copy(queue, confidence_phase_3, confidence_phase_3_buf_1).wait()
    #cl.enqueue_copy(queue, dist_winner, dist_winner_buf_1).wait()
    #cl.enqueue_copy(queue, dist_second, dist_second_buf_1).wait()
    #cl.enqueue_copy(queue, dist_winner, dist_winner_buf_1).wait()
    #cl.enqueue_copy(queue, dist_third,  dist_third_buf_1).wait()
    #cl.enqueue_copy(queue, total_count_first, total_count_first_buf_1).wait()
    #cl.enqueue_copy(queue, total_count_second, total_count_second_buf_1).wait()
    #cl.enqueue_copy(queue, total_count_third, total_count_third_buf_1).wait()
    #cl.enqueue_copy(queue, confidence_phase_3, confidence_phase_3_buf_1).wait()
    #cl.enqueue_copy(queue, maxconfidenceclass, maxconfidenceclass_buf_1).wait()
    #cl.enqueue_copy(queue, confidence_winner, confidence_winner_buf_1).wait()
    #cl.enqueue_copy(queue, confidence_second, confidence_second_buf_1).wait()
    #cl.enqueue_copy(queue,  confidence_third,  confidence_third_buf_1).wait()
    #cl.enqueue_copy(queue, dominant_class_winner, dominant_class_winner_buf_1).wait()
    #cl.enqueue_copy(queue, dominant_class_second, dominant_class_second_buf_1).wait()
    #cl.enqueue_copy(queue, dominant_class_third, dominant_class_third_buf_1).wait()
    #cl.enqueue_copy(queue, winner_neuron, winner_neuron_buf_1).wait()
    #cl.enqueue_copy(queue, second_neuron, second_neuron_buf_1).wait()
    #cl.enqueue_copy(queue, third_neuron, third_neuron_buf_1).wait()
 

    #print("\nTime for Confidence and Prediction Calculation : {0:.5f} secs".format(time.time()-confidenceScoreTime))
    distance_map=distance_map.ravel()
    #print("total_count",total_count)
    #np.savetxt("active_total.csv", zip(total_count_first,total_count_second,total_count_third),header="fdist,sdist,tdist", delimiter=",",fmt="%s", comments='' )
    #np.savetxt("active_dominant.csv",zip( dominant_count_first,dominant_count_second,dominant_count_third),header="first,second,third", delimiter=",",fmt="%s", comments='')
    #np.savetxt("confidence.csv",zip( confidence_winner,confidence_second,confidence_third),header="first,second,third", delimiter=",",fmt="%s", comments='')
    #np.savetxt("distance.csv",zip( dist_winner,dist_second,dist_third),header="first,second,third", delimiter=",",fmt="%s", comments='')
    #np.savetxt("dominant_class.csv",zip( dominant_class_winner,dominant_class_second,dominant_class_third),header="first,second,third", delimiter=",",fmt="%s", comments='')
    #np.savetxt("neurons.csv",zip( winner_neuron,second_neuron,third_neuron),header="first,second,third", delimiter=",",fmt="%s", comments='')
    #np.savetxt("prediction.csv",( prediction_per_part_array_neuron_KNN),header="confidence_array", delimiter=",",fmt="%s", comments='')
    #np.savetxt("final.csv",zip( dist_winner,winner_neuron,dominant_class_winner,dominant_count_first,total_count_first,confidence_winner,
    #dist_second,second_neuron,dominant_class_second,dominant_count_second,total_count_second,confidence_second,
    #dist_third,third_neuron,dominant_class_third,dominant_count_third,total_count_third,confidence_third),header="distance_winner,winner_neuron,dominant_class_winner,             dominant_count_first,dominant_count_total,confidence_winner,distance_second,second_neuron,dominant_class_second, dominant_count_second,total_count_second,confidence_second,      distance_third,third_neuron,dominant_class_third, dominant_count_third,total_count_third,confidence_third", delimiter=",",fmt="%s", comments='')
    length_pred = len(target_arr_int) * NUM_PARTITIONS * NUM_LEARNING_RATES
    no_attributes_per_part_original = NUM_FEATURES_PER_PARTITION
    for i in range(0,NUM_LEARNING_RATES):
        NUM_FEATURES_PER_PARTITION = NUM_FEATURES_PER_PARTITION + no_attributes_per_part_original	


    New_Predictions = np.reshape(prediction_per_part_array_neuron_KNN, (NUM_LEARNING_RATES, NUM_PARTITIONS, NUM_DATAPOINTS))
   
    Accuracies = []
    precsion = []
    re_call = []
    f_score= []
    temp = 0
    for learnRatePredictions in New_Predictions:
        for partitionPredictions in learnRatePredictions:
            # if temp < 5:
            #     for i in zip(target_arr_int, partitionPredictions):
            #         print(i)
            #     temp += 1
            #     print("\n")
            curr_accuracy = accuracy_score(target_arr_int, partitionPredictions)
            prec, recall, fscore, support=met.precision_recall_fscore_support(target_arr_int, partitionPredictions, average="macro")
            Accuracies.append(curr_accuracy)
            precsion.append( prec)
            re_call.append(recall)
            f_score.append(fscore)

    return Accuracies, New_Predictions,precsion,re_call,f_score
