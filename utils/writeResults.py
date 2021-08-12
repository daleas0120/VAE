#!/usr/bin/env python

import os
from tabulate import tabulate
from array import *

def dict2str(dict_obj, key, start_pt = 0):
    # when this function is called for the class list, we want to omit the background class at index 0
    dict_end = len(dict_obj) - 1 - start_pt
    new_str = ''

    for idx in range(dict_end):
        i = idx + start_pt
        new_str = new_str + dict_obj[i][key] + ','
    new_str = new_str + dict_obj[i + 1][key] + '\n'

    return new_str

def dict2arr(dict_obj, key, start_pt = 0):

    #dict_end = len(dict_obj) - start_pt - 1

    new_array = []

    for idx in range(len(dict_obj) - start_pt):
        i = idx + start_pt
        new_array.append(dict_obj[i][key])

    return new_array

def getStringVersion(num_array):
    next_line = ''
    for idx in range(len(num_array) - 1):
        next_line = next_line + str(num_array[idx]) + ','
    next_line = next_line + str(num_array[idx + 1]) + '\n'
    return next_line

def getStringArr(num_array):
    next_line = []
    for jdx in range((num_array.size)):
        next_line.append(str(num_array[jdx]))

    return next_line

def simpleResults(PATH_TO_RESULTS, results_file_name, orig_list, loss_per_img):
    """

    """
    results_file_name = os.path.join(PATH_TO_RESULTS, results_file_name)
    results_file = open(results_file_name, 'a')

    for idx in range(len(orig_list)):

        itm = orig_list[idx]['img_path'] + "," + str(loss_per_img[idx]) + "\n"

        results_file.write(itm)

    data = 0


def ResultsCSV(PATH_TO_RESULTS, log_dir, epoch, groundTruthFile, results_file_name, numbers, num_input_gt,\
                           num_found_dets, num_missing_detects, num_incorrect_detects, num_correct_detects, list_of_classes,\
                           percent_correct, avg_detection_accuracy, precision, recall, F, stats_by_class):
    """
    PATH_TO_RESULTS: string - file path to network results folder
    log_dir:
    epoch:
    groundTruthFile:
    results_file_name:
    numbers:
    num_input_gt:
    num_found_dets:
    num_missing_detects:
    num_incorrect_detects:
    num_correct_detects:
    list_of_classes:
    percent_correct:
    avg_detection_accuracy:
    precision:
    recall:
    F:
    stats_by_class:
    """
    class_names = dict2str(list_of_classes, 'name', 1)

    class_list = dict2arr(list_of_classes, 'name', 1)

    results_file_name = os.path.join(PATH_TO_RESULTS, results_file_name)
    results_file = open(results_file_name, 'a')

    log_details = 'log_dir: ' + log_dir + '\n'
    results_file.write(log_details)

    input_details = 'Input File: ' + groundTruthFile + '\n'
    results_file.write(input_details)

    weight_details = 'Weights: ' + str(epoch) + '\n\n'
    results_file.write(weight_details)

    header = 'INPUT_IMAGE,OUTPUT_IMAGE,DETECT,GROUND TRUTH,CONFIDENCE SCORE,' + class_names + ',' + '\n'
    results_file.write(header)

    for idx in range(len(numbers)):
        results_file.write(numbers[idx])

    results_file.write('===============================STATS FOR WHOLE SET==================================\n')

    num_in = 'NUMBER INPUT,' + str(num_input_gt) + '\n'
    results_file.write(num_in)

    num_found = 'NUMBER FOUND,' + str(num_found_dets) + '\n'
    results_file.write(num_found)

    num_miss = 'NUMBER MISSING,' + str(num_missing_detects) + '\n'
    results_file.write(num_miss)

    num_wrong = 'NUMBER INCORRECT,' + str(num_incorrect_detects) + '\n'
    results_file.write(num_wrong)

    num_right = 'NUMBER CORRECT,' + str(num_correct_detects) + '\n'
    results_file.write(num_right)

    per_right = 'PERCENT CORRECT,' + str(percent_correct) + '\n'
    results_file.write(per_right)

    avg_acc = 'AVG DETECTION ACCURACY,' + str(avg_detection_accuracy) + '\n'
    results_file.write(avg_acc)

    prec = 'PRECISION,' + str(precision) + '\n'
    results_file.write(prec)

    rec = 'RECALL,' + str(recall) + '\n'
    results_file.write(rec)

    f_num = 'F, ' + str(F) + '\n'
    results_file.write(f_num)

    results_file.write('================================STATS BY CLASS=================================\n')
    results_file.write('CLASS,INSTANCES INPUT,INSTANCES FOUND,INSTANCES CORRECT,PRECISION,RECALL,F\n')

    if stats_by_class.shape[0] == (len(list_of_classes) - 1):

        for idx in range(stats_by_class.shape[0]):

            next_line = getStringVersion(stats_by_class[idx])
            results_file.write(class_list[idx])
            results_file.write(',')
            results_file.write(next_line)


def ResultToCommandLine(source_id, conf_by_class, r, detection, detect_score, detect_label, list_of_classes):

    header = ['DETECT', 'GROUND TRUTH', 'CONFIDENCE SCORE']
    info = [detect_label, str(r['detect'][detection]), str(detect_score)]

    for idx in range(len(list_of_classes) - 1):
        header.append(list_of_classes[idx+1]['name'])
        info.append(conf_by_class[r['class_ids'][detection]][idx+1])

    print(tabulate([header, info], headers="firstrow"))


def statsByClassToCommandLine(stats_by_class, list_of_classes):

    class_list = dict2arr(list_of_classes, 'name', 1)
    header = ['CLASS','INSTANCES INPUT','INSTANCES FOUND','INSTANCES CORRECT','PRECISION','RECALL','F']
    info = []
    print('\n==========================================STATS BY CLASS===========================================\n')
    if stats_by_class.shape[0] == (len(list_of_classes) - 1):

        for idx in range(stats_by_class.shape[0]):
            class_name = class_list[idx]
            #next_line = stats_by_class[idx].tolist()
            next_line = getStringArr(stats_by_class[idx])
            next_line.insert(0, class_name)
            #print(tabulate([header, next_line], headers = "firstrow"))
            info.append(next_line)

    info.insert(0,header)
    print(tabulate(info, headers="firstrow"))
    print("\n")


def StatsToCommandLine(avg_detection_accuracy, num_correct_detects, num_incorrect_detects, num_missing_detects,\
                       results_by_class, num_input_gt, num_found_dets, percent_correct, precision, recall, F):
    header = ['________']
    numCorrectList = ['Num_Correct']
    numIncorrectList = ['Num_Incorrect']
    totalList = ['Total']

    for key in results_by_class:
        header.append(key)
        numCorrectList.append(results_by_class[key]['correct'])
        numIncorrectList.append(results_by_class[key]['incorrect'])
        totalList.append(results_by_class[key]['correct'] + results_by_class[key]['incorrect'])

    print('===================================================================================================')
    print(tabulate([['DETECTION ACCURACY', 'CORRECT DETECTS', 'INCORRECT DETECTS', 'MISSING DETECTS'],
                    [avg_detection_accuracy, num_correct_detects, num_incorrect_detects, num_missing_detects]]))
    print('===================================================================================================')
    print(tabulate([header, numCorrectList, numIncorrectList, totalList], headers="firstrow"))
    print('===================================================================================================')
    print('NUMBER INPUT : ', num_input_gt)
    print('NUMBER FOUND: ', num_found_dets)
    print('NUMBER MISSING: ', num_missing_detects)
    print('NUMBER INCORRECT: ', num_incorrect_detects)
    print('NUMBER CORRECT: ', num_correct_detects)
    print('PERCENT CORRECT: ', percent_correct)
    print('AVG DETECTION ACCURACY: ', avg_detection_accuracy)
    print('PRECISION: ', precision)
    print('RECALL: ', recall)
    print('F: ', F)
    print('===================================================================================================')


def ConfusionMatrixToCommandLine(var = False):
    bp = 0
# print(tabulate[['-',dataset.class_info[1]['name'], dataset.class_info[2]['name'], dataset.class_info[3]['name'], dataset.class_info[4]['name'], dataset.class_info[5]['name']],
#                [dataset.class_info[1]['name'], str(confusion_matrix[0,0]), str(confusion_matrix[0,1]), str(confusion_matrix[0,2]), str(confusion_matrix[0,3]), str(confusion_matrix[0,4])],
#                [dataset.class_info[2]['name'], str(confusion_matrix[1,0]), str(confusion_matrix[1,1]), str(confusion_matrix[1,2]), str(confusion_matrix[1,3]), str(confusion_matrix[1,4])],
#                [dataset.class_info[3]['name'], str(confusion_matrix[2,0]), str(confusion_matrix[2,1]), str(confusion_matrix[2,2]), str(confusion_matrix[2,3]), str(confusion_matrix[2,4])],
#                [dataset.class_info[4]['name'], str(confusion_matrix[3,0]), str(confusion_matrix[3,1]), str(confusion_matrix[3,2]), str(confusion_matrix[3,3]), str(confusion_matrix[3,4])],
#                [dataset.class_info[5]['name'], str(confusion_matrix[4,0]), str(confusion_matrix[4,1]), str(confusion_matrix[4,2]), str(confusion_matrix[4,3]), str(confusion_matrix[4,4])]
#       ])