import math
import os

import numpy as np
from tqdm import tqdm

import config


def rowConversion(row):
    ped_id, x_min, y_min, x_max, y_max, frame, _, _, _, label = row
    return float(frame), float(ped_id), (float(x_min) + float(x_max)) / (2.0), (
            float(y_min) + float(y_max)) / (2.0), label.strip("\"")


def scaleCoordinates(row):
    frame, ped_id, x, y = row
    return float(frame), float(ped_id), float(x / config.annotationScale), float(y / config.annotationScale)


def convertData(data, trainingTestSplit=0.7, testValidSplit=0.5, samplingRate=5, labels=None):
    trainingData = {}
    testData = {}
    validationData = {}
    maxTrainingFrame = int(math.floor(len(data) * trainingTestSplit))
    maxTestFrame = maxTrainingFrame + int(math.floor(len(data) * (1 - trainingTestSplit) * testValidSplit))
    frame = 0
    maxX = 0
    maxY = 0
    for label in labels:
        testData[label] = {}
    for i in range(samplingRate):
        trainingData[i] = []
        if not (labels is None):
            for label in labels:
                testData[label][i] = []
        else:
            testData[i] = []
        validationData[i] = []
    for row in data:
        row = rowConversion(row)
        if (row[2] > maxX):
            maxX = row[2]
        if (row[3] > maxY):
            maxY = row[3]

        if (labels is None or row[4] in labels):
            label = row[-1]
            row = (math.floor(row[0] / samplingRate),) + row[1:-1]
            if (frame <= maxTrainingFrame):
                trainingData[frame % samplingRate].append(scaleCoordinates(row))
            elif (frame <= maxTestFrame):
                if not (labels is None):
                    testData[label][frame % samplingRate].append(row)
                else:
                    testData[frame % samplingRate].append(row)
            else:
                validationData[frame % samplingRate].append(scaleCoordinates(row))
        frame += 1
    # take middle 90% of image to train, test and validate on it
    for i in range(samplingRate):
        trainingData[i] = list(filter(lambda row: ((row[2] >= (
                maxX / (config.annotationScale * config.fractionToRemove)) and row[2] <= (maxX - maxX / (
                config.annotationScale * config.fractionToRemove))) and (row[3] >= (
                maxY / (config.annotationScale * config.fractionToRemove)) and row[3] <= (maxY - (
                maxY / (config.annotationScale * config.fractionToRemove))))), trainingData[i]))
        if (labels is None):
            testData[i] = list(filter(lambda row: ((row[2] >= (maxX / config.fractionToRemove) and row[2] <= (
                    maxX - (maxX / config.fractionToRemove))) and (
                                                           row[3] >= (maxY / config.fractionToRemove) and row[3] <= (
                                                           maxY - (maxY / config.fractionToRemove)))), testData[i]))
        else:
            for label in labels:
                testData[label][i] = list(
                    filter(lambda row: ((row[2] >= (maxX / config.fractionToRemove) and row[2] <= (
                            maxX - (maxX / config.fractionToRemove))) and (
                                                row[3] >= (maxY / config.fractionToRemove) and row[
                                            3] <= (
                                                        maxY - (maxY / config.fractionToRemove)))),
                           testData[label][i]))
        validationData[i] = list(filter(lambda row: ((row[2] >= (
                maxX / (config.annotationScale * config.fractionToRemove)) and row[2] <= (maxX - maxX / (
                config.annotationScale * config.fractionToRemove))) and (row[3] >= (
                maxY / (config.annotationScale * config.fractionToRemove)) and row[3] <= (maxY - (
                maxY / (config.annotationScale * config.fractionToRemove))))), validationData[i]))
    return trainingData, testData, validationData


def read_file(_path, delim='space'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            data.append(line)
    return np.asarray(data)


# todo create file showing current training data settings
def createTrainingData(inputFolder, outputFolder, samplingRate=15, labels=None):
    locations = os.listdir(inputFolder)
    pbar = tqdm(total=len(locations))
    for location in locations:
        pbar.update(1)
        videos = os.listdir(os.path.join(inputFolder, location))
        for video in videos:
            path = os.path.join(inputFolder, location, video, "annotations.txt")
            data = read_file(path, 'space')

            trainingDataDict, testDataDict, validationDataDict = convertData(data, samplingRate=samplingRate,
                                                                             labels=labels)
            if (not (os.path.isdir(os.path.join(outputFolder, location, video, "train")))):
                os.makedirs(os.path.join(outputFolder, location, video, "train"))
            if (not (os.path.isdir(os.path.join(outputFolder, location, video, "test")))):
                os.makedirs(os.path.join(outputFolder, location, video, "test"))
            if (not (os.path.isdir(os.path.join(outputFolder, location, video, "val")))):
                os.makedirs(os.path.join(outputFolder, location, video, "val"))
            for i in range(samplingRate):
                trainingData = np.asarray(trainingDataDict[i])
                validationData = np.asarray(validationDataDict[i])
                if (not np.any(np.isnan(trainingData))):
                    np.savetxt(
                        os.path.join(outputFolder, location, video, "train",
                                     "stan" + "_" + location + "_" + video + "_" + str(i) + ".txt"),
                        trainingData, fmt='%.5e', delimiter='\t', newline='\n', header='', footer='', comments='# ',
                        encoding=None)
                else:
                    print("Invalid Training Data")
                if not (labels is None):
                    for label in labels:
                        if (not (os.path.isdir(os.path.join(outputFolder, location, video, "test", label)))):
                            os.makedirs(os.path.join(outputFolder, location, video, "test", label))
                        testData = np.asarray(testDataDict[label][i])
                        if (not np.any(np.isnan(testData))):
                            np.savetxt(
                                os.path.join(outputFolder, location, video, "test", label,
                                             "stan" + "_" + location + "_" + video + "_" + str(i) + ".txt"),
                                testData, fmt='%.5e', delimiter='\t', newline='\n', header='', footer='', comments='# ',
                                encoding=None)
                        else:
                            print("Invalid Test Data")
                else:
                    testData = np.asarray(testDataDict[i])

                    if (not np.any(np.isnan(testData))):
                        np.savetxt(
                            os.path.join(outputFolder, location, video, "test",
                                         "stan" + "_" + location + "_" + video + "_" + str(i) + ".txt"),
                            testData, fmt='%.5e', delimiter='\t', newline='\n', header='', footer='', comments='# ',
                            encoding=None)
                    else:
                        print("Invalid Test Data")
                if (not np.any(np.isnan(validationData))):
                    np.savetxt(
                        os.path.join(outputFolder, location, video, "val",
                                     "stan" + "_" + location + "_" + video + "_" + str(i) + ".txt"),
                        validationData, fmt='%.5e', delimiter='\t', newline='\n', header='', footer='', comments='# ',
                        encoding=None)
                else:
                    print("Invalid Validation Data")
    pbar.close()
