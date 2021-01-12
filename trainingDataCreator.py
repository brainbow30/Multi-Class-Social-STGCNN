import math
import os

import numpy as np
from tqdm import tqdm

import config


def rowConversion(row):
    ped_id, x_min, y_min, x_max, y_max, frame, _, _, _, label = row
    return float(frame), float(ped_id), (float(x_min) + float(x_max)) / 2.0, (
                float(y_min) + float(y_max)) / 2.0, label.strip("\"")


def convertData(data, trainingTestSplit=0.7, testValidSplit=0.5, samplingRate=5, labels=None):
    trainingData = {}
    testData = {}
    validationData = {}
    maxTrainingFrame = int(math.floor(len(data) * trainingTestSplit))
    maxTestFrame = maxTrainingFrame + int(math.floor(len(data) * (1 - trainingTestSplit) * testValidSplit))
    frame = 0
    for i in range(samplingRate):
        trainingData[i] = []
        testData[i] = []
        validationData[i] = []
    for row in data:
        row = rowConversion(row)
        if (labels is None or row[4] in labels):
            row = (math.floor(row[0] / samplingRate),) + row[1:-1]
            if (frame <= maxTrainingFrame):
                trainingData[frame % samplingRate].append(row)
            elif (frame <= maxTestFrame):
                testData[frame % samplingRate].append(row)
            else:
                validationData[frame % samplingRate].append(row)
        frame += 1
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
                testData = np.asarray(testDataDict[i])
                validationData = np.asarray(validationDataDict[i])
                if (not np.any(np.isnan(trainingData))):
                    np.savetxt(
                        os.path.join(outputFolder, location, video, "train",
                                     "stan" + "_" + location + "_" + video + "_" + str(i) + ".txt"),
                        trainingData, fmt='%.5e', delimiter='\t', newline='\n', header='', footer='', comments='# ',
                        encoding=None)
                else:
                    print("Invalid Training Data")
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


print("Converting Stanford Dataset...")
createTrainingData("trainingData\\stanford", "trainingData\\stanfordProcessed", samplingRate=config.samplingRate,
                   labels=["Pedestrian"])
print("Done")
