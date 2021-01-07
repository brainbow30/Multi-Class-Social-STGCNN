import math
import os

import numpy as np


def rowConversion(row):
    ped_id, x_min, y_min, x_max, y_max, frame, _, _, _, label = row
    return float(frame), float(ped_id), (float(x_min) + float(x_max)) / 2.0, (float(y_min) + float(y_max)) / 2.0, eval(
        label)


def convertData(data, trainingTestSplit=0.7, testValidSplit=0.5, samplingRate=15, labels=None):
    trainingData = []
    testData = []
    validationData = []
    maxTrainingFrame = int(math.floor(len(data) * trainingTestSplit))
    maxTestFrame = maxTrainingFrame + int(math.floor(len(data) * (1 - trainingTestSplit) * testValidSplit))
    frame = 0
    for row in data:
        row = rowConversion(row)
        if (labels is None or row[4] in labels):
            if (row[0] % samplingRate == 0.0):
                row = (row[0] / samplingRate,) + row[1:-1]
                if (frame <= maxTrainingFrame):
                    trainingData.append(row)
                elif (frame <= maxTestFrame):
                    testData.append(row)
                else:
                    validationData.append(row)
        frame += 1
    return np.asarray(trainingData), np.asarray(testData), np.asarray(validationData)


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
    for location in locations:
        videos = os.listdir(os.path.join(inputFolder, location))
        for video in videos:
            path = os.path.join(inputFolder, location, video, "annotations.txt")
            data = read_file(path, 'space')

            trainingData, testData, validationData = convertData(data, samplingRate=samplingRate, labels=None)

            if (not (os.path.isdir(os.path.join(inputFolder + "Processed", location, video, "train")))):
                os.makedirs(os.path.join(inputFolder + "Processed", location, video, "train"))
            if (not (os.path.isdir(os.path.join(inputFolder + "Processed", location, video, "test")))):
                os.makedirs(os.path.join(inputFolder + "Processed", location, video, "test"))
            if (not (os.path.isdir(os.path.join(inputFolder + "Processed", location, video, "val")))):
                os.makedirs(os.path.join(inputFolder + "Processed", location, video, "val"))
            if (not np.any(np.isnan(trainingData))):
                np.savetxt(
                    os.path.join(outputFolder, location, video, "train",
                                 "stan" + "_" + location + "_" + video + ".txt"),
                    trainingData, fmt='%.5e', delimiter='\t', newline='\n', header='', footer='', comments='# ',
                    encoding=None)
            else:
                print("Invalid Training Data")
            if (not np.any(np.isnan(testData))):
                np.savetxt(
                    os.path.join(outputFolder, location, video, "test", "stan" + "_" + location + "_" + video + ".txt"),
                    testData, fmt='%.5e', delimiter='\t', newline='\n', header='', footer='', comments='# ',
                    encoding=None)
            else:
                print("Invalid Test Data")
            if (not np.any(np.isnan(validationData))):
                np.savetxt(
                    os.path.join(outputFolder, location, video, "val", "stan" + "_" + location + "_" + video + ".txt"),
                    validationData, fmt='%.5e', delimiter='\t', newline='\n', header='', footer='', comments='# ',
                    encoding=None)
            else:
                print("Invalid Validation Data")


print("Converting Stanford Dataset...")
createTrainingData("trainingData\\stanford", "trainingData\\stanfordProcessed", samplingRate=5, labels=["Biker"])
print("Done")
