import json
import math
import os
import shutil

import numpy as np
from sklearn.utils import compute_class_weight

import config


def rowConversion(row):
    ped_id, x_min, y_min, x_max, y_max, frame, _, _, _, label = row
    return float(frame), float(ped_id), (float(x_min) + float(x_max)) / 2.0, (
            float(y_min) + float(y_max)) / 2.0, label.strip("\"")


def convertData(data):
    return list(map(lambda row: rowConversion(row), data))


def splitIntoLabels(data, labels):
    if not (labels is None):
        labelledData = {}
        for label in labels:
            labelledData[label] = []
        for row in data:
            if row[4] in labels:
                labelledData[row[4]].append(row)
        return labelledData
    else:
        return data


def splitDataIntoSeqs(data, samplingRate=5):
    dict = {}
    for i in range(samplingRate):
        dict[i] = []
    for row in data:
        dict[row[0] % samplingRate].append((math.floor(row[0] / samplingRate),) + row[1:])
    return dict


# todo check if improves
def removeEdgeData(data, maxX, maxY):
    # take middle 90% of image to train, test and validate on it
    data = list(filter(lambda row: ((row[2] >= (
            maxX / config.percentageToRemove) and row[2] <= (maxX - maxX / (
        config.percentageToRemove))) and (row[3] >= (
            maxY / config.percentageToRemove) and row[3] <= (maxY - (
            maxY / config.percentageToRemove)))), data))
    return data


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


def convertSplitTrainingData(inputFolder, outputFolder, samplingRate=15, labels=None):
    new_config = {"samplingRate": config.frameSkip,
                  "labels": labels,
                  "inputFolder": inputFolder,
                  "outputFolder": outputFolder,
                  "fractionToRemove": config.percentageToRemove,
                  "annotationType": config.annotationType,
                  "combineLocations": config.combineLocationVideos,
                  "complete": False
                  }
    if os.path.exists(os.path.join(outputFolder, 'trainingDataConfig.json')):
        with open(os.path.join(outputFolder, 'trainingDataConfig.json')) as f:
            old_config = json.load(f)
        # check if config is the same and creation completed
        new_config["complete"] = True
        if new_config == old_config:
            print("No new config, skipping data creation")
            return
        # write json showing data is incomplete
        new_config["complete"] = False
        with open(os.path.join(outputFolder, 'trainingDataConfig.json'), 'w') as json_file:
            json.dump(new_config, json_file)
    # delete any current data in output folder
    if os.path.exists(outputFolder):
        shutil.rmtree(outputFolder)
    locations = os.listdir(inputFolder)
    trainingDataDict = {}
    testDataDict = {}
    validationDataDict = {}
    for location in locations:
        class_list = []
        for i in range(samplingRate):
            trainingDataDict[i] = []
            testDataDict[i] = []
            validationDataDict[i] = []
        print("Converting " + str(locations.index(location) + 1) + "/" + str(len(locations)) + " Locations...")
        videos = os.listdir(os.path.join(inputFolder, location))
        for video in videos:
            path = os.path.join(inputFolder, location, video, "annotations.txt")
            data = read_file(path, 'space')
            data = convertData(data)
            if (config.percentageToRemove > 0):
                maxX = max(data, key=lambda x: x[2])[2]
                maxY = max(data, key=lambda x: x[3])[3]
                data = removeEdgeData(data, maxX, maxY)
            testSplit = 0.15
            validSplit = 0.15
            maxTestFrame = int(math.floor(len(data) * testSplit))
            maxValidFrame = maxTestFrame + int(math.floor(len(data) * validSplit))

            testData = list(
                filter(lambda row: labels is None or row[4] in labels, data[:maxTestFrame]))
            videoTestDataDict = splitDataIntoSeqs(testData, samplingRate=samplingRate)

            validationData = list(
                filter(lambda row: labels is None or row[4] in labels, data[maxTestFrame:maxValidFrame]))
            videoValidationDataDict = splitDataIntoSeqs(validationData, samplingRate=samplingRate)

            trainingData = data[maxValidFrame:]
            trainingData = list(
                filter(lambda row: labels is None or row[4] in labels, trainingData))
            # undersampling
            # videoTrainingDataDict=splitIntoLabels(trainingData, labels)
            # desiredLength = sorted(list(map(lambda label: len(videoTrainingDataDict[label]), videoTrainingDataDict)))[-2]
            # trainingData=[]
            # for label in labels:
            #     #todo find better sampling method
            #     trainingData+=(videoTrainingDataDict[label][:desiredLength])
            videoTrainingDataDict = splitDataIntoSeqs(trainingData, samplingRate=samplingRate)

            for i in range(samplingRate):
                if videoTrainingDataDict[i]:
                    trainingDataDict[i] += videoTrainingDataDict[i]
                    class_list += (list(map(lambda row: row[4], videoTrainingDataDict[i])))
                if videoTestDataDict[i]:
                    testDataDict[i] += videoTestDataDict[i]
                if videoValidationDataDict[i]:
                    validationDataDict[i] += videoValidationDataDict[i]
            if not config.combineLocationVideos:
                saveClassInfo(class_list, labels, os.path.join(outputFolder, location, video))
                class_list = []
                saveData(trainingDataDict, testDataDict, validationDataDict, samplingRate,
                         os.path.join(outputFolder, location, video))
                for i in range(samplingRate):
                    trainingDataDict[i] = []
                    testDataDict[i] = []
                    validationDataDict[i] = []

        if config.combineLocationVideos:
            saveClassInfo(class_list, labels, os.path.join(outputFolder, location))
            saveData(trainingDataDict, testDataDict, validationDataDict, samplingRate,
                     os.path.join(outputFolder, location))


    new_config["complete"] = True
    with open(os.path.join(outputFolder, 'trainingDataConfig.json'), 'w') as json_file:
        json.dump(new_config, json_file)


def convertTrainingData(inputFolder, outputFolder, samplingRate=15, labels=None):
    new_config = {"samplingRate": config.frameSkip,
                  "labels": labels,
                  "inputFolder": inputFolder,
                  "outputFolder": outputFolder,
                  "fractionToRemove": config.percentageToRemove,
                  "annotationType": config.annotationType,
                  "complete": False
                  }
    if os.path.exists(os.path.join(outputFolder, 'trainingDataConfig.json')):
        with open(os.path.join(outputFolder, 'trainingDataConfig.json')) as f:
            old_config = json.load(f)
        # check if config is the same and creation completed
        new_config["complete"] = True
        if new_config == old_config:
            print("No new config, skipping data creation")
            return
        # write json showing data is incomplete
        new_config["complete"] = False
        with open(os.path.join(outputFolder, 'trainingDataConfig.json'), 'w') as json_file:
            json.dump(new_config, json_file)
    # delete any current data in output folder
    if os.path.exists(outputFolder):
        shutil.rmtree(outputFolder)
    trainLocations = os.listdir(os.path.join(inputFolder, "train"))
    testLocations = os.listdir(os.path.join(inputFolder, "test"))
    valLocations = os.listdir(os.path.join(inputFolder, "val"))
    locations = [trainLocations, valLocations, testLocations]
    class_list = []
    trainingDataDict = {}
    testDataDict = {}
    validationDataDict = {}

    for i in range(samplingRate):
        trainingDataDict[i] = []
        testDataDict[i] = []
        validationDataDict[i] = []
    dicts = [trainingDataDict, validationDataDict, testDataDict]
    folders = ["train", "val", "test"]
    for trainValTest in range(3):
        for video in locations[trainValTest]:
            path = os.path.join(inputFolder, folders[trainValTest], video)
            data = read_file(path, 'space')
            data = convertData(data)
            if (len(data) == 0):
                continue
            if (config.percentageToRemove > 0):
                maxX = max(data, key=lambda x: x[2])[2]
                maxY = max(data, key=lambda x: x[3])[3]
                data = removeEdgeData(data, maxX, maxY)
            data = list(
                filter(lambda row: labels is None or row[4] in labels, data))
            videoDataDict = splitDataIntoSeqs(data, samplingRate=samplingRate)
            for i in range(samplingRate):
                if videoDataDict[i]:
                    dicts[trainValTest][i] += videoDataDict[i]
                    class_list += (list(map(lambda row: row[4], videoDataDict[i])))

    saveClassInfo(class_list, labels, os.path.join(outputFolder))
    saveData(dicts[0], dicts[2], dicts[1], samplingRate,
             os.path.join(outputFolder))

    new_config["complete"] = True
    with open(os.path.join(outputFolder, 'trainingDataConfig.json'), 'w') as json_file:
        json.dump(new_config, json_file)


def saveClassInfo(class_list, labels, outputFolder):
    class_counts = []
    for label in labels:
        class_counts.append(class_list.count(label))
    try:
        class_weights = compute_class_weight("balanced", classes=labels, y=class_list)
    except ValueError:
        class_weights = compute_class_weight("balanced", classes=labels, y=class_list + labels)
    if not (os.path.isdir(outputFolder)):
        os.makedirs(outputFolder)
    with open(os.path.join(outputFolder, "classInfo.json"), 'w') as json_file:
        json.dump({"class_weights": class_weights.tolist(), "class_counts": class_counts}, json_file)


def saveData(trainingDataDict, testDataDict, validationDataDict, samplingRate, outputFolder):
    if not (os.path.isdir(os.path.join(outputFolder, "train"))):
        os.makedirs(os.path.join(outputFolder, "train"))
    if not (os.path.isdir(os.path.join(outputFolder, "test"))):
        os.makedirs(os.path.join(outputFolder, "test"))
    if not (os.path.isdir(os.path.join(outputFolder, "val"))):
        os.makedirs(os.path.join(outputFolder, "val"))

    for i in range(samplingRate):
        trainingData = np.asarray(trainingDataDict[i])
        testData = np.asarray(testDataDict[i])
        validationData = np.asarray(validationDataDict[i])
        np.savetxt(
            os.path.join(outputFolder, "train",
                         str(i) + ".txt"),
            trainingData, fmt="%s", delimiter=' ', newline='\n', header='', footer='', comments='# ',
            encoding=None)
        np.savetxt(
            os.path.join(outputFolder, "test",
                         str(i) + ".txt"),
            testData, fmt='%s', delimiter=' ', newline='\n', header='', footer='', comments='# ',
            encoding=None)

        np.savetxt(
            os.path.join(outputFolder, "val",
                         str(i) + ".txt"),
            validationData, fmt='%s', delimiter=' ', newline='\n', header='', footer='', comments='# ',
            encoding=None)


if config.trainingDataAction == "create":
    print("Creating Dataset...")
    convertSplitTrainingData(os.path.join("trainingData\\", config.path),
                             os.path.join("trainingData\\", config.path + "Processed"),
                             samplingRate=config.frameSkip,
                             labels=config.labels)

elif config.trainingDataAction == "convert":
    print("Converting Dataset...")
    convertTrainingData(os.path.join("trainingData\\", config.path),
                        os.path.join("trainingData\\", config.path + "Processed"),
                        samplingRate=config.frameSkip,
                        labels=config.labels)
