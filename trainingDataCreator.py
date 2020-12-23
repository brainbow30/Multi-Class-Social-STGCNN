import os
import shutil

import numpy as np


def rowConversion(row):
    ped_id, x_min, y_min, x_max, y_max, frame, _, _, _, label = row
    return float(frame), float(ped_id), (float(x_min) + float(x_max)) / 2.0, (float(y_min) + float(y_max)) / 2.0


def convertData(data):
    trainingData = []
    for row in data:
        trainingData.append(rowConversion(row))
    return np.asarray(trainingData)


def read_file(_path, delim='space'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    print(_path)
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            data.append(line)
    return np.asarray(data)


def createTrainingData(inputFolder, outputFolder):
    locations = os.listdir(inputFolder)
    for location in locations:
        videos = os.listdir(os.path.join(inputFolder, location))
        for video in videos:
            path = os.path.join(inputFolder, location, video, "annotations.txt")
            data = read_file(path, 'space')

            trainingData = convertData(data)
            if (not (os.path.isdir(os.path.join(inputFolder + "Processed", location, video, "test")))):
                os.makedirs(os.path.join(inputFolder + "Processed", location, video, "test"))
            shutil.rmtree(os.path.join(outputFolder, location, video, "val"), ignore_errors=True)
            shutil.rmtree(os.path.join(outputFolder, location, video, "train"), ignore_errors=True)
            shutil.copytree("testValFolders\\val", os.path.join(outputFolder, location, video, "val"))
            shutil.copytree("testValFolders\\train", os.path.join(outputFolder, location, video, "train"))
            np.savetxt(
                os.path.join(outputFolder, location, video, "test", "stan" + "_" + location + "_" + video + ".txt"),
                trainingData, fmt='%.5e', delimiter='\t', newline='\n', header='', footer='', comments='# ',
                encoding=None)
