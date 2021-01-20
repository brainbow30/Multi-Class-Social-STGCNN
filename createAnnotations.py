import math

import config


class annotations(object):

    def __init__(self, path):
        self.path = path
        self.annotations = self.loadAnnotations()

    def loadAnnotations(self):
        annotations = open(
            './annotations/' + self.path + '/annotations.txt', 'r')
        Lines = annotations.readlines()
        annotationsByFrame = {}
        for line in Lines:
            if (config.annotationType == "stanford"):
                ped_id, x_min, y_min, x_max, y_max, frame, _, _, _, label = line.strip().split(" ")
                frame = math.floor(float(frame))
                frame = str(frame)
            elif (config.annotationType == "seq"):
                line = line.strip().split(" ")
                while "" in line:
                    line.remove("")
                frame, ped_id, x, _, y, _, _, _, = line
                frame = math.floor(float(frame))
                frame = str(frame)
                x_min = x
                x_max = x
                y_min = y
                y_max = y
                label = "Pedestrian"
            if (frame in annotationsByFrame):
                annotationsByFrame[frame].append([ped_id, x_min, y_min, x_max, y_max, label])
            else:
                annotationsByFrame[frame] = [[ped_id, x_min, y_min, x_max, y_max, label]]
        return annotationsByFrame

    def getFrameAnnotations(self, frameNum):
        return self.annotations[str(int(frameNum))]
