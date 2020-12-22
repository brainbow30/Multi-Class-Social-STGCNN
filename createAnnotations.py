import math


class annotations(object):

    def __init__(self, path):
        self.path = path
        self.annotations = self.loadAnnotations()

    def loadAnnotations(self):
        annotations = open(
            'annotations/' + self.path + '/annotations.txt', 'r')
        Lines = annotations.readlines()
        annotationsByFrame = {}
        for line in Lines:
            ped_id, x_min, y_min, x_max, y_max, frame, _, _, _, label = line.strip().split(" ")
            frame = math.floor(float(frame))
            frame = str(frame)
            if (frame in annotationsByFrame):
                annotationsByFrame[frame].append([ped_id, x_min, y_min, x_max, y_max, label])
            else:
                annotationsByFrame[frame] = [[ped_id, x_min, y_min, x_max, y_max, label]]
        return annotationsByFrame

    def getFrameAnnotations(self, frameNum):
        return self.annotations[str(int(frameNum))]
