import math

import cv2


class VideoCamera(object):
    def __init__(self):
        # capturing video
        self.video = cv2.VideoCapture('videos/bookstore/video0/video.mov')
        self.annotations = self.getAnnotations()
        self.pedPastTraj = {}

    def __del__(self):
        # releasing camera
        self.video.release()

    def getAnnotation(self):
        annotations = open(
            'D:\\University\\Project\\ewap_dataset_full\\ewap_dataset\\seq_hotel\\obsmat.txt', 'r')
        Lines = self.annotations.readlines()
        annotationsByFrame = {}
        for line in Lines:
            # ped_id, x_min, y_min, x_max, y_max, frame, _, _, _, label = line.strip().split(" ")
            line = line.strip().split(" ")
            line = list(filter(None, line))
            frame, ped_id, x, z, y, _, _, _ = line
            x = float(x)
            y = float(y)
            x_min, x_max = x, x
            y_min, y_max = y, y
            label = "\"\""
            frame = math.floor(float(frame))
            frame = str(frame)
            if (frame in annotationsByFrame):
                annotationsByFrame[frame].append([ped_id, x_min, y_min, x_max, y_max, label])
            else:
                annotationsByFrame[frame] = [[ped_id, x_min, y_min, x_max, y_max, label]]
        return annotations

    def get_frame(self):
        # extracting frames
        ret, frame = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', frame)
        return ret, jpeg.tobytes()
