import os

import cv2
import numpy as np

import utils
from createAnnotations import annotations
from trajectoryPrediction import trajectoryPrediction


class VideoCamera(object):
    def __init__(self):
        self.path = "bookstore/video0"
        # capturing video
        self.video = cv2.VideoCapture('videos/' + self.path + '/video.mov')
        self.pedPastTraj = {}
        self.colours = {}
        self.annotations = annotations(self.path)
        homog_file = "annotations/" + self.path + "/H.txt"
        self.H = np.linalg.inv((np.loadtxt(homog_file))) if os.path.exists(homog_file) else np.eye(3)
        self.trajectoryPrediction = trajectoryPrediction()
        self.samplingRate = 5

    def __del__(self):
        # releasing camera
        self.video.release()

    def get_frame(self, displayCircles=True):
        # extracting frames
        ret, frame = self.video.read()
        frameNum = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
        annotations = self.annotations.getFrameAnnotations(frameNum)
        newPedPastTraj = {}
        keys = list(self.pedPastTraj.keys())
        # plot tracking circles and update past trajectories
        for annotation in annotations:
            if (displayCircles):
                self.displayAnnotation(frame, annotation)
            if (frameNum % self.samplingRate == 0):
                self.updatePastTraj(annotation, newPedPastTraj)
        if (frameNum % self.samplingRate == 0):
            self.pedPastTraj = newPedPastTraj

        # predict trajectories
        predTrajectories = self.trajectoryPrediction.predict(self.pedPastTraj)
        for framePrediction in predTrajectories:
            for i in range(len(framePrediction)):
                predX, predY = framePrediction[i]
                pos = [predX, predY]
                y, x = utils.to_image_frame(self.H, np.array(pos))
                cv2.circle(frame, center=(x, y), radius=3, color=self.colours[keys[i]], thickness=2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return ret, jpeg.tobytes()

    def displayAnnotation(self, frame, annotation):
        ped_id, x_min, y_min, x_max, y_max, label = annotation
        min_coords = np.array([x_min, y_min])
        max_coords = np.array([x_max, y_max])
        y_min, x_min = utils.to_image_frame(self.H, min_coords)
        y_max, x_max = utils.to_image_frame(self.H, max_coords)
        if (not (ped_id in self.colours)):
            self.colours[ped_id] = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)),
                                    int(np.random.randint(0, 255)))
        centerX, centerY = utils.centerCoord([x_min, y_min, x_max, y_max])
        centerX, centerY = int(centerX), int(centerY)
        if (eval(label) == "Pedestrian"):
            cv2.circle(frame, center=(centerX, centerY), radius=7, color=self.colours[ped_id],
                       thickness=-1)
        else:
            cv2.circle(frame, center=(centerX, centerY), radius=7, color=self.colours[ped_id],
                       thickness=-1)
        cv2.circle(frame, center=(centerX, centerY), radius=7, color=(0, 0, 0), thickness=2)

    def updatePastTraj(self, annotation, newPedPastTraj):
        ped_id, x_min, y_min, x_max, y_max, label = annotation
        if (ped_id in self.pedPastTraj):
            currentList = self.pedPastTraj[ped_id]
            if (len(currentList) > 7):
                newPedPastTraj[ped_id] = currentList[:7]
            else:
                newPedPastTraj[ped_id] = currentList
        else:
            newPedPastTraj[ped_id] = []
        newPedPastTraj[ped_id].append([x_min, y_min, x_max, y_max])

