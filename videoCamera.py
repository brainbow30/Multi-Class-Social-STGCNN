import os

import cv2
import numpy as np

import config
import utils
from createAnnotations import annotations
from trajectoryPrediction import trajectoryPrediction


class VideoCamera(object):
    def __init__(self, samplingRate):
        self.path = config.path
        # capturing video
        self.video = cv2.VideoCapture('videos/' + self.path + '/video.mov')
        self.pedPastTraj = {}
        self.colours = {}
        self.annotations = annotations(self.path)
        homog_file = "annotations/" + self.path + "/H.txt"
        self.H = np.linalg.inv((np.loadtxt(homog_file))) if os.path.exists(homog_file) else np.eye(3)
        self.samplingRate = samplingRate
        self.trajectoryPrediction = trajectoryPrediction(self.path, self.samplingRate)
        self.predTrajectories = []

    def __del__(self):
        # releasing camera
        self.video.release()

    def get_frame(self, displayCircles=True):
        # extracting frames
        ret, frame = self.video.read()
        frameNum = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
        annotations = self.annotations.getFrameAnnotations(frameNum)
        newPedPastTraj = {}
        keys = sorted(list(self.pedPastTraj.keys()))
        # plot tracking circles and update past trajectories
        for annotation in annotations:
            if (displayCircles):
                self.displayAnnotation(frame, annotation)
            if (frameNum % self.samplingRate == 0):
                newPedPastTraj = self.updatePastTraj(annotation, newPedPastTraj)
        if (frameNum % (self.samplingRate) == 0):
            self.pedPastTraj = newPedPastTraj
        if (frameNum % (self.samplingRate * 6) == 0):
            # predict trajectories
            self.predTrajectories = self.trajectoryPrediction.predict(self.pedPastTraj, samples=config.predSamples)
            keys = sorted(list(self.pedPastTraj.keys()))
        prevFrame = None
        # view past trajectory
        # for key in self.pedPastTraj.keys():
        #     pastMovement = self.pedPastTraj[key]
        #     for i in range(len(pastMovement)):
        #         x, y = utils.centerCoord(pastMovement[i])
        #         cv2.circle(frame, center=(int(x), int(y)), radius=3, color=self.colours[key], thickness=2)
        for framePrediction in self.predTrajectories:
            for i in range(len(framePrediction)):
                predX, predY = framePrediction[i]
                pos = [predX, predY]
                x, y = utils.to_image_frame(self.H, np.array(pos))
                # todo work out why frameprediction length can be larger than colours
                if (i < len(keys)):
                    cv2.circle(frame, center=(x, y), radius=4, color=self.colours[keys[i]], thickness=-1)
                if (not (prevFrame is None)):
                    predX, predY = prevFrame[i]
                    pos = [predX, predY]
                    x2, y2 = utils.to_image_frame(self.H, np.array(pos))
                    if (i < len(keys)):
                        cv2.line(frame, (x, y), (x2, y2), self.colours[keys[i]], 2)
            prevFrame = framePrediction
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
        if (label == "\"Biker\""):
            if (ped_id in self.pedPastTraj):
                currentList = self.pedPastTraj[ped_id]
                if (len(currentList) > 7):
                    newPedPastTraj[ped_id] = currentList[1:]
                else:
                    newPedPastTraj[ped_id] = currentList
            else:
                newPedPastTraj[ped_id] = []
            newPedPastTraj[ped_id].append([x_min, y_min, x_max, y_max])
        return newPedPastTraj
