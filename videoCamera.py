import math
import os

import cv2
import numpy as np


class VideoCamera(object):
    def __init__(self):
        self.path = "bookstore/video0"
        # capturing video
        self.video = cv2.VideoCapture('videos/' + self.path + '/video.mov')
        self.annotations = self.getAnnotations()
        self.pedPastTraj = {}
        self.colours = {}
        homog_file = "annotations/" + self.path + "/H.txt"
        self.H = np.linalg.inv((np.loadtxt(homog_file))) if os.path.exists(homog_file) else np.eye(3)

    def __del__(self):
        # releasing camera
        self.video.release()

    def centerCoord(self, coordArray):
        coordArray = [float(x) for x in coordArray]
        x_min, y_min, x_max, y_max = coordArray
        return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0

    def to_image_frame(self, Hinv, loc):
        """
        Given H^-1 and world coordinates, returns (u, v) in image coordinates.
        """

        if loc.ndim > 1:
            locHomogenous = np.hstack((loc, np.ones((loc.shape[0], 1))))
            loc_tr = np.transpose(locHomogenous)
            loc_tr = np.matmul(Hinv, loc_tr)  # to camera frame
            locXYZ = np.transpose(loc_tr / loc_tr[2])  # to pixels (from millimeters)
            imgCoord = locXYZ[:, :2].astype(int)
        else:
            locHomogenous = np.hstack((loc, 1))
            locHomogenous = np.dot(Hinv, locHomogenous.astype(float))  # to camera frame
            locXYZ = locHomogenous / locHomogenous[2]  # to pixels (from millimeters)
            imgCoord = locXYZ[:2].astype(int)
        if (np.array_equal(np.eye(3), Hinv)):
            imgCoord = np.flip(imgCoord)
        return imgCoord

    def getAnnotations(self):
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

    def get_frame(self):
        # extracting frames
        ret, frame = self.video.read()
        annotations = self.annotations[str(int(self.video.get(cv2.CAP_PROP_POS_FRAMES)))]
        newPedPastTraj = {}

        for annotation in annotations:
            ped_id, x_min, y_min, x_max, y_max, label = annotation
            min_coords = np.array([x_min, y_min])
            max_coords = np.array([x_max, y_max])
            if (ped_id in self.pedPastTraj):
                currentList = self.pedPastTraj[ped_id]
                if (len(currentList) > 7):
                    newPedPastTraj[ped_id] = currentList[:7]
                else:
                    newPedPastTraj[ped_id] = currentList
            else:
                newPedPastTraj[ped_id] = []

            newPedPastTraj[ped_id].append([x_min, y_min, x_max, y_max])

            y_min, x_min = self.to_image_frame(self.H, min_coords)
            y_max, x_max = self.to_image_frame(self.H, max_coords)
            if (not (ped_id in self.colours)):
                self.colours[ped_id] = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)),
                                        int(np.random.randint(0, 255)))
            centerX, centerY = self.centerCoord([x_min, y_min, x_max, y_max])
            centerX, centerY = int(centerX), int(centerY)
            if (eval(label) == "Pedestrian"):
                cv2.circle(frame, center=(centerX, centerY), radius=7, color=self.colours[ped_id],
                           thickness=-1)
            else:
                cv2.circle(frame, center=(centerX, centerY), radius=7, color=self.colours[ped_id],
                           thickness=-1)
            cv2.circle(frame, center=(centerX, centerY), radius=7, color=(0, 0, 0), thickness=2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return ret, jpeg.tobytes()
