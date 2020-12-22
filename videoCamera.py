import os

import cv2
import numpy as np
import torch
import torch.distributions.multivariate_normal as torchdist

import metrics
import utils
from createAnnotations import annotations
from model import social_stgcnn


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
        self.model = social_stgcnn(n_stgcnn=1, n_txpcnn=5,
                                   output_feat=5, seq_len=8,
                                   kernel_size=3, pred_seq_len=12).cuda()
        self.model.load_state_dict(
            torch.load(
                "C:\\Users\\brain\\PycharmProjects\\Social-STGCNN\\checkpoint\\social-stgcnn-hotel\\val_best.pth"))
        self.model.cuda()

    def __del__(self):
        # releasing camera
        self.video.release()

    def get_frame(self, displayCircles=True):
        # extracting frames
        ret, frame = self.video.read()
        annotations = self.annotations.getFrameAnnotations(self.video.get(cv2.CAP_PROP_POS_FRAMES))
        newPedPastTraj = {}
        keys = list(self.pedPastTraj.keys())
        # plot tracking circles and update past trajectories
        for annotation in annotations:
            if (displayCircles):
                self.displayAnnotation(frame, annotation)
            self.updatePastTraj(annotation, newPedPastTraj)
        self.pedPastTraj = newPedPastTraj

        # predict trajectories
        predTrajectories = self.predict()
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

    def predict(self, keys=None, samples=20):
        pedPastTraj = self.pedPastTraj
        if (keys is None):
            pedPastTraj = dict(filter(lambda elem: len(elem[1]) == 8, pedPastTraj.items()))
        else:
            pedPastTraj = dict(filter(lambda elem: elem[0] in keys and len(elem[1]) == 8, pedPastTraj.items()))
        if (len(pedPastTraj) > 2):
            seq_list = []
            for key in sorted(list(pedPastTraj.keys())):
                pedestrianSeq = pedPastTraj[key]
                xcoords = []
                ycoords = []
                for coords in pedestrianSeq:
                    x, y = utils.centerCoord(coords)

                    xcoords.append(x)
                    ycoords.append(y)
                seq_list.append([np.array([xcoords, ycoords])])
            seq_list = np.concatenate(seq_list, axis=0)
            seq_list_rel = utils.convertToRelativeSequence(seq_list)
            obs_traj = torch.from_numpy(
                seq_list).type(torch.float)
            obs_traj_rel = torch.from_numpy(
                seq_list_rel).type(torch.float)
            V_obs = []
            A_obs = []
            v_, a_ = utils.seq_to_graph(obs_traj, obs_traj_rel, True)

            V_obs.append(v_.clone())
            A_obs.append(a_.clone())
            V_obs = torch.stack(V_obs).cuda()
            A_obs = torch.stack(A_obs).cuda()
            V_pred, _ = self.model(V_obs.permute(0, 3, 1, 2), A_obs.squeeze())
            V_pred = V_pred.permute(0, 2, 3, 1)
            V_pred = V_pred.squeeze()
            num_of_objs = obs_traj_rel.shape[1]
            # V_pred = V_pred[:, :num_of_objs, :]

            V_x = metrics.seq_to_nodes(obs_traj.data.cpu().numpy().copy())

            sx = torch.exp(V_pred[:, :, 2])  # sx
            sy = torch.exp(V_pred[:, :, 3])  # sy
            corr = torch.tanh(V_pred[:, :, 4])  # corr

            cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).cuda()
            cov[:, :, 0, 0] = sx * sx
            cov[:, :, 0, 1] = corr * sx * sy
            cov[:, :, 1, 0] = corr * sx * sy
            cov[:, :, 1, 1] = sy * sy
            mean = V_pred[:, :, 0:2]

            # create a guassian distribution and sample it
            mvnormal = torchdist.MultivariateNormal(mean, cov)
            # shape 12 - frames, n - pedestrians, 2 - x,y
            trajectories = []
            # take multiple samples and average
            for i in range(samples):
                V_pred = mvnormal.sample()
                # take predictions and add the pedestrians start pos to find actual position
                trajectories.append(metrics.nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                                   V_x[-1, :, :].copy()))
            return np.average(trajectories, axis=0)

        else:
            return []
