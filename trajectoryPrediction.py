import numpy as np
import torch
import torch.distributions.multivariate_normal as torchdist

import metrics
import utils
from model import social_stgcnn


class trajectoryPrediction(object):
    def __init__(self, path, frame_skip, checkpoint=None):
        self.model = social_stgcnn(n_stgcnn=1, n_txpcnn=5,
                                   output_feat=5, seq_len=8,
                                   kernel_size=3, pred_seq_len=12).cuda()
        if (checkpoint is None):
            nnPath = "checkpoint\\" + path + "-" + str(frame_skip) + "\\val_best.pth"
        else:
            nnPath = checkpoint
        self.model.load_state_dict(
            torch.load(nnPath))
        self.model.cuda()

    def predict(self, pedPastTraj, keys=None, samples=20):
        if (keys is None):
            pedPastTraj = dict(filter(lambda elem: len(elem[1]) == 8, pedPastTraj.items()))
        else:
            pedPastTraj = dict(filter(lambda elem: elem[0] in keys and len(elem[1]) == 8, pedPastTraj.items()))
        if (len(pedPastTraj) > 2):
            seq_list = []
            for key in (sorted(list(pedPastTraj.keys()))):
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
            # todo check order
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
