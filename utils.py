import math
import os

import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset
from tqdm import tqdm

import config


def to_image_frame(Hinv, loc):
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

def get_index_of_one_hot(enc):
    return list(config.one_hot_encoding.values()).index(enc)

def centerCoord(coordArray):
    coordArray = [float(x) for x in coordArray]
    x_min, y_min, x_max, y_max = coordArray
    return (x_min + x_max) / 2.0, (y_min + y_max) / 2.0


def convertToRelativeSequence(sequence):
    rel_curr_ped_seq = np.zeros(sequence.shape)
    rel_curr_ped_seq[:, :, 1:] = sequence[:, :, 1:] - [sequence[:, :, :-1]]
    return rel_curr_ped_seq


def seq_to_graph(seq_, seq_rel, norm_lap_matr=True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    A = np.zeros((seq_len, max_nodes, max_nodes))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]
            A[s, h, h] = 1
            for k in range(h + 1, len(step_)):
                l2_norm = anorm(step_rel[h], step_rel[k])
                A[s, h, k] = l2_norm
                A[s, k, h] = l2_norm
        if norm_lap_matr:
            G = nx.from_numpy_matrix(A[s, :, :])
            A[s, :, :] = nx.normalized_laplacian_matrix(G).toarray()

    return torch.from_numpy(V).type(torch.float), \
           torch.from_numpy(A).type(torch.float)


def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / (NORM)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            if (len(line) == 5):
                for i in range(len(line)):
                    try:
                        line[i] = float(line[i])
                    except ValueError:
                        line[i] = str(line[i])
                data.append(line)
    return np.asarray(data, dtype=object)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory trainingData"""

    def __init__(
            self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
            min_ped=1, delim='space', norm_lap_matr=True, scaleData=False, scaler=None):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()
        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        if (scaleData and scaler is None):
            self.vScaler = RobustScaler()
        elif (scaleData):
            self.vScaler = scaler
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        seq_list_class = []
        loss_mask_list = []
        non_linear_ped = []
        for path in all_files:
            data = read_file(path, delim)
            if (np.array_equal(data, [])):
                print(str(path) + " - No data in file")
                continue
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))
            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)

                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq_class = np.empty((len(peds_in_curr_seq)), dtype=object)
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq[:, :-1] = np.round(np.asarray(curr_ped_seq[:, :-1], dtype=float), decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    classEncoding = np.asarray(config.one_hot_encoding[curr_ped_seq[-1][0]], dtype=float)
                    curr_ped_seq = np.array(curr_ped_seq[:-1], dtype=float)
                    if ((curr_ped_seq.shape[1] != self.seq_len) or (pad_end - pad_front != self.seq_len)):
                        continue
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    curr_seq_class[_idx] = classEncoding
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1
                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    seq_list_class.append(curr_seq_class[:num_peds_considered])
        self.num_seq = len(seq_list)
        if not (np.array_equal(seq_list, [])):
            seq_list = np.concatenate(seq_list, axis=0)
            seq_list_rel = np.concatenate(seq_list_rel, axis=0)
            seq_list_class = np.concatenate(seq_list_class, axis=0)
            loss_mask_list = np.concatenate(loss_mask_list, axis=0)
            non_linear_ped = np.asarray(non_linear_ped)
            # Convert numpy -> Torch Tensor
            self.C_obs = torch.tensor(np.stack(seq_list_class)).type(torch.float)
            self.obs_traj = torch.from_numpy(
                seq_list[:, :, :self.obs_len]).type(torch.float)
            self.pred_traj = torch.from_numpy(
                seq_list[:, :, self.obs_len:]).type(torch.float)
            self.obs_traj_rel = torch.from_numpy(
                seq_list_rel[:, :, :self.obs_len]).type(torch.float)
            self.pred_traj_rel = torch.from_numpy(
                seq_list_rel[:, :, self.obs_len:]).type(torch.float)
            self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
            self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
            cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
            self.seq_start_end = [
                (start, end)
                for start, end in zip(cum_start_idx, cum_start_idx[1:])
            ]
            # Convert to Graphs
            self.v_obs = []
            self.A_obs = []
            self.v_pred = []
            self.A_pred = []
            print("Processing Data .....")
            pbar = tqdm(total=len(self.seq_start_end))
            for ss in range(len(self.seq_start_end)):
                pbar.update(1)

                start, end = self.seq_start_end[ss]

                v_, a_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], self.norm_lap_matr)
                self.v_obs.append(v_.clone())
                self.A_obs.append(a_.clone())
                v_, a_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :],
                                      self.norm_lap_matr)
                self.v_pred.append(v_.clone())
                self.A_pred.append(a_.clone())
            if (scaleData):
                v_obs_list = []
                for v in self.v_obs:
                    for obj in v.data.cpu().tolist():
                        v_obs_list += obj
                if (scaler is None):
                    self.vScaler.fit(v_obs_list)
                for i in range(len(self.v_obs)):
                    v = self.v_obs[i]
                    new_v = []
                    for obj in v.data.cpu().tolist():
                        new_v.append(self.vScaler.transform(obj))
                    self.v_obs[i] = torch.tensor(new_v).type(torch.float)

            pbar.close()

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index], self.C_obs[start:end]

        ]
        return out
