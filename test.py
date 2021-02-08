import copy
import glob
import pickle
from multiprocessing.spawn import freeze_support

import torch.distributions.multivariate_normal as torchdist
from torch.utils.data import DataLoader

import trainingDataCreator
from metrics import *
from model import social_stgcnn
from utils import *


def test(KSTEPS=20):
    global loader_test, model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step = 0
    for batch in loader_test:
        step += 1
        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr, obs_classes = batch

        num_of_objs = obs_traj_rel.shape[1]

        # Forward
        # V_obs = batch,seq,node,feat
        # V_obs_tmp = batch,feat,seq,node
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        V_obs_tmp[:, :, :1] = V_obs_tmp[:, :, :1]
        V_obs_tmp[:, :, 1:] = V_obs_tmp[:, :, 1:]
        V_pred, _ = model(V_obs_tmp, A_obs.squeeze(), obs_classes)
        # print(V_pred.shape)
        # torch.Size([1, 5, 12, 2])
        # torch.Size([12, 2, 5])
        V_pred = V_pred.permute(0, 2, 3, 1)
        # torch.Size([1, 12, 2, 5])>>seq,node,feat
        # V_pred= torch.rand_like(V_tr).cuda()

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]
        # print(V_pred.shape)

        # For now I have my bi-variate parameters
        # normx =  V_pred[:,:,0:1]
        # normy =  V_pred[:,:,1:2]
        sx = torch.exp(V_pred[:, :, 2])  # sx
        sy = torch.exp(V_pred[:, :, 3])  # sy
        corr = torch.tanh(V_pred[:, :, 4])  # corr

        cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).cuda()
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = V_pred[:, :, 0:2]

        mvnormal = torchdist.MultivariateNormal(mean, cov)

        ### Rel to abs
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len 

        # Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                V_x[0, :, :].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                V_x[-1, :, :].copy())

        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        for n in range(num_of_objs):
            ade_ls[n] = []
            fde_ls[n] = []

        for k in range(KSTEPS):

            V_pred = mvnormal.sample()
            V_pred[:, :, :1] = V_pred[:, :, :1]
            V_pred[:, :, 1:] = V_pred[:, :, 1:]
            # V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                       V_x[-1, :, :].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))

            # print(V_pred_rel_to_abs.shape) #(12, 3, 2) = seq, ped, location
            for n in range(num_of_objs):
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n:n + 1, :])
                target.append(V_y_rel_to_abs[:, n:n + 1, :])
                obsrvs.append(V_x_rel_to_abs[:, n:n + 1, :])
                number_of.append(1)

                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))

        for n in range(num_of_objs):
            # todo mean vs min
            ade_bigls.append(np.mean(ade_ls[n]))
            fde_bigls.append(np.mean(fde_ls[n]))
    if (len(ade_bigls) > 0):
        ade_ = sum(ade_bigls) / len(ade_bigls)
    else:
        ade_ = math.inf
    if (len(fde_bigls) > 0):
        fde_ = sum(fde_bigls) / len(fde_bigls)
    else:
        fde_ = math.inf

    return ade_, fde_, raw_data_dict


def main():
    global loader_test, model
    if config.annotationType == "stanford":
        trainingDataCreator.createTrainingData("trainingData\\stanford", "trainingData\\stanfordProcessed",
                                               samplingRate=config.samplingRate,
                                               labels=config.labels)
    if (config.checkpoint is None):
        path = os.path.join('checkpoint', config.path + "-" + str(config.samplingRate))
        if not (config.labels is None):
            checkpoint_labels = ""
            for i in range(len(config.labels)):
                if (i == 0):
                    checkpoint_labels += config.labels[i]
                else:
                    checkpoint_labels += ("-" + config.labels[i])
            path = os.path.join(path, checkpoint_labels)
    else:
        path = config.checkpoint
    KSTEPS = 20

    print("*" * 50)
    print('Number of samples:', KSTEPS)
    print("*" * 50)

    exps = glob.glob(path)
    print('Model being tested are:', exps)
    for label in config.labels:
        for exp_path in exps:
            print("*" * 50)
            print("Evaluating model:", exp_path)

            model_path = os.path.join(exp_path, 'val_best.pth')
            args_path = os.path.join(exp_path, 'args.pkl')
            with open(args_path, 'rb') as f:
                args = pickle.load(f)

            stats = os.path.join(exp_path, 'constant_metrics.pkl')
            with open(stats, 'rb') as f:
                cm = pickle.load(f)
            print("Stats:", cm)

            # Data prep
            obs_seq_len = args.obs_seq_len
            pred_seq_len = args.pred_seq_len
            data_set = os.path.join('trainingData', config.path)

            dset_test = TrajectoryDataset(
                os.path.join(data_set, 'test', label),
                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=1, norm_lap_matr=True)

            loader_test = DataLoader(
                dset_test,
                batch_size=1,  # This is irrelative to the args batch size parameter
                shuffle=False,
                num_workers=1)

            # Defining the model
            model = social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                                  output_feat=args.output_size, seq_len=args.obs_seq_len,
                                  kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len).cuda()
            model.load_state_dict(torch.load(model_path))
            model.cuda()

            print(label + " Testing ....")
            ad, fd, raw_data_dic_ = test()
            print("ADE:", ad, " FDE:", fd)


if __name__ == '__main__':
    freeze_support()
    main()
