import argparse
import json
import pickle
from multiprocessing.spawn import freeze_support

from torch import optim
from torch.utils.data import DataLoader

import config
import trainingDataCreator
from metrics import *
from model import *
from utils import *


def train(model, epoch, optimizer, trainingData, metrics):
    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(trainingData)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(trainingData):
        batch_count += 1

        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        optimizer.zero_grad()
        # Forward
        # V_obs = batch,seq,node,feat
        # V_obs_tmp = batch,feat,seq,node
        V_obs_tmp = V_obs.permute(0, 3, 1, 2).contiguous()

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())

        V_pred = V_pred.permute(0, 2, 3, 1).contiguous()

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_tr)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss = l + loss

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            loss.backward()

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            # Metrics
            loss_batch = loss.item() + loss_batch
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

    metrics['train_loss'].append(loss_batch / batch_count)


def valid(model, epoch, checkpoint_dir, validationData, metrics, constant_metrics):
    model.eval()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(validationData)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(validationData):
        batch_count += 1

        # Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, \
        loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        V_obs_tmp = V_obs.permute(0, 3, 1, 2).contiguous()

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())

        V_pred = V_pred.permute(0, 2, 3, 1).contiguous()

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_tr)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss = l + loss

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            # Metrics
            loss_batch = loss.item() + loss_batch
            print('VALD:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

    metrics['val_loss'].append(loss_batch / batch_count)

    if abs(metrics['val_loss'][-1]) < abs(constant_metrics['min_val_loss']):
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'val_best.pth'))  # OK


def graph_loss(V_pred, V_target):
    return bivariate_loss(V_pred, V_target)


def start_training(datasetLocation, sampling_rate=15, num_epochs=250):
    checkpointLocation = datasetLocation + "-" + str(sampling_rate)
    print('*' * 30)
    print("Training initiating....")
    print(args)

    # Data prep
    obs_seq_len = args.obs_seq_len
    pred_seq_len = args.pred_seq_len
    data_set = os.path.join('trainingData', datasetLocation)
    dset_train = TrajectoryDataset(
        os.path.join(data_set, 'train'),
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1, norm_lap_matr=True)
    loader_train = DataLoader(
        dset_train,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=True,
        num_workers=0)

    dset_val = TrajectoryDataset(
        os.path.join(data_set, 'val'),
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1, norm_lap_matr=True)

    loader_val = DataLoader(
        dset_val,
        batch_size=1,  # This is irrelative to the args batch size parameter
        shuffle=True,
        num_workers=0)

    # Defining the model
    if (os.path.exists(os.path.join(data_set, 'normalising.json'))):
        with open(os.path.join(data_set, 'normalising.json')) as f:
            normalising_data = json.load(f)
        model = social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                              output_feat=args.output_size, seq_len=args.obs_seq_len,
                              kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len,
                              mean=normalising_data["mean"], std=normalising_data["std"]).cuda()
    else:
        model = social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                              output_feat=args.output_size, seq_len=args.obs_seq_len,
                              kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len).cuda()

    # Training settings
    # todo sgd vs adam
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if args.use_lrschd:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)

    checkpoint_dir = os.path.join('checkpoint', checkpointLocation)
    if not (config.labels is None):
        checkpoint_labels = ""
        for i in range(len(config.labels)):
            if (i == 0):
                checkpoint_labels += config.labels[i]
            else:
                checkpoint_labels += ("-" + config.labels[i])
        checkpoint_dir = os.path.join(checkpoint_dir, checkpoint_labels)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(os.path.join(checkpoint_dir, 'args.pkl'), 'wb') as fp:
        pickle.dump(args, fp)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)

    # Training
    metrics = {'train_loss': [], 'val_loss': []}
    constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}

    print('Training started ...')
    for epoch in range(num_epochs):
        train(model, epoch, optimizer, loader_train, metrics)
        valid(model, epoch, checkpoint_dir, loader_val, metrics, constant_metrics)
        if args.use_lrschd:
            scheduler.step()

        print('*' * 30)
        print('Epoch:', checkpointLocation, ":", epoch)
        for k, v in metrics.items():
            if len(v) > 0:
                print(k, v[-1])

        print(constant_metrics)
        print('*' * 30)

        with open(os.path.join(checkpoint_dir, 'metrics.pkl'), 'wb') as fp:
            pickle.dump(metrics, fp)

        with open(os.path.join(checkpoint_dir, 'constant_metrics.pkl'), 'wb') as fp:
            pickle.dump(constant_metrics, fp)


if __name__ == '__main__':
    freeze_support()
    if (config.annotationType == "stanford"):
        print("Converting Stanford Dataset...")
        trainingDataCreator.createTrainingData("trainingData\\stanford", "trainingData\\stanfordProcessed",
                                               samplingRate=config.samplingRate,
                                               labels=config.labels)
        print("Done")
    parser = argparse.ArgumentParser()

    # Model specific parameters
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--n_stgcnn', type=int, default=1, help='Number of ST-GCNN layers')
    parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
    parser.add_argument('--kernel_size', type=int, default=3)

    # Data specifc paremeters
    parser.add_argument('--obs_seq_len', type=int, default=8)
    parser.add_argument('--pred_seq_len', type=int, default=12)

    # Training specifc parameters
    parser.add_argument('--batch_size', type=int, default=512,
                        help='minibatch size')
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gadient clipping')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=150,
                        help='number of steps to drop the lr')
    parser.add_argument('--use_lrschd', action="store_true", default=True,
                        help='Use lr rate scheduler')

    args = parser.parse_args()
    start_training(config.path, sampling_rate=config.samplingRate, num_epochs=config.epochs)
