
import torch
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import argparse
import time
import util
import os
from util import *
import random
from model import STTBCL
from ranger21 import Ranger


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="")
parser.add_argument("--data", type=str, default="PEMS08", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="number of input_dim")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.0001, help="weight decay rate"
)
parser.add_argument("--epochs", type=int, default=120, help="")
parser.add_argument("--channels", type=int, default=96, help="")
parser.add_argument("--print_every", type=int, default=50, help="")
parser.add_argument(
    "--save",
    type=str,
    default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-",
    help="save path",
)
parser.add_argument("--expid", type=int, default=1, help="experiment id")
parser.add_argument(
    "--es_patience",
    type=int,
    default=100,
    help="quit if no improvement after this many iterations",
)
parser.add_argument("--trainflag", type=str, default='Fusion', help="")
parser.add_argument("--lambda_t", type=float, default=0.1, help="weight of CTSD temporal self-supervised loss L_T")
parser.add_argument("--lambda_s", type=float, default=0.1, help="weight of PivotalGraphGuidedRegionRoleLearning spatial self-supervised loss L_S")
parser.add_argument("--temporal_filter_delta", type=int, default=1, help="neighbor radius for CTSD negative time-context sampling; delta=1 avoids adjacent time steps")
parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision training to save GPU memory")

args = parser.parse_args()


class trainer:
    def __init__(
            self,
            scaler,
            input_dim,
            num_nodes,
            channels,
            dropout,
            lrate,
            wdecay,
            device,
            granularity,
            supports,
            lambda_t=0.1,
            lambda_s=0.1,
            use_amp=False,
            temporal_filter_delta=1,
    ):
        self.model = STTBCL(
            device,
            input_dim,
            num_nodes,
            channels,
            granularity,
            supports,
            dropout,
            temporal_filter_delta=temporal_filter_delta,
        )
        self.model.to(device)
        self.optimizer = Ranger(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = util.MAE_torch
        self.scaler = scaler
        self.clip = 5

        self.lambda_T = lambda_t
        self.lambda_S = lambda_s

        self.use_amp = bool(use_amp) and ("cuda" in str(device))
        self.amp_scaler = GradScaler(enabled=self.use_amp)

        print("The number of parameters: {}".format(self.model.param_num()))
        print("lambda_T = {}, lambda_S = {}".format(self.lambda_T, self.lambda_S))
        print("temporal_filter_delta = {}".format(temporal_filter_delta))
        print("AMP enabled = {}".format(self.use_amp))
        print(self.model)

    def train(self, inputx, real_val):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=self.use_amp):
            output, loss_t, loss_s = self.model(inputx)
            output = output.transpose(1, 3)  # [B, 1, N, T_out]
            real = torch.unsqueeze(real_val, dim=1)  # [B, 1, N, T_out]

            predict = self.scaler.inverse_transform(output)
            maeloss = self.loss(predict, real, 0.0)

            loss = maeloss + self.lambda_T * loss_t + self.lambda_S * loss_s

        if self.use_amp:
            self.amp_scaler.scale(loss).backward()
            if self.clip is not None:
                self.amp_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            loss.backward()
            if self.clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()

        return (
            loss.item(),
            mape,
            rmse,
            wmape,
            maeloss.item(),
            loss_t.item(),
            loss_s.item(),
        )

    def eval(self, inputx, real_val):
        self.model.eval()
        with torch.no_grad():
            output, _, _ = self.model(inputx)
        output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)
        maeloss = self.loss(predict, real, 0.0)
        mape = util.MAPE_torch(predict, real, 0.0).item()
        rmse = util.RMSE_torch(predict, real, 0.0).item()
        wmape = util.WMAPE_torch(predict, real, 0.0).item()
        return maeloss.item(), mape, rmse, wmape


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def main():
    seed_it(6666)

    data = args.data

    if args.data == "PEMS08":
        adj_path = "data//" + args.data + "//adj_" + args.data + '.pkl'
        args.data = "data//" + args.data
        num_nodes = 170
        granularity = 288
        channels = 96
        wave = 'coif1'

    elif args.data == "PEMS03":
        adj_path = "data//" + args.data + "//adj_" + args.data + '.pkl'
        args.data = "data//" + args.data
        num_nodes = 358
        granularity = 288
        channels = 32
        wave = 'sym2'

    elif args.data == "PEMS04":

        adj_path = "data//" + args.data + "//adj_" + args.data + '.pkl'
        args.data = "data//" + args.data
        num_nodes = 307
        granularity = 288
        channels = 64
        wave = 'coif1'


    elif args.data == "PEMS07":
        adj_path = "data//" + args.data + "//adj_" + args.data + '.pkl'
        args.data = "data//" + args.data
        num_nodes = 883
        granularity = 288
        channels = 128
        wave = 'coif1'


    elif args.data == "bike_drop":
        adj_path = "data//" + args.data + "//adj_" + args.data + '.pkl'
        args.data = "data//" + args.data
        num_nodes = 250
        granularity = 48
        channels = 32
        wave = 'coif1'


    elif args.data == "bike_pick":
        adj_path = "data//" + args.data + "//adj_" + args.data + '.pkl'
        args.data = "data//" + args.data
        num_nodes = 250
        granularity = 48
        channels = 32
        wave = 'coif1'


    elif args.data == "taxi_drop":
        adj_path = "data//" + args.data + "//adj_" + args.data + '.pkl'
        args.data = "data//" + args.data
        num_nodes = 266
        granularity = 48
        channels = 96
        wave = 'coif1'

    elif args.data == "taxi_pick":
        adj_path = "data//" + args.data + "//adj_" + args.data + '.pkl'
        args.data = "data//" + args.data
        num_nodes = 266
        granularity = 48
        channels = 96
        wave = 'coif1'

    device = torch.device(args.device)
    adj_mx = util.load_adj(adj_path)
    supports = [torch.tensor(i).float().to(device) for i in adj_mx]

    dataloader = util.load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size
    )
    scaler = dataloader["scaler"]

    loss = 9999999
    test_log = 999999
    epochs_since_best_mae = 0
    path = args.save + data + "/"
    bestid = 0

    his_loss = []
    val_time = []
    train_time = []
    result = []
    test_result = []

    print(args)

    if not os.path.exists(path):
        os.makedirs(path)

    engine = trainer(
        scaler,
        args.input_dim,
        num_nodes,
        channels,
        args.dropout,
        args.learning_rate,
        args.weight_decay,
        device,
        granularity,
        supports,
        args.lambda_t,
        args.lambda_s,
        args.amp,
        args.temporal_filter_delta,
    )

    print("start training...", flush=True)

    for i in range(1, args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        train_wmape = []

        train_mae = []
        train_tloss = []
        train_sloss = []

        t1 = time.time()

        for iter, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)

            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])

            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_wmape.append(metrics[3])

            train_mae.append(metrics[4])
            train_tloss.append(metrics[5])
            train_sloss.append(metrics[6])

            if iter % args.print_every == 0:
                log = "Iter: {:03d}, Train Loss: {:.4f}, Train_MAE: {:.4f}, Train_TLoss: {:.4f}, Train_SLoss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}"
                print(
                    log.format(
                        iter,
                        train_loss[-1],
                        train_mae[-1],
                        train_tloss[-1],
                        train_sloss[-1],
                        train_rmse[-1],
                        train_mape[-1],
                        train_wmape[-1],
                    ),
                    flush=True,
                )

        t2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        valid_loss = []
        valid_mape = []
        valid_wmape = []
        valid_rmse = []

        s1 = time.time()

        for iter, (x, y) in enumerate(dataloader["val_loader"].get_iterator()):
            testx = torch.Tensor(x).to(device).transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_wmape.append(metrics[3])

        s2 = time.time()

        log = "Epoch: {:03d}, Inference Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

        mtrain_mae = np.mean(train_mae)
        mtrain_tloss = np.mean(train_tloss)
        mtrain_sloss = np.mean(train_sloss)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_wmape = np.mean(train_wmape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_wmape = np.mean(valid_wmape)
        mvalid_rmse = np.mean(valid_rmse)

        his_loss.append(mvalid_loss)
        train_m = dict(
            train_mae=np.mean(train_mae),
            train_tloss=np.mean(train_tloss),
            train_sloss=np.mean(train_sloss),
            train_ssl_loss=np.mean(train_tloss) + np.mean(train_sloss),
            train_loss=np.mean(train_loss),
            train_rmse=np.mean(train_rmse),
            train_mape=np.mean(train_mape),
            train_wmape=np.mean(train_wmape),
            valid_loss=np.mean(valid_loss),
            valid_rmse=np.mean(valid_rmse),
            valid_mape=np.mean(valid_mape),
            valid_wmape=np.mean(valid_wmape),
        )
        train_m = pd.Series(train_m)
        result.append(train_m)

        log = "Epoch: {:03d}, Train Loss: {:.4f}, Train_MAE: {:.4f}, Train_TLoss: {:.4f}, Train_SLoss: {:.4f}, Train RMSE: {:.4f}, Train MAPE: {:.4f}, Train WMAPE: {:.4f}, "
        print(
            log.format(i, mtrain_loss, mtrain_mae, mtrain_tloss, mtrain_sloss, mtrain_rmse, mtrain_mape, mtrain_wmape),
            flush=True,
        )
        log = "Epoch: {:03d}, Valid MAELoss: {:.4f}, Valid RMSE: {:.4f}, Valid MAPE: {:.4f}, Valid WMAPE: {:.4f}"
        print(
            log.format(i, mvalid_loss, mvalid_rmse, mvalid_mape, mvalid_wmape),
            flush=True,
        )

        if mvalid_loss < loss:
            print("###Update tasks appear###")
            if i < 80:
                loss = mvalid_loss
                torch.save(engine.model.state_dict(), path + "best_model.pth")
                bestid = i
                epochs_since_best_mae = 0
                print("Updating! Valid Loss:", mvalid_loss, end=", ")
                print("epoch: ", i)

            elif i > 80:
                outputs = []
                realy = torch.Tensor(dataloader["y_test"]).to(device)
                realy = realy.transpose(1, 3)[:, 0, :, :]

                for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
                    testx = torch.Tensor(x).to(device).transpose(1, 3)

                    with torch.no_grad():
                        preds, _, _ = engine.model(testx)
                        preds = preds.transpose(1, 3)
                    outputs.append(preds.squeeze())

                yhat = torch.cat(outputs, dim=0)
                yhat = yhat[: realy.size(0), ...]

                amae = []
                amape = []
                awmape = []
                armse = []

                for j in range(12):
                    pred = scaler.inverse_transform(yhat[:, :, j])
                    real = realy[:, :, j]
                    metrics = util.metric(pred, real)
                    log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
                    print(
                        log.format(
                            j + 1, metrics[0], metrics[2], metrics[1], metrics[3]
                        )
                    )

                    amae.append(metrics[0])
                    amape.append(metrics[1])
                    armse.append(metrics[2])
                    awmape.append(metrics[3])

                log = "On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
                print(
                    log.format(
                        np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape)
                    )
                )

                if np.mean(amae) < test_log:
                    test_log = np.mean(amae)
                    loss = mvalid_loss
                    torch.save(engine.model.state_dict(), path + "best_model.pth")
                    epochs_since_best_mae = 0
                    print("Test low! Updating! Test Loss:", np.mean(amae), end=", ")
                    print("Test low! Updating! Valid Loss:", mvalid_loss, end=", ")
                    bestid = i
                    print("epoch: ", i)
                else:
                    epochs_since_best_mae += 1
                    print("No update")

        else:
            epochs_since_best_mae += 1
            print("No update")

        train_csv = pd.DataFrame(result)
        train_csv.round(8).to_csv(f"{path}/train.csv")
        if epochs_since_best_mae >= args.es_patience and i >= 300:
            break

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    print("Training ends")
    print("The epoch of the best result：", bestid)
    if bestid > 0:
        print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))

    engine.model.load_state_dict(torch.load(path + "best_model.pth"))
    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            preds, _, _ = engine.model(testx)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]

    amae = []
    amape = []
    armse = []
    awmape = []

    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
        print(log.format(i + 1, metrics[0], metrics[2], metrics[1], metrics[3]))

        test_m = dict(
            test_loss=np.mean(metrics[0]),
            test_rmse=np.mean(metrics[2]),
            test_mape=np.mean(metrics[1]),
            test_wmape=np.mean(metrics[3]),
        )
        test_m = pd.Series(test_m)
        test_result.append(test_m)

        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    log = "On average over 12 horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}"
    print(log.format(np.mean(amae), np.mean(armse), np.mean(amape), np.mean(awmape)))

    test_m = dict(
        test_loss=np.mean(amae),
        test_rmse=np.mean(armse),
        test_mape=np.mean(amape),
        test_wmape=np.mean(awmape),
    )
    test_m = pd.Series(test_m)
    test_result.append(test_m)

    test_csv = pd.DataFrame(test_result)
    test_csv.round(8).to_csv(f"{path}/test.csv")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
