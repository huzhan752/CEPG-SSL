# -*- coding: utf-8 -*-
"""
STTBCL 模型测试脚本

用法示例：
    python test.py --data PEMS08 --checkpoint logs/hz08PEMS08/best_model.pth

也兼容 notebook：
    %run test.py --data PEMS08 --checkpoint logs/hz08PEMS08/best_model.pth

目录要求（默认）：
    data/PEMS08/train.npz
    data/PEMS08/val.npz
    data/PEMS08/test.npz
    data/PEMS08/adj_PEMS08.pkl
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 保证从任意工作目录运行时，都能导入同目录下的 model.py / util.py
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import util  # noqa: E402
from model import STTBCL  # noqa: E402


DATASET_CONFIG = {
    "PEMS03": {"num_nodes": 358, "granularity": 288, "channels": 32},
    "PEMS04": {"num_nodes": 307, "granularity": 288, "channels": 64},
    "PEMS07": {"num_nodes": 883, "granularity": 288, "channels": 128},
    "PEMS08": {"num_nodes": 170, "granularity": 288, "channels": 96},
    "bike_drop": {"num_nodes": 250, "granularity": 48, "channels": 32},
    "bike_pick": {"num_nodes": 250, "granularity": 48, "channels": 32},
    "taxi_drop": {"num_nodes": 266, "granularity": 48, "channels": 96},
    "taxi_pick": {"num_nodes": 266, "granularity": 48, "channels": 96},
}


def seed_everything(seed: int = 6666) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate STTBCL on test set")
    parser.add_argument("--data", type=str, default="PEMS08", help="数据集名，如 PEMS08；或数据目录路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径，如 logs/.../best_model.pth")
    parser.add_argument("--adj", type=str, default=None, help="邻接矩阵 pkl 路径；不填则默认 data/<data>/adj_<data>.pkl")
    parser.add_argument("--device", type=str, default=None, help="cuda:0 / cpu；不填则自动选择")
    parser.add_argument("--batch_size", type=int, default=64, help="测试 batch size")
    parser.add_argument("--input_dim", type=int, default=None, help="输入特征维度；通常为 3，不填则从 checkpoint 或数据推断")
    parser.add_argument("--channels", type=int, default=None, help="模型通道数；不填则按数据集或 checkpoint 推断")
    parser.add_argument("--granularity", type=int, default=None, help="一天的时间片数；PEMS=288，bike/taxi=48")
    parser.add_argument("--dropout", type=float, default=0.1, help="模型 dropout，测试时不会启用，但需与结构初始化一致")
    parser.add_argument("--strict", action="store_true", help="严格加载 checkpoint；默认允许报告并忽略不匹配键")
    parser.add_argument("--result_csv", type=str, default=None, help="保存测试指标 csv 的路径，可选")
    parser.add_argument("--save_preds", type=str, default=None, help="保存预测和真实值 npz 的路径，可选")
    parser.add_argument("--seed", type=int, default=6666, help="随机种子")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    """优先按用户当前工作目录解析，其次按 test.py 所在目录解析。"""
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    p2 = SCRIPT_DIR / path_str
    if p2.exists():
        return p2.resolve()
    return p.resolve()


def resolve_data_dir(data_arg: str) -> Path:
    p = Path(data_arg)
    if p.exists():
        return p.resolve()

    candidates = [
        Path.cwd() / "data" / data_arg,
        SCRIPT_DIR / "data" / data_arg,
        SCRIPT_DIR.parent / "data" / data_arg,
        Path.cwd() / data_arg,
        SCRIPT_DIR / data_arg,
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    # 返回最常见的默认位置，后续会给出清晰报错
    return (Path.cwd() / "data" / data_arg).resolve()


def get_state_dict(checkpoint_path: Path, map_location):
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    if not isinstance(ckpt, dict):
        raise TypeError(f"checkpoint 格式不支持：{type(ckpt)}")

    # 兼容 DataParallel 保存的 module.xxx
    state_dict = {}
    for k, v in ckpt.items():
        new_k = k[7:] if k.startswith("module.") else k
        state_dict[new_k] = v
    return state_dict


def infer_from_checkpoint(state_dict: dict) -> dict:
    info = {}
    if "start_conv.weight" in state_dict:
        # [channels, input_dim, 1, 1]
        w = state_dict["start_conv.weight"]
        info["channels"] = int(w.shape[0])
        info["input_dim"] = int(w.shape[1])
    if "Temb.time_day" in state_dict:
        # [granularity, channels]
        w = state_dict["Temb.time_day"]
        info["granularity"] = int(w.shape[0])
        info.setdefault("channels", int(w.shape[1]))
    if "mgf_emb" in state_dict:
        # [hidden_channels, num_nodes, input_len]
        w = state_dict["mgf_emb"]
        info["num_nodes"] = int(w.shape[1])
    return info


def infer_from_data(data_dir: Path) -> dict:
    test_npz = data_dir / "test.npz"
    if not test_npz.exists():
        return {}
    arr = np.load(test_npz)
    x = arr["x"]
    info = {}
    # 训练代码的 x 形状通常是 [B, T, N, C]
    if x.ndim >= 4:
        info["num_nodes"] = int(x.shape[2])
        info["input_dim"] = int(x.shape[3])
    return info


def build_model(args, state_dict, data_dir: Path, device: torch.device):
    dataset_name = Path(args.data).name
    cfg = dict(DATASET_CONFIG.get(dataset_name, {}))
    cfg.update(infer_from_data(data_dir))
    cfg.update(infer_from_checkpoint(state_dict))

    # 命令行参数优先级最高
    if args.input_dim is not None:
        cfg["input_dim"] = args.input_dim
    if args.channels is not None:
        cfg["channels"] = args.channels
    if args.granularity is not None:
        cfg["granularity"] = args.granularity

    missing = [k for k in ["input_dim", "num_nodes", "channels", "granularity"] if k not in cfg]
    if missing:
        raise ValueError(
            "无法推断这些模型参数：{}。请手动传入，例如 "
            "--input_dim 3 --channels 96 --granularity 288".format(", ".join(missing))
        )

    adj_path = resolve_path(args.adj) if args.adj else data_dir / f"adj_{dataset_name}.pkl"
    if not adj_path.exists():
        raise FileNotFoundError(
            f"找不到邻接矩阵：{adj_path}\n"
            f"请用 --adj 指定，例如 --adj data/{dataset_name}/adj_{dataset_name}.pkl"
        )

    adj_mx = util.load_adj(str(adj_path))
    supports = [torch.tensor(i).float().to(device) for i in adj_mx]

    model = STTBCL(
        device=device,
        input_dim=int(cfg["input_dim"]),
        num_nodes=int(cfg["num_nodes"]),
        channels=int(cfg["channels"]),
        granularity=int(cfg["granularity"]),
        supports=supports,
        dropout=args.dropout,
    ).to(device)

    print("Model config:")
    print(f"  data_dir    : {data_dir}")
    print(f"  adj_path    : {adj_path}")
    print(f"  input_dim   : {cfg['input_dim']}")
    print(f"  num_nodes   : {cfg['num_nodes']}")
    print(f"  channels    : {cfg['channels']}")
    print(f"  granularity : {cfg['granularity']}")
    print(f"  params      : {model.param_num():,}")
    return model


@torch.no_grad()
def evaluate(model, dataloader, scaler, device: torch.device):
    model.eval()
    outputs = []

    realy = torch.tensor(dataloader["y_test"], dtype=torch.float32, device=device)
    realy = realy.transpose(1, 3)[:, 0, :, :]  # [B, N, 12]

    start = time.time()
    for x, _ in dataloader["test_loader"].get_iterator():
        testx = torch.tensor(x, dtype=torch.float32, device=device).transpose(1, 3)  # [B, C, N, T]
        preds, _, _ = model(testx)                  # [B, 12, N, 1]
        preds = preds.transpose(1, 3).squeeze(1)    # [B, N, 12]
        outputs.append(preds)

    inference_time = time.time() - start
    yhat = torch.cat(outputs, dim=0)[: realy.size(0), ...]

    horizon_rows = []
    mae_list, mape_list, rmse_list, wmape_list = [], [], [], []

    for horizon in range(yhat.shape[-1]):
        pred = scaler.inverse_transform(yhat[:, :, horizon])
        real = realy[:, :, horizon]
        mae, mape, rmse, wmape = util.metric(pred, real)

        row = {
            "horizon": horizon + 1,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "wmape": wmape,
        }
        horizon_rows.append(row)
        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)
        wmape_list.append(wmape)

        print(
            "Evaluate on test data for horizon {:02d}, "
            "Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}".format(
                horizon + 1, mae, rmse, mape, wmape
            )
        )

    avg = {
        "horizon": "avg",
        "mae": float(np.mean(mae_list)),
        "rmse": float(np.mean(rmse_list)),
        "mape": float(np.mean(mape_list)),
        "wmape": float(np.mean(wmape_list)),
    }
    horizon_rows.append(avg)

    print(
        "On average over {} horizons, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test WMAPE: {:.4f}".format(
            yhat.shape[-1], avg["mae"], avg["rmse"], avg["mape"], avg["wmape"]
        )
    )
    print("Inference time: {:.4f} secs".format(inference_time))

    return horizon_rows, yhat, realy


def main():
    args = parse_args()
    seed_everything(args.seed)

    device_name = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    checkpoint_path = resolve_path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")

    data_dir = resolve_data_dir(args.data)
    for name in ["train.npz", "val.npz", "test.npz"]:
        if not (data_dir / name).exists():
            raise FileNotFoundError(f"找不到数据文件：{data_dir / name}")

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = get_state_dict(checkpoint_path, map_location=device)

    model = build_model(args, state_dict, data_dir, device)
    load_info = model.load_state_dict(state_dict, strict=args.strict)
    if not args.strict:
        if load_info.missing_keys:
            print("Missing keys:", load_info.missing_keys)
        if load_info.unexpected_keys:
            print("Unexpected keys:", load_info.unexpected_keys)

    dataloader = util.load_dataset(
        str(data_dir),
        batch_size=args.batch_size,
        valid_batch_size=args.batch_size,
        test_batch_size=args.batch_size,
    )
    scaler = dataloader["scaler"]

    rows, yhat, realy = evaluate(model, dataloader, scaler, device)

    if args.result_csv:
        import pandas as pd

        result_path = resolve_path(args.result_csv)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).round(8).to_csv(result_path, index=False)
        print(f"Saved result csv to: {result_path}")

    if args.save_preds:
        pred_path = resolve_path(args.save_preds)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            pred_path,
            pred=yhat.detach().cpu().numpy(),
            real=realy.detach().cpu().numpy(),
        )
        print(f"Saved predictions to: {pred_path}")


if __name__ == "__main__":
    main()
