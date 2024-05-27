import argparse

import os
import jax
import numpy as np

from others.data import prepare_swissroll_data, prepare_25gaussian_data
from others.serialize import save, load
from others.metrics import w2

from training.train import get_trained_model
from training.src.diffusion import get_sde_with

from inference.samplers import show_gen_trajectories

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="sr")
    parser.add_argument('--sde', type=str, default="simple")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n', type=int, default=16000)
    parser.add_argument('--bs', type=int, default=512)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    home = os.getcwd()
    if args.data=="sr":
        x_train = prepare_swissroll_data(3000)
    elif args.data=="25":
        x_train = prepare_25gaussian_data(3000)
    else:
        raise NotImplementedError("--data should be sr or 25")
    
    if not (args.sde in ["simple", "cosine"]):
        raise NotImplementedError("--sde should be simple or cosine")
    
    ckpt_path = f'{home}/checkpoints/diff_{args.data}_{args.sde}_{args.lr}_{args.n}_{args.bs}'
    if os.path.exists(ckpt_path):
        raise FileExistsError(f"please change the setting or remove {ckpt_path}")
    sde = get_sde_with(args.sde) 
    ckpt = get_trained_model(x_train, learning_rate=args.lr, num_epochs=args.n, batch_size=args.bs, 
                                         training_scheme = "diffusion", sde=sde) 
    show_gen_trajectories(x_train, ckpt["model"], sde, h=1, show_N=4)
    save(ckpt, ckpt_path)