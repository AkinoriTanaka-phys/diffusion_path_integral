import argparse

import os
import jax
import numpy as np

from others.data import prepare_swissroll_data, prepare_25gaussian_data
from others.serialize import load
from others.metrics import get_w2_along_h, get_nll_with_1st_correction_by_solve_ivp_outer_integral_with_model_err_estimate, get_nll_with_1st_correction_by_solve_ivp_outer_integral_with_subtract_err_estimate

from training.train import get_trained_model
from training.src.diffusion import get_sde_with


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--measure', type=str, default='nll') # w2, nll
    parser.add_argument('--head', type=str, default='test')
    parser.add_argument('--dx', type=float, default=0.01)     # used for nabla and laplacian. if args.measure == "w2" it will be just ignored.
    parser.add_argument('--err_estimate', type=str, default='model')  # subtraction is also fine
    parser.add_argument('--inner_rtol', type=float, default=1e-5)     # used for computation of log q_t inside
    parser.add_argument('--inner_atol', type=float, default=1e-5)     # used for computation of log q_t inside
    parser.add_argument('--outer_rtol', type=float, default=1e-3)     # used for outer integral
    parser.add_argument('--outer_atol', type=float, default=1e-3)     # used for outer integral
    return parser.parse_args()

def get_data_labels_dir_from(ckpt_path):
    labels_str = ckpt_path.split("/")[-1]
    labels_dir = {"train": labels_str.split("_")[0],
                  "data": labels_str.split("_")[1],
                  "sde": labels_str.split("_")[2],
                  "learning_rate": labels_str.split("_")[3],
                  "num_epochs": labels_str.split("_")[4],
                  "batch_size": labels_str.split("_")[5]}
    return labels_dir

def get_relative_path_infos(ckpt_path):
    labels_str = ckpt_path.split("/")[-1]
    labels_dir = get_data_labels_dir_from(ckpt_path)
    
    head_relative_path = f'{args.measure}/{args.head}'
    save_relative_path = f'{head_relative_path}/{labels_str}'
    return head_relative_path, save_relative_path, labels_dir

def measure_w2s(ckpt_path):
    # load ckpt & make save_path    
    home = os.getcwd()
    ckpt = load(f"{home}/{ckpt_path}")
    head_relative_path, save_relative_path, labels_dir = get_relative_path_infos(ckpt_path)
    save_path = f"{home}/{save_relative_path}.txt"
    if args.head == "test":
        h_steps = 2
        N_trials = 2
        N_sde_steps = 10
    elif args.head == "paper":
        h_steps = 10
        N_trials = 10
        N_sde_steps = 1000
    else:
        raise NotImplementedError("--head should be test or paper")
    
    if not os.path.exists(f'{home}/{head_relative_path}'):
        os.makedirs(f'{home}/{head_relative_path}')

    # generate validation data
    data_size = ckpt["data_shape"][0]
    if labels_dir["data"] == "sr":
        x_train = prepare_swissroll_data(data_size)
        with open(save_path, 'a') as f:
            f.write(f"{data_size} swiss roll data generated\n")
    elif labels_dir["data"] == "25":
        x_train = prepare_25gaussian_data(data_size)
        with open(save_path, 'a') as f:
            f.write(f"{data_size} 25 gaussian data\n")

    # main
    hs, w2ss = get_w2_along_h(x_train, state=ckpt["model"], sde=ckpt["sde"], 
                              h_steps = h_steps,
                              N_trials = N_trials,
                              N_sde_steps = N_sde_steps)
    with open(save_path, 'a') as f:
        f.write(f"h: {hs.tolist()}\n")
        f.write(f"mean: {[np.mean(w2s) for w2s in w2ss]}\n")
        f.write(f"std: {[np.std(w2s) for w2s in w2ss]}\n")
        f.write(f"std/sqrt(n): {[np.std(w2s)/len(w2s)**(1/2) for w2s in w2ss]}\n")
        f.write("raw data:\n")
        f.write(f"{w2ss.tolist()}\n")

def measure_nll(ckpt_path):
    # load ckpt & make save_path    
    home = os.getcwd()
    ckpt = load(f"{home}/{ckpt_path}")
    head_relative_path, save_relative_path, labels_dir = get_relative_path_infos(ckpt_path)
    if args.head == "test":
        N_outer_trials = 10
        N_inner_trials = 10
    elif args.head == "paper":
        N_outer_trials = 100
        N_inner_trials = 10
    else:
        raise NotImplementedError("--head should be test or paper")
    
    if args.err_estimate == "model":
        nll_estimator = get_nll_with_1st_correction_by_solve_ivp_outer_integral_with_model_err_estimate
    elif args.err_estimate == "subtraction":
        nll_estimator = get_nll_with_1st_correction_by_solve_ivp_outer_integral_with_subtract_err_estimate
    else:
        raise NotImplementedError("--err_estimate should be model or subtraction")

    if not os.path.exists(f'{home}/{head_relative_path}'):
        os.makedirs(f'{home}/{head_relative_path}')
    if not os.path.exists(f'{home}/{save_relative_path}'):
        os.makedirs(f'{home}/{save_relative_path}')

    save_path = f"{home}/{save_relative_path}/dx{args.dx}inner_rtol{args.inner_rtol}inner_atol{args.inner_atol}outer_rtol{args.outer_rtol}outer_atol{args.outer_atol}err_est_{args.err_estimate}.txt"

    # generate validation data-generator
    data_size = ckpt["data_shape"][0]
    if labels_dir["data"] == "sr":
        data_generator = prepare_swissroll_data
        with open(save_path, 'a') as f:
            f.write(f"swiss roll data-generator will be used.\n")
    elif labels_dir["data"] == "25":
        data_generator = prepare_25gaussian_data
        with open(save_path, 'a') as f:
            f.write(f"25 gaussian data-generator will be used.\n")

    # main
    data_size = ckpt["data_shape"][0]
    rtol_dir = {"inner": args.inner_rtol, "outer": args.outer_rtol}
    atol_dir = {"inner": args.inner_atol, "outer": args.outer_atol}
    nlls = []
    corrs = []
    corrs_errs = []
    for _ in range(N_outer_trials):
        x0 = data_generator(data_size)[:N_inner_trials]
        with open(save_path, 'a') as f:
            f.write(f"# new samples are generated, and start solving ODE\n")

        nll, corr, err = nll_estimator(x0, ckpt["model"], ckpt["sde"], dx=args.dx, rtol_dir=rtol_dir, atol_dir=atol_dir, save_path=save_path)
        if len(nll)==0: # integral failed somewhere
            with open(save_path, 'a') as f:
                f.write(f"\nnew data excluded: {x0.tolist()}\n\n")
        else:
            nlls.append(np.array(nll).tolist())
            corrs.append(np.array(corr).tolist())
            corrs_errs.append(np.array(err).tolist())
            
            nll_flatten = np.array(sum(nlls, [])).reshape(-1)
            corr_flatten = np.array(sum(corrs, [])).reshape(-1)
            corrs_errs_flatten = np.array(sum(corrs_errs, [])).reshape(-1)
            with open(save_path, 'a') as f:
                f.write(f"ODE calculation completed with new data {x0.tolist()}\n")
                f.write(f"## nll statistics\n")
                f.write(f"  nll: mean={np.mean(nll_flatten)}, std={np.std(nll_flatten)}, std/sqrt(n)={np.std(nll_flatten)/len(nll_flatten)**(1/2)}, n={len(nll_flatten)}\n")
                f.write(f"  corr: mean={np.mean(corr_flatten)}, std={np.std(corr_flatten)}, std/sqrt(n)={np.std(corr_flatten)/len(corr_flatten)**(1/2)}, n={len(corr_flatten)}\n")
                f.write(f"  └── corr error: mean={np.mean(corrs_errs_flatten)}, std={np.std(corrs_errs_flatten)}, std/sqrt(n)={np.std(corrs_errs_flatten)/len(corrs_errs_flatten)**(1/2)}, n={len(corrs_errs_flatten)}\n\n")

if __name__ == '__main__':
    args = parse_args()    
    ckpt_path = args.ckpt_path

    if args.measure == 'w2':
        measure_w2s(ckpt_path)
    if args.measure == 'nll':
        measure_nll(ckpt_path)