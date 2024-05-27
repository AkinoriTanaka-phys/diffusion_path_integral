# Introduction

This repository contains the implementation of the experiments with 2D synthetic data conducted in our paper, "Understanding Diffusion Models by Feynman's Path Integral," https://arxiv.org/abs/2403.11262. In this repository, we support the entire process from model training to likelihood calculations on simple data.

We use the following libraries.
- `numpy`
- `sympy`
- `scipy`
- `matplotlib`
- `sklearn`
- `jax`
- `flax`
- `tqdm`
- `os`

# How to obtain the trained model checkpoint

Please run `get_trained_model.py`, for example:

```bash
$ python get_trained_model.py --data sr --sde simple --n 1000 --bs 32 --lr 0.001
```

Each argument means:
- `--data` should be `sr` or `25`, meaning swiss-roll and 25-gaussian for the training respectively.
- `--sde` shoule be `simple` or `cosine`.
- `--n` means number of epochs for the training
- `--bs` means minibatch size 
- `--lr` means learning rate

After the training, the model checkpoint will be saved to `f'{home}/checkpoints/diff_{data}_{sde}_{lr}_{n}_{bs}'`.
In the paper, we use `--n 16000 --bs 512 --lr 0.001`, but it takes a little time.

> (Note for time-independent models) This is not default, but we support training based on score-matching or denoising score-matching by function `get_trained_model()` in `training/train.py` (not `get_trained_model.py`) with option `training_scheme == "score matching"` or `training_scheme == "denoising score matching"`. Note that the corresponding checkpoints do not contain information for `sde` in this case.
> To execute sampling from these checkpoints, please use `step_overdamped_langevin()` defined in `inference/samplers.py`. The score function execution in these cases are defined as `state.s(x, t=None)`. To use `step_euler_maruyama()`, we need additional `sde` info.

# How to retrieve metrics from a checkpoint

Please run `measure_metrics.py`, for example:

```bash
$ python measure_metrics.py --ckpt_path checkpoints/diff_sr_simple_0.001_1000_32 --measure nll
```

Then, the nll (or precisely, empirical cross entropy) will be calculated. The log is at `{home}/nll/test/diff_sr_simple_0.001_1000_32/dx0.01inner_rtol1e-05inner_atol1e-05outer_rtol0.001outer_atol0.001err_est_model.txt`.

You can also measure 2-wasserstein:

```bash
$ python measure_metrics.py --ckpt_path checkpoints/diff_sr_simple_0.001_1000_32 --measure w2
```

Details on the argument of this script is as follows:
- `--ckpt_path` checkpoint path
- `--measure` should be `nll` or `we`
- `--head` should be 
    - `test`: runs relatively light program, and 
    - `paper`: runs it paper setting.
- `--dx`: means \Delta x in the paper. When you set `--measure w2`, this argument is ignored.
- `--err_estimate`: should be `model` or `subtraction`
- `--inner_rtol`: small number used for computation of log q_t inside
- `--inner_atol`: small number used for computation of log q_t inside
- `--outer_rtol`: small number used for outer integral
- `--outer_atol`: small number used for outer integral

> (Note for time-independent models) If one want to apply likelihood calculation with checkpoints trained by function `get_trained_model()` with option `training_scheme == "score matching"` or `training_scheme == "denoising score matching"` defined in `training/train.py`, it is necessary to add `sde` info to these checkpoints because the above subroutine read `sde` info automatically.

## Tips for stabilizing NLL calculation

In the main functions for nll calculation defined in `others/metrics.py`, i.e., 
- `get_nll_with_1st_correction_by_solve_ivp_outer_integral_with_subtract_err_estimate()` or
- `get_nll_with_1st_correction_by_solve_ivp_outer_integral_with_model_err_estimate()`,

we implemented some tips to stabilize the calculation process by excluding unstable solutions that occur with negligible frequency. The basic strategies are as follows.
1. if the `integrate.solve_ivp()` fails to calculate integral within required precision, we terminate inner subroutine and pop out the vectors for nll calculation to the log file, 
    - the functions raises `OverflowError` in this case  in `others/metrics.py`,
2. the above exception seems to be occurring when the ODE drives particles kicked out from the origin, so we implemented similar termination process in this case,
    - see line 11-15 in `others/metrics.py` for example.
