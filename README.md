# Introduction

This repository contains the implementation of the experiments with 2D synthetic data conducted in our paper, "Understanding Diffusion Models by Feynman's Path Integral," https://arxiv.org/abs/2403.11262. In this repository, we support the entire process from model training to likelihood calculations on simple data.

## Requirements

To set up the environment for running this repository, please refer to the `requirements.txt` file. You can use 
```
pip install -r requirements.txt
```
but please note that the library versions listed are the ones we used during our tests. The code should work with other versions as well.

# How to obtain the trained model checkpoint

Please run `get_trained_model.py`, for example:

```bash
python get_trained_model.py --data sr --sde simple --n 1000 --bs 32 --lr 0.001
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
python measure_metrics.py --ckpt_path checkpoints/diff_sr_simple_0.001_1000_32 --measure nll
```

Then, the nll (or precisely, empirical cross entropy) will be calculated. The log is at `{home}/nll/test/diff_sr_simple_0.001_1000_32/dx0.01inner_rtol1e-05inner_atol1e-05outer_rtol0.001outer_atol0.001err_est_model.txt`.

You can also measure 2-wasserstein:

```bash
python measure_metrics.py --ckpt_path checkpoints/diff_sr_simple_0.001_1000_32 --measure w2
```

Details on the argument of this script is as follows:
- `--ckpt_path` checkpoint path
- `--measure` should be `nll` or `w2`
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

## How to read logs 


The log files with `--measure nll` option are formatted as follows. The part marked as # (which is not written in real log), corresponds intermediate state of the integral from time $T_{min}$ to $t$, and the part marked as ## (which is not written in real log) shows the results.
```
[data-generator] will be used.
...
# logs for solving ODE with the 1st 10 data formatted as
outer t:[ODE time]
x = [array for 10 x(t)]
delta x = [array for 10 delta x(t)]
└── delta x error: [array for 10 delta x(t) local error]
corrs = [log-likelihood correction value (coefficient of h) array at t]
└── corrs error: [its accumulated error array]
...
new data: [1st 10 data array]
  ## result of the integral up to the 1st 10 data formatted
  nll: mean=[mean(nll(10 data))], std=[its srd], std/sqrt(n)=[normalized value], n=10
  corr: mean=[mean(nll_corr(10 data))], std=[its srd], std/sqrt(n)=[normalized value], n=10
  └── corr error: mean=[mean(nll_corr_accumulated_error(10 data))], std=[its srd], std/sqrt(n)=[normalized value], n=10
...
# logs for solving ODE with the 2nd 10 data
...
new data: [2nd 10 data array]
  ## result of the integral up to the 2nd 10 data (i.e., 20 data if the 1st trial succeeded in)
...
```
If it failed to calculate nll with given 10 data, it outputs
```
new data excluded: [excluded data array]
```
to the same log file, and continue next 10 point calculation.


The log files with `--measure w2` option are formatted as follows:
```
3000 [data] generated
h: [array for h values used in the calculations]
mean: [array for mean w2 values]
std: [array for w2 values std]
std/sqrt(n): [normalized values]
raw data: [raw w2 array with shape (number of h values, number of datapoints in each calculation)]
```
The log file for w2 will be updated after all calculations are terminated.

## Tips for stabilizing NLL calculation

In the main functions for nll calculation defined in `others/metrics.py`, i.e., 
- `get_nll_with_1st_correction_by_solve_ivp_outer_integral_with_subtract_err_estimate()` or
- `get_nll_with_1st_correction_by_solve_ivp_outer_integral_with_model_err_estimate()`,

we implemented some tips to stabilize the calculation process by excluding unstable solutions that occur with negligible frequency. The basic strategies are as follows.
1. if the `integrate.solve_ivp()` fails to calculate integral within required precision, we terminate inner subroutine and pop out the vectors for nll calculation to the log file, 
    - the functions raises `OverflowError` in this case  in `others/metrics.py`,
2. the above exception seems to be occurring when the ODE drives particles kicked out from the origin, so we implemented similar termination process in this case,
    - see line 11-15 in `others/metrics.py` for example.
