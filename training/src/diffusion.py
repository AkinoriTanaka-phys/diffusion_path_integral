import sympy as sp 
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.training import train_state

import optax

### class for sde dx = f(t)x + g(t)dw
### `sym` means symbol object

class SDE():
  Tmax:float
  Tmin:float   # should be close to 0
  name:str

  def __init__(self, f_sym, g_sym, t_sym, Tmax=1, Tmin=0.01):
     self.f_sym = f_sym
     self.g_sym = g_sym
     self.t_sym = t_sym
     self.Tmax = Tmax
     self.Tmin = Tmin
     self.f = None
     self.g = None
     self.alpha = None
     self.sigma = None

  def get_drift_sym(self):
     dt_sym = sp.Symbol('dt', commutative=False)
     x_sym = sp.Symbol('x_t', commutative=False)
     return self.f_sym*x_sym*dt_sym
  
  def get_noise_sym(self):
      dw_sym = sp.Symbol('dw_t', commutative=False)
      return self.g_sym*dw_sym

  def load_alpha_sym(self):
    alpha_sym = sp.exp(sp.integrate(self.f_sym, (self.t_sym, 0, self.t_sym)))
    self.alpha_sym = sp.simplify(alpha_sym)

  def load_sigma_sym(self):
    try:
      sigma_sym = (self.alpha_sym**2*sp.integrate((self.g_sym/self.alpha_sym)**2, (self.t_sym, 0, self.t_sym)))**(1/2)
      self.sigma_sym = sp.simplify(sigma_sym)
    except AttributeError:
      print("ERROR: .load_sigma_sym()")
      print("    please execute `.load_alpha_sym()` or define variable symbolic `.alpha_sym` beforehand.")

  def load_integrals_sym(self):
    self.load_alpha_sym()
    self.load_sigma_sym()

  def load_f_g_alpha_sigma_for_array(self):
    self.alpha = sp.lambdify(self.t_sym, self.alpha_sym)
    self.sigma = sp.lambdify(self.t_sym, self.sigma_sym)  # TODO: make integral numerical??
    self.f = sp.lambdify(self.t_sym, self.f_sym)
    self.g = sp.lambdify(self.t_sym, self.g_sym)

  def load_t_alpha_sigma_g_as_array(self, N_intervals=1000):
    self.t_intervals = jnp.linspace(self.Tmin, self.Tmax, N_intervals).reshape(-1,1)            # see def diffusion_train_step() for the reason of starting from Tmin.
    self.alpha_intervals = self.alpha(self.t_intervals)*jnp.ones_like(self.t_intervals)
    self.sigma_intervals = self.sigma(self.t_intervals)*jnp.ones_like(self.t_intervals)
    self.g_intervals = self.g(self.t_intervals)*jnp.ones_like(self.t_intervals)

  def plot_SN_schedules(self):
    t = jnp.linspace(self.Tmin, self.Tmax,100)
    plt.plot(t, self.alpha(t), label=r"$\alpha(t)$")
    plt.plot(t, self.sigma(t), label=r"$\sigma(t)$")
    #plt.plot(t,g(t), label=r"$g(t)$")
    plt.title(fr"SDE: $dx_t = {sp.latex(self.get_drift_sym())} + {sp.latex(self.get_noise_sym())}$")
    plt.legend()
    plt.show()

def get_sde_with(scheduler):
  t_sym = sp.Symbol('t')

  if scheduler == "cosine":
    # ref: https://arxiv.org/abs/2102.09672
    f_sym = -sp.pi*sp.tan(sp.pi/2*t_sym)/2
    g_sym = (-2*f_sym)**(1/2)
    sde = SDE(f_sym, g_sym, t_sym)

    sde.load_alpha_sym()
    sde.sigma_sym = (1-sde.alpha_sym**2)**(1/2)

    sde.f = sp.lambdify(t_sym, f_sym)
    sde.g = sp.lambdify(t_sym, g_sym)
    sde.alpha = sp.lambdify(t_sym, sde.alpha_sym)
    sde.sigma = sp.lambdify(t_sym, sde.sigma_sym)
    sde.Tmax = 1-1e-3     # to avoid overflow from tan(pi/2)
    sde.Tmin = 0.01
    sde.name = "cosine"

  elif scheduler == "simple":
    # simple version of SDE in https://arxiv.org/abs/2011.13456
    f_sym = -1/2*20*t_sym
    g_sym = (20*t_sym)**(1/2)
    sde = SDE(f_sym, g_sym, t_sym)
    sde.load_integrals_sym()
    sde.load_f_g_alpha_sigma_for_array()
    sde.Tmax = 1
    sde.Tmin = 0.01
    sde.name = "simple"

  else:
    raise NotImplementedError("choose  \"simple\" or \"cosine\" in get_sde_with()")
  
  return sde

### NN model

class FNNtc(nn.Module):
  dim_feature = 128

  @nn.compact
  def __call__(self, x, t):
    xt = jnp.concatenate([x, t], axis=1)
    h = nn.Dense(self.dim_feature)(xt)
    h = nn.swish(h)

    h = nn.Dense(self.dim_feature)(h)
    h = nn.swish(h)

    h = nn.Dense(self.dim_feature)(h)
    h = nn.swish(h)

    h = nn.Dense(2)(h)
    return h
  
@jax.jit
def score_estimate_by(params, x, t):
    t = t*jnp.ones(shape=(x.shape[0], 1))
    return FNNtc().apply({'params': params}, x, t)

### state

class FNNtcState(train_state.TrainState):
    '''
    FNNtcState = object including
                 - neural net params
                 - optimizer
                 - estimate of the score function by state.s(x, t)
    '''
    def s(self, x, t):
       return score_estimate_by(self.params, x, t)       

def create_time_dependent_train_state(key, learning_rate, state=None):
    fnn = FNNtc()
    if state is None:
        params = fnn.init(key, jnp.ones([1, 2]), jnp.ones([1, 1]))['params']
    else:
      params = state.params
    state = FNNtcState.create(
              apply_fn = fnn.apply,
              params = params,
              tx = optax.adam(learning_rate),
              )
    return state

### training routines

@jax.jit
def diffusion_train_step(state, batch, key, t_intervals, alpha_intervals, sigma_intervals, g_intervals):
  '''
  training step based on KL bound loss 
  '''
  def loss_batch(params, key, t_intervals, alpha_intervals, sigma_intervals, g_intervals):
    s = lambda x,t :score_estimate_by(params, x, t)
    x = batch['input']
    z = jax.random.normal(key, shape=x.shape)

    n = jax.random.randint(key, shape=(x.shape[0],), minval=0, maxval=len(t_intervals))
    t = t_intervals[n]
    alpha_t = alpha_intervals[n]
    sigma_t = sigma_intervals[n]
    g_t = g_intervals[n]
    tx = alpha_t*x + sigma_t*z
    true_score_t = - (tx - alpha_t*x)/(sigma_t**2) # dividing by sigma potentially causes overflow

    loss = jnp.mean((s(tx, t) - true_score_t)**2*g_t**2)  
    return loss
  grad_fn = jax.grad(loss_batch, argnums=0) 
  grads = grad_fn(state.params, key, t_intervals, alpha_intervals, sigma_intervals, g_intervals) 
  state = state.apply_gradients(grads=grads)
  return state, loss_batch(state.params, key, t_intervals, alpha_intervals, sigma_intervals, g_intervals)