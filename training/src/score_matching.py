import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.training import train_state

import optax

### NN model

class FNN(nn.Module):
  dim_feature = 32

  @nn.compact
  def __call__(self, x):
    h = nn.Dense(self.dim_feature)(x)
    h = nn.softplus(h)
    h = nn.Dense(self.dim_feature)(h)
    h = nn.softplus(h)
    h = nn.Dense(2)(h)
    return h

@jax.jit
def score_estimate_by(params, x):
    return FNN().apply({'params': params}, x)

### state

class FNNState(train_state.TrainState):
    '''
    FNNState = object including
               - neural net params
               - optimizer
               - estimate of the score function by state.s(x)
    '''
    sigma = None
    def s(self, x, t=None):
       return score_estimate_by(self.params, x)

def create_time_INdependent_train_state(key, learning_rate, state=None):
  fnn = FNN()
  if state is None:
      params = fnn.init(key, jnp.ones([1, 2]))['params'] 
  else:
      params = state.params
  state = FNNState.create(
            apply_fn = fnn.apply,
            params = params,
            tx=optax.adam(learning_rate),
            )
  return state

### training routines

@jax.jit
def score_matching_step(state, batch, key):
  '''
  training step based on naive score-matching loss 
  '''
  def loss_batch(params):
    s = lambda x: score_estimate_by(params, x)
    ds = jax.vmap(jax.jacfwd(s))
    x = batch['input']

    fst = jnp.sum(s(x)**2, axis=1)
    snd = 2*jnp.trace(ds(x), axis1=1, axis2=2)
    loss = jnp.mean(fst + snd)
    return loss
  grad_fn = jax.grad(loss_batch)
  grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss_batch(state.params)

@jax.jit
def denoising_score_matching_step(state, batch, key, sigma=.1):
  '''
  training step based on naive denoising score-matching loss 
  '''
  def loss_batch(params, key):
    s = lambda x: score_estimate_by(params, x)
    x = batch['input']
    z = jax.random.normal(key, shape=x.shape)
    tx = x + sigma*z
    true_score = - (tx-x)/sigma**2
    
    loss = jnp.mean((s(tx) - true_score)**2)
    return loss
  grad_fn = jax.grad(loss_batch, argnums=0) 
  grads = grad_fn(state.params, key)
  state = state.apply_gradients(grads=grads)
  return state, loss_batch(state.params, key)

