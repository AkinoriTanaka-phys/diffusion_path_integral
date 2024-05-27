import orbax.checkpoint

from flax.training import orbax_utils

import jax
import jax.numpy as jnp
from training.src.score_matching import create_time_INdependent_train_state
from training.src.diffusion import create_time_dependent_train_state, get_sde_with

def save(ckpt, save_path=""):
    # ckpt = {'model': state, 'loss_hist': loss_hist}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(save_path, ckpt, save_args=save_args)

def load(save_path=""):
    ''' automatic sde loading is only available for the raw_restored ckpt that has raw_restored['sde_name'] in ['simple', 'cosine']... '''
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    raw_restored = orbax_checkpointer.restore(save_path)
    target = {'loss_hist': jnp.zeros(shape=(1)),
              'StateClass': raw_restored['StateClass'],
              'data_shape': raw_restored['data_shape'],
              'batch_size': raw_restored['batch_size'],
              'learning_rate': raw_restored['learning_rate'],
              'num_epochs': raw_restored['num_epochs']
              }
    
    if raw_restored['StateClass'] == 'FNNState':
        key = jax.random.PRNGKey(3)
        target_state = create_time_INdependent_train_state(key, learning_rate=0)
    elif raw_restored['StateClass'] == 'FNNtcState':
        key = jax.random.PRNGKey(3)
        target_state = create_time_dependent_train_state(key, learning_rate=0)
    target['model'] = target_state

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    ckpt_restored = orbax_checkpointer.restore(save_path, item=target)
    if raw_restored['sde_name'] in ['simple', 'cosine']:
        sde = get_sde_with(raw_restored['sde_name'])
        ckpt_restored['sde'] = sde
    return ckpt_restored




