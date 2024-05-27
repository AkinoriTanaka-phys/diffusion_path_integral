import tqdm
import jax
import jax.numpy as jnp

from .src.score_matching import create_time_INdependent_train_state, score_matching_step, denoising_score_matching_step
from .src.diffusion import create_time_dependent_train_state, diffusion_train_step, get_sde_with

def train_one_epoch(state, train_ds, batch_size, rng, train_step):
    train_ds_size = len(train_ds['input'])
    steps_per_epoch = train_ds_size // batch_size
    # for making mini-batch 
    rng, key = jax.random.split(rng)
    perms = jax.random.permutation(key, train_ds_size)
    del key
    perms = perms[:steps_per_epoch*batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    loss_values = []
    for perm in perms: # loop along mini-batches
        batch = {k: v[perm, ...] for k,v in train_ds.items()} 
        rng, key = jax.random.split(rng)
        state, loss = train_step(state, batch, key)
        del key
        loss_values.append(loss)
    return state, jnp.array(loss_values)

def get_trained_model(x_train, learning_rate=1e-3, num_epochs=1000, batch_size=32, seed=0, init_state=None, 
                      training_scheme="diffusion", sde=None):
    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)
    if training_scheme == "score matching":
        state = create_time_INdependent_train_state(key, learning_rate, state=init_state)
        train_step = score_matching_step
    elif training_scheme == "denoising score matching":
        state = create_time_INdependent_train_state(key, learning_rate, state=init_state)
        train_step = denoising_score_matching_step
    elif training_scheme == "diffusion":
        if sde is None:
            sde = get_sde_with("cosine")
        sde.load_t_alpha_sigma_g_as_array()
        state = create_time_dependent_train_state(key, learning_rate, state=init_state)
        train_step = lambda state, batch, key: diffusion_train_step(state, batch, key, 
                                                                    sde.t_intervals, 
                                                                    sde.alpha_intervals, 
                                                                    sde.sigma_intervals, 
                                                                    sde.g_intervals)
    else:
        raise NotImplementedError("`get_trained_model()` supports only with `training_scheme = \"score matching\"/\"denoising score matching\"/\"diffusion\".`")    

    train_ds = {"input": x_train }
    losses = []

    for epoch in tqdm.tqdm(range(1, num_epochs + 1)):
      rng, input_rng = jax.random.split(rng)
      state, loss_values = train_one_epoch(state, train_ds, batch_size, input_rng, train_step)
      losses.append(jnp.mean(loss_values))

    ckpt = {'model': state, 
            'loss_hist': jnp.array(losses),
            'StateClass': type(state).__name__,
            'data_shape': x_train.shape,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            }
    if training_scheme == "diffusion":
        ckpt['sde_name'] = sde.name
        ckpt['sde'] = sde
    return ckpt
    