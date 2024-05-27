import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def step_overdamped_langevin(state, key, x, epsilon=0.1, mode="time backward"):
    z = jax.random.normal(key, shape=(x.shape[0], 2))
    if mode == "time backward":
        x = x + epsilon*state.s(x, t=jnp.ones(shape=(x.shape[0], 1))*.01) + jnp.sqrt(2*epsilon)*z
    else:
        x = x
    return x

def step_euler_maruyama(state, key, x, t, sde, h=1, epsilon=0.1, mode="time backward"):
    z = jax.random.normal(key, shape=(x.shape[0], 2))
    if mode == "time backward":
        x = x - sde.f(t)*epsilon*x + (1+h)/2*epsilon*sde.g(t)**2*state.s(x, t) + h**(1/2)*sde.g(t)*jnp.sqrt(epsilon)*z
    elif mode == "time forward":
        if h == 1:
            x = x + sde.f(t)*epsilon*x + sde.g(t)*jnp.sqrt(epsilon)*z
        else:
            x = x + sde.f(t)*epsilon*x - (1-h)/2*epsilon*sde.g(t)**2*state.s(x, t) + h**(1/2)*sde.g(t)*jnp.sqrt(epsilon)*z
    return x

def generate(state, sde, size, h=1, rng=jax.random.PRNGKey(3)):
    N_steps = 1000
    dt = (sde.Tmax-sde.Tmin)/N_steps

    x = jax.random.normal(rng, shape=(size, 2))
    for n in range(N_steps+1):
        t = sde.Tmax-n*dt
        rng, key = jax.random.split(rng)
        x = step_euler_maruyama(state, key, x, t, sde, h=h, epsilon=dt, mode="time backward")
    return x


def show_gen_trajectories(x_train, state, sde, h=1, show_N=10, rng=jax.random.PRNGKey(3), color="blue"):
    show_n = 1
    
    plt.figure(figsize=(show_N*4.5,4))
    plt.suptitle(r"$\mathfrak{h}=$"+f"{h}", fontsize=20)

    N_steps = 1000
    dt = (sde.Tmax-sde.Tmin)/N_steps
    size = 2000

    x = jax.random.normal(rng, shape=(size, 2))
    for n in range(N_steps+1):
        t = sde.Tmax-n*dt
        rng, key = jax.random.split(rng)
        x = step_euler_maruyama(state, key, x, t, sde, h=h, epsilon=dt, mode="time backward")
        if (n>3*N_steps/4 or n==0) and n%(N_steps//show_N//4)==0:
            plt.subplot(1, show_N+1, show_n)
            plt.title(f"t={t:.2f}")
            plt.scatter(x_train[:, 0], x_train[:, 1], alpha=.9, marker=".", label="validation data", color="gray", rasterized=True)
            plt.scatter(x[:, 0], x[:, 1], alpha=.3, marker=".", label="generated", color=color, rasterized=True)
            plt.legend()
            show_n += 1
    plt.show()