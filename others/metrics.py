import ot
import numpy as np

import jax
import jax.numpy as jnp
from scipy import integrate
import tqdm

from inference.samplers import step_euler_maruyama

# Score-function approximated by nn sometimes provides very strong force at t=sde.Tmin, and the orbit based on the probability flow ODE t: sde.Tmin -> sde.Tmax will be kicked out quickly. This strong force may cause overflow in `scipy.integrate.solve_ivp()`.
# To remedy this, we set bounds `X_MAX`, `X_MIN` for each coordinate as follows.
X_MAX = 5
X_MIN = -5
# see definition of `get_logq()` to see how it is used.

def w2(x, y):
    a, b = np.ones((len(x),)) / len(x), np.ones((len(y),)) / len(y)
    M = ot.dist(x, y)
    return ot.emd2(a, b, M)

def get_w2_along_h(x_train, state, sde, 
                   rng = jax.random.PRNGKey(3), 
                   h_steps = 10,
                   N_trials = 10,
                   N_sde_steps = 1000):
    hs = np.linspace(0, .2, h_steps)
    w2ss = []

    dt = (sde.Tmax-sde.Tmin)/N_sde_steps
    for h in hs:
        w2s = []
        for _ in range(N_trials):
            rng, key = jax.random.split(rng)
            x = jax.random.normal(key, shape=(x_train.shape[0], 2))
            for n in range(N_sde_steps+1):
                t = sde.Tmax-(sde.Tmax-sde.Tmin)*n*dt
                rng, key = jax.random.split(rng)
                x = step_euler_maruyama(state, key, x, t, sde, h=h, epsilon=dt, mode="time backward")
            x_np = np.array(x)
            w2s.append(w2(x_train, x_np))
        w2ss.append(w2s)
    return hs, np.array(w2ss)

## for llh

def llh_standard_normal(x):
    batchsize, dim = x.shape
    return -np.sum(x**2, axis=1)/2 - np.log(2*np.pi)*dim/2

def get_drift_for_probability_flow(x, t, state, sde):
    return sde.f(t)*x - sde.g(t)**2/2*state.s(x, t)

def directional_differential(x, f, v):
    ''' x.shape = v.shape = f(x).shape = (batchsize, dim=2) 
        for nabla \cdot s:
            `f = lambda x: jnp.expand_dims(get_div_s(x, 0.2, state, sde), axis=1)`
        will work with `2_f_dir` = 1
    '''
    dfx = jacobian(x, f)                       # shape = (2_f_dir  , 2_diff_dir, batchsize) see `def jacobian(x, f)`
    dfx = jnp.transpose(dfx, axes=[2, 1, 0])   # shape = (batchsize, 2_diff_dir, 2_f_dir  )
    v = jnp.expand_dims(v, axis=2)             # shape = (batchsize, 2,          1)
    v_dfx = jnp.sum(v*dfx, axis=1)             # shape = (batchsize, 2_f_dir)
    return v_dfx

def jacobian(x, f):
    " return (f_i, d_j, batch) "
    df = jax.jacfwd(f) 
    dfx = df(x)                                
    dfx = jnp.transpose(dfx, [0, 2, 1, 3]) 
    dfx = jnp.diagonal(dfx)
    return dfx

def get_div_drift(x, t, state, sde):
    f = lambda x: get_drift_for_probability_flow(x, t, state=state, sde=sde)
    dfx = jacobian(x, f)
    return jnp.trace(dfx)

def get_x_from(x_logq, batchsize, dim):
    return x_logq[:batchsize*dim].reshape(batchsize, dim)
def get_logq_from(x_logq, batchsize, dim):
    return x_logq[batchsize*dim:]
    
def get_logq(xt, t, state, sde, rtol, atol, method):
    batchsize, dim = xt.shape
    def func(t, x_logq):
        #print("inner t:", t)
        x = get_x_from(x_logq, batchsize, dim)
        drift = get_drift_for_probability_flow(x, t, state, sde)
        drift_flatten = np.array(drift).reshape(-1)
        div_drift = np.array(get_div_drift(x, t, state, sde))
        if not (np.prod(X_MIN < x) and np.prod(x < X_MAX)):
            raise OverflowError()
        return np.concatenate([drift_flatten, div_drift], axis=0)
    init = jnp.concatenate([xt.reshape(-1), np.zeros((batchsize))], axis=0)
    try:
        solution = integrate.solve_ivp(func, (t, sde.Tmax), init, rtol=rtol, atol=atol, method=method)
    except OverflowError:
        return jnp.array([]), None, None, False  

    x_traj = solution.y[:batchsize*dim].reshape(batchsize, dim, -1)
    xTlogT = solution.y[:, -1]
    xT = get_x_from(xTlogT, batchsize, dim)
    int_div_f = get_logq_from(xTlogT, batchsize, dim)
    logqT = llh_standard_normal(xT)
    logqt = logqT + int_div_f

    return logqt, solution.t, x_traj, solution.success

### log q derivatives smarter versions:

def get_nabla_logq_from(logq_xpos, logq_xneg, logq_ypos, logq_yneg, dx):
    logq_over_dx =  (logq_xpos - logq_xneg)/(2*dx)
    logq_over_dy =  (logq_ypos - logq_yneg)/(2*dx)
    
    logq_over_dx = logq_over_dx.reshape(-1, 1)
    logq_over_dy = logq_over_dy.reshape(-1, 1)

    out = jnp.concatenate([logq_over_dx, logq_over_dy], axis=1)
    return out

def get_laplacian_logq_from(logq, logq_xpos, logq_xneg, logq_ypos, logq_yneg, dx):
    out = (logq_xpos + logq_xneg + logq_ypos + logq_yneg - 4*logq)/dx**2
    return out

def get_div_s(x, t, state, sde):
    f = lambda x: state.s(x, t)
    return get_div(x, f)

def get_div(x, f):
    dfx = jacobian(x, f)
    return jnp.trace(dfx)

### for error estimates

def get_err_lap_logq(tol, dx, C=1):
    return jnp.abs(tol/dx**2 + C*dx**2)

def get_err_nabla_logq(tol, dx, C=1):
    return jnp.abs(tol/dx + C*dx**2)

### for nll calculation

def get_x_from_in_corr_subroutine(x_deltax_deltalogq, batchsize, dim):
    return x_deltax_deltalogq[: batchsize*dim].reshape(batchsize, dim)
def get_deltax_from_in_corr_subroutine(x_deltax_deltalogq, batchsize, dim):
    return x_deltax_deltalogq[batchsize*dim: 2*batchsize*dim].reshape(batchsize, dim)
def get_logq_from_in_corr_subroutine(x_deltax_deltalogq, batchsize, dim):
    return x_deltax_deltalogq[2*batchsize*dim: 2*batchsize*dim+batchsize]
def get_errdeltax_from_in_corr_subroutine(x_deltax_deltalogq, batchsize, dim):
    return x_deltax_deltalogq[2*batchsize*dim+batchsize: 2*batchsize*dim+3*batchsize].reshape(batchsize, dim)
def get_errlogqx_from_in_corr_subroutine(x_deltax_deltalogq, batchsize, dim):
    return x_deltax_deltalogq[2*batchsize*dim+3*batchsize: ]

def get_nll_with_1st_correction_by_solve_ivp_outer_integral_with_subtract_err_estimate(x0, state, sde, dx, rtol_dir, atol_dir, save_path):
    logq0, _, _, success = get_logq(x0, sde.Tmin, state, sde, rtol_dir["inner"], atol=atol_dir["inner"], method="RK45")
    if not success:
        return jnp.array([]), jnp.array([]), jnp.array([])
    
    batchsize, dim = x0.shape
    def func(t, x_deltax_deltalogq):
        #print(f"outer t:{t}")
        with open(save_path, 'a') as f:
            f.write(f"outer t:{t}\n")

        # 0th-logqSolver part
        xt = get_x_from_in_corr_subroutine(x_deltax_deltalogq, batchsize, dim)

        # logq for main calc
        rtol, atol = rtol_dir["inner"], atol_dir["inner"]
        logq, t_traj, x_traj, success = get_logq(xt, t, state, sde, rtol=rtol, atol=atol, method="RK45")
        if not success:
            raise OverflowError()
        logq_xpos, _, _, success = get_logq(xt + dx*np.tile(np.array([1,0]), (xt.shape[0], 1)), t, state, sde, rtol=rtol, atol=atol, method="RK45")
        if not success:
            raise OverflowError()
        logq_xneg, _, _, success = get_logq(xt - dx*np.tile(np.array([1,0]), (xt.shape[0], 1)), t, state, sde, rtol=rtol, atol=atol, method="RK45")
        if not success:
            raise OverflowError()
        logq_ypos, _, _, success = get_logq(xt + dx*np.tile(np.array([0,1]), (xt.shape[0], 1)), t, state, sde, rtol=rtol, atol=atol, method="RK45")
        if not success:
            raise OverflowError()
        logq_yneg, _, _, success = get_logq(xt - dx*np.tile(np.array([0,1]), (xt.shape[0], 1)), t, state, sde, rtol=rtol, atol=atol, method="RK45")
        if not success:
            raise OverflowError()
        
        # logq for precision error estimation
        rtol, atol = 1.1*rtol, 1.1*atol
        logq_for_error, t_traj, x_traj, success = get_logq(xt, t, state, sde, rtol=rtol, atol=atol, method="RK45")
        if not success:
            raise OverflowError()
        logq_xpos_for_error, _, _, success = get_logq(xt + dx*np.tile(np.array([1,0]), (xt.shape[0], 1)), t, state, sde, rtol=rtol, atol=atol, method="RK45")
        if not success:
            raise OverflowError()
        logq_xneg_for_error, _, _, success = get_logq(xt - dx*np.tile(np.array([1,0]), (xt.shape[0], 1)), t, state, sde, rtol=rtol, atol=atol, method="RK45")
        if not success:
            raise OverflowError()
        logq_ypos_for_error, _, _, success = get_logq(xt + dx*np.tile(np.array([0,1]), (xt.shape[0], 1)), t, state, sde, rtol=rtol, atol=atol, method="RK45")
        if not success:
            raise OverflowError()
        logq_yneg_for_error, _, _, success = get_logq(xt - dx*np.tile(np.array([0,1]), (xt.shape[0], 1)), t, state, sde, rtol=rtol, atol=atol, method="RK45")
        if not success:
            raise OverflowError()
        
        # Calc f^{PF}_\theta
        drift = get_drift_for_probability_flow(xt, t, state, sde)
        drift_flatten = np.array(drift).reshape(-1)
        
        # Calc \delta f^{PF}_\theta
        deltax = get_deltax_from_in_corr_subroutine(x_deltax_deltalogq, batchsize, dim)
        nabla_logq = get_nabla_logq_from(logq_xpos, logq_xneg, logq_ypos, logq_yneg, dx)
        ### for precition error calc 
        nabla_logq_for_error= get_nabla_logq_from(logq_xpos_for_error, logq_xneg_for_error, logq_ypos_for_error, logq_yneg_for_error, dx)
        ####
        f = lambda y: get_drift_for_probability_flow(y, t, state, sde)
        div_f = lambda x: jnp.expand_dims(get_div(x, f), axis=1)       # used below
        v_deltax1 = directional_differential(xt, f, deltax)
        v_deltax2 = -sde.g(t)**2/2*(state.s(xt, t) - nabla_logq)
        v_deltax = v_deltax1 + v_deltax2
        v_deltax_flatten = v_deltax.reshape(-1)

        # Calc \nabla \cdot \delta {\bm f}^{\rm PF}_{\theta}
        lap_log_q = get_laplacian_logq_from(logq, logq_xpos, logq_xneg, logq_ypos, logq_yneg, dx)
        ### for precition error calc 
        lap_log_q_for_error = get_laplacian_logq_from(logq_for_error, logq_xpos_for_error, logq_xneg_for_error, logq_ypos_for_error, logq_yneg_for_error, dx)
        ####
        div_s = get_div_s(xt, t, state, sde)
        corr1 = directional_differential(xt, div_f, deltax).reshape(-1)
        corr2 = -sde.g(t)**2/2*(div_s - lap_log_q)
        corr = corr1 + corr2
        corr_flatten = corr.reshape(-1)

        # Local error estimation
        err_deltax = get_errdeltax_from_in_corr_subroutine(x_deltax_deltalogq, batchsize, dim)

        err_nabla_logq = sde.g(t)**2/2*jnp.abs(nabla_logq - nabla_logq_for_error)
        err_nabla_logq += jnp.abs(directional_differential(xt, f, err_deltax))
        
        err_lap_logq = sde.g(t)**2/2*jnp.abs(lap_log_q - lap_log_q_for_error).reshape(-1, 1)
        err_lap_logq += jnp.abs(directional_differential(xt, div_f, err_deltax))

        err_nabla_logq_flatten = err_nabla_logq.reshape(-1)
        err_lap_logq_flatten = err_lap_logq.reshape(-1)

        with open(save_path, 'a') as f:
            f.write(f"x = {xt}\n")
            f.write(f"delta x = {deltax}\n")
            f.write(f"└── delta x error: {get_errdeltax_from_in_corr_subroutine(x_deltax_deltalogq, batchsize, dim)}\n")
            f.write(f"corrs = {get_logq_from_in_corr_subroutine(x_deltax_deltalogq, batchsize, dim)}\n")
            f.write(f"└── corrs error: {get_errlogqx_from_in_corr_subroutine(x_deltax_deltalogq, batchsize, dim)}\n")

        return np.concatenate([drift_flatten, 
                               v_deltax_flatten, 
                               corr_flatten,
                               err_nabla_logq_flatten,
                               err_lap_logq_flatten], axis=0)

    init = jnp.concatenate([x0.reshape(-1), np.zeros((batchsize*dim)), np.zeros((4*batchsize))], axis=0) # x, deltax, delta logq, err(deltax), err(delta logq)
    try:
        solution = integrate.solve_ivp(func, (sde.Tmin, sde.Tmax), init, rtol=rtol_dir["outer"], atol=atol_dir["outer"], method="RK45")
    except OverflowError:
        return jnp.array([]), jnp.array([]), jnp.array([])          

    #x_traj = solution.y[:batchsize*dim].reshape(batchsize, dim, -1)
    x_deltax_deltalogq_T = solution.y[:, -1]
    xT = get_x_from_in_corr_subroutine(x_deltax_deltalogq_T, batchsize, dim)
    deltaxT = get_deltax_from_in_corr_subroutine(x_deltax_deltalogq_T, batchsize, dim)
    corrT = get_logq_from_in_corr_subroutine(x_deltax_deltalogq_T, batchsize, dim)
    corrT += -jnp.sum(deltaxT*xT, axis=1)

    # final error
    err_deltaxT = get_errdeltax_from_in_corr_subroutine(x_deltax_deltalogq_T, batchsize, dim)
    err_corrT  = jnp.abs(jnp.sum(err_deltaxT*xT, axis=1))
    err_corrT += get_errlogqx_from_in_corr_subroutine(x_deltax_deltalogq_T, batchsize, dim)

    return -logq0, -corrT, err_corrT

def get_nll_with_1st_correction_by_solve_ivp_outer_integral_with_model_err_estimate(x0, state, sde, dx, rtol_dir, atol_dir, save_path):
    logq0, _, _, success = get_logq(x0, sde.Tmin, state, sde, rtol_dir["inner"], atol=atol_dir["inner"], method="RK45")
    if not success:
        return jnp.array([]), jnp.array([]), jnp.array([])
    
    batchsize, dim = x0.shape
    def func(t, x_deltax_deltalogq):
        #print(f"outer t:{t}")
        with open(save_path, 'a') as f:
            f.write(f"outer t:{t}\n")

        # 0th-logqSolver part
        xt = get_x_from_in_corr_subroutine(x_deltax_deltalogq, batchsize, dim)

        # logq for main calc
        rtol, atol = rtol_dir["inner"], atol_dir["inner"]
        logq, t_traj, x_traj, success = get_logq(xt, t, state, sde, rtol=rtol, atol=atol, method="RK45")
        if not success:
            raise OverflowError()
        logq_xpos, _, _, success = get_logq(xt + dx*np.tile(np.array([1,0]), (xt.shape[0], 1)), t, state, sde, rtol=rtol, atol=atol, method="RK45")
        if not success:
            raise OverflowError()
        logq_xneg, _, _, success = get_logq(xt - dx*np.tile(np.array([1,0]), (xt.shape[0], 1)), t, state, sde, rtol=rtol, atol=atol, method="RK45")
        if not success:
            raise OverflowError()
        logq_ypos, _, _, success = get_logq(xt + dx*np.tile(np.array([0,1]), (xt.shape[0], 1)), t, state, sde, rtol=rtol, atol=atol, method="RK45")
        if not success:
            raise OverflowError()
        logq_yneg, _, _, success = get_logq(xt - dx*np.tile(np.array([0,1]), (xt.shape[0], 1)), t, state, sde, rtol=rtol, atol=atol, method="RK45")
        if not success:
            raise OverflowError()
        
        # Calc f^{PF}_\theta
        drift = get_drift_for_probability_flow(xt, t, state, sde)
        drift_flatten = np.array(drift).reshape(-1)
        
        # Calc \delta f^{PF}_\theta
        deltax = get_deltax_from_in_corr_subroutine(x_deltax_deltalogq, batchsize, dim)
        nabla_logq = get_nabla_logq_from(logq_xpos, logq_xneg, logq_ypos, logq_yneg, dx)
        ####
        f = lambda y: get_drift_for_probability_flow(y, t, state, sde)
        div_f = lambda x: jnp.expand_dims(get_div(x, f), axis=1)       # used below
        v_deltax1 = directional_differential(xt, f, deltax)
        v_deltax2 = -sde.g(t)**2/2*(state.s(xt, t) - nabla_logq)
        v_deltax = v_deltax1 + v_deltax2
        v_deltax_flatten = v_deltax.reshape(-1)

        # Calc \nabla \cdot \delta {\bm f}^{\rm PF}_{\theta}
        lap_log_q = get_laplacian_logq_from(logq, logq_xpos, logq_xneg, logq_ypos, logq_yneg, dx)
        ####
        div_s = get_div_s(xt, t, state, sde)
        corr1 = directional_differential(xt, div_f, deltax).reshape(-1)
        corr2 = -sde.g(t)**2/2*(div_s - lap_log_q)
        corr = corr1 + corr2
        corr_flatten = corr.reshape(-1)

        # Local error estimation
        err_deltax = get_errdeltax_from_in_corr_subroutine(x_deltax_deltalogq, batchsize, dim)
        
        ## estimate of dx^2 coefficients
        ds = lambda x: jnp.expand_dims(get_div_s(x, t, state=state, sde=sde), axis=1)
        dds = lambda x: jnp.squeeze(jnp.transpose(jacobian(x, ds), axes=[2, 1, 0]))
        ddds = lambda x: get_div(x, dds)

        C1 = dds(xt)
        C2 = ddds(xt).reshape(-1, 1)

        ## calc error for log q
        logq_for_error, t_traj, x_traj, success = get_logq(xt, t, state, sde, rtol=1.1*rtol, atol=1.1*atol, method="RK45")
        if not success:
            raise OverflowError()
        logq_error = jnp.abs(logq - logq_for_error).reshape(-1, 1)   # measuring precision order of the 0th solver

        err_nabla_logq = sde.g(t)**2/2*get_err_nabla_logq(tol=logq_error, dx=dx, C=C1)
        err_nabla_logq += jnp.abs(directional_differential(xt, f, err_deltax))
        
        err_lap_logq = sde.g(t)**2/2*get_err_lap_logq(tol=logq_error, dx=dx, C=C2)
        err_lap_logq += jnp.abs(directional_differential(xt, div_f, err_deltax))

        err_nabla_logq_flatten = err_nabla_logq.reshape(-1)
        err_lap_logq_flatten = err_lap_logq.reshape(-1)

        with open(save_path, 'a') as f:
            f.write(f"x = {xt}\n")
            f.write(f"delta x = {deltax}\n")
            f.write(f"└── delta x error: {get_errdeltax_from_in_corr_subroutine(x_deltax_deltalogq, batchsize, dim)}\n")
            f.write(f"corrs = {get_logq_from_in_corr_subroutine(x_deltax_deltalogq, batchsize, dim)}\n")
            f.write(f"└── corrs error: {get_errlogqx_from_in_corr_subroutine(x_deltax_deltalogq, batchsize, dim)}\n")

        return np.concatenate([drift_flatten, 
                               v_deltax_flatten, 
                               corr_flatten,
                               err_nabla_logq_flatten,
                               err_lap_logq_flatten], axis=0)

    init = jnp.concatenate([x0.reshape(-1), np.zeros((batchsize*dim)), np.zeros((4*batchsize))], axis=0) # x, deltax, delta logq, err(deltax), err(delta logq)
    try:
        solution = integrate.solve_ivp(func, (sde.Tmin, sde.Tmax), init, rtol=rtol_dir["outer"], atol=atol_dir["outer"], method="RK45")
    except OverflowError:
        return jnp.array([]), jnp.array([]), jnp.array([])          

    x_deltax_deltalogq_T = solution.y[:, -1]
    xT = get_x_from_in_corr_subroutine(x_deltax_deltalogq_T, batchsize, dim)
    deltaxT = get_deltax_from_in_corr_subroutine(x_deltax_deltalogq_T, batchsize, dim)
    corrT = get_logq_from_in_corr_subroutine(x_deltax_deltalogq_T, batchsize, dim)
    corrT += -jnp.sum(deltaxT*xT, axis=1)

    # final error
    err_deltaxT = get_errdeltax_from_in_corr_subroutine(x_deltax_deltalogq_T, batchsize, dim)
    err_corrT  = jnp.abs(jnp.sum(err_deltaxT*xT, axis=1))
    err_corrT += get_errlogqx_from_in_corr_subroutine(x_deltax_deltalogq_T, batchsize, dim)

    return -logq0, -corrT, err_corrT