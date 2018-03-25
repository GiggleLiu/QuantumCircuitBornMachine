import numpy as np

def train(bm, theta_list, method, max_iter=1000, popsize=50, step_rate=0.1):
    '''train a Born Machine.'''
    theta_list = np.array(theta_list)
    if method == 'DE':
        from scipy.optimize import differential_evolution
        res = differential_evolution(bm.mmd_loss,
                                     maxiter=max_iter, popsize=popsize,
                                     options={'disp': 2, 'iprint': 2})
    elif method == 'Adam':
        from climin import Adam
        optimizer = Adam(wrt=theta_list, fprime=bm.gradient,step_rate=step_rate)
        for info in optimizer:
            step = info['n_iter']
            loss = bm.mmd_loss(theta_list)
            print('step = %d, loss = %s'%(step, loss))
            if step == max_iter:
                break
        return bm.mmd_loss(theta_list), theta_list
    else:
        from scipy.optimize import minimize
        res = minimize(bm.mmd_loss, x0=theta_list,
                       method=method, jac = bm.gradient, tol=1e-12,
                       options={'maxiter': max_iter, 'disp': 2, 'gtol':1e-10, 'ftol':0},
                       )
        return res.fun, res.x
