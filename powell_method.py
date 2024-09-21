import numpy as np
from scipy_functions import _line_for_search, _minimize_scalar_bounded

def line_search(function, x, direction, bounds, n, tol=1e-4):
    def f1d(alpha):
        return function(x + alpha * direction)

    bound = _line_for_search(x, direction, bounds[0], bounds[1])
    res = _minimize_scalar_bounded(f1d, bound, xatol=tol)
    alpha = res.x
    new_dir = res.x * direction
    return x + alpha * direction, new_dir, res.fun

def powell(function, x0, bounds, max_fun_evals, get_function_evals, tol=1e-4, max_iter=100, ftol=1e-4):
    n = len(x0)
    a, b = bounds
    lower_bound_array = np.full(n,a)
    upper_bound_array = np.full(n,b)
    bounds_array = np.array([lower_bound_array, upper_bound_array])
    directions = np.eye(n)
    x = x0.copy()
    f0 = function(x)
    f_ret = f0
    for _ in range(max_iter):
        x_old = x.copy()
        f_old = f_ret
        delta = 0.0
        biggest_decrease_index = 0
        for i in range(n):
            f_aux = f_ret
            direction = directions[i]
            x, _, f_ret = line_search(function, x, direction, bounds_array, n, tol)
            decrease = f_aux - f_ret
            if decrease > delta:
                delta = decrease
                biggest_decrease_index = i

        bnd = ftol * (np.abs(f_old) + np.abs(f_ret)) + 1e-20
        if 2.0 * (f_old - f_ret) <= bnd:
            break
        function_evals = get_function_evals()
        if max_fun_evals is not None and function_evals >= max_fun_evals:
            break
        if np.isnan(f_old) and np.isnan(f_ret):
            break

        new_direction = x - x_old
        _,lmax = _line_for_search(x, new_direction, lower_bound_array, upper_bound_array)
        x_extrapolated = x + min(lmax,1) * new_direction
        f_ext = function(x_extrapolated)

        if(f_ext < f_old):
          t = 2.0 * (f_old - 2.0 * f_ret + f_ext) * pow(f_old - f_ret - delta, 2) - delta * pow(f_old - f_ext, 2)
          if(t < 0.0):
            x, _, f_ret = line_search(function, x, new_direction, bounds_array, n, tol)
            directions[biggest_decrease_index] = directions[-1]
            directions[-1] = new_direction


    return x