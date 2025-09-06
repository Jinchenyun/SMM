import numpy as np


def Linex(u, a):
    return np.exp(u * a) - u * a - 1


def Blinex(u, miu, lambda_, a):
    return (1 - 1 / (1 + lambda_ * Linex(u, a))) / miu


def Aen(u, tau, theta):
    out = np.where(
        u >= 0,
        theta * (u ** 2) + (1 - theta) * u,
        tau * (theta * (u ** 2) - (1 - theta) * u)
    )
    return out


def Baen(u, eta, lambda_, tau, theta):
    L = 1 / eta * (1 - 1 / (1 + lambda_ * Aen(u, tau, theta)))
    return L


def Pin(u, tau):
    return np.where(u >= 0, u, -1 * tau * u)


def is_fisher_consistency(lambda_, theta):
    """
    Determine whether the given parameter's Bean loss function is Fisher-consistent

    :param lambda_:
    :param theta:
    :return: 0(not Fisher-consistent) or 1(Fisher-consistent)
    """

    if lambda_ * (1 + 3 * theta) < (1 - theta) * (1 + 2 * lambda_ * (theta + 1))**2:
        return 1
    else:
        return 0


def clip_dcd_optimizer(H, q, lb, ub, eps, max_steps, u0):
    """带边界约束的坐标下降优化器"""
    n = H.shape[0]
    u = u0.copy().reshape(-1, 1)
    lb = lb.reshape(-1, 1)
    ub = ub.reshape(-1, 1)

    diagH = np.diag(H).reshape(-1, 1)
    Hu = H @ u
    ub_u = (ub - u).reshape(-1, 1)
    lb_u = (lb - u).reshape(-1, 1)

    for step in range(max_steps):
        numerator = q.reshape(-1, 1) - Hu
        L_idx_val = numerator / diagH
        L_val = numerator * L_idx_val

        # 找到满足更新条件的索引
        cond1 = (u > lb.reshape(-1, 1)) & (L_idx_val < 0)
        cond2 = (u < ub.reshape(-1, 1)) & (L_idx_val > 0)
        valid_idx = np.where(cond1 | cond2)[0]

        if len(valid_idx) == 0:
            break

        # 在有效索引中找到最大L_val
        max_L_val = np.max(L_val[valid_idx])
        if max_L_val < eps:
            break

        # 找到所有具有最大L_val的索引
        max_indices = np.where(L_val == max_L_val)[0]

        # 在候选索引中找到最佳更新
        best_idx = None
        best_lambda = 0
        for idx in max_indices:
            lambda_opt_temp = np.clip(L_idx_val[idx], lb_u[idx], ub_u[idx])

            if abs(lambda_opt_temp) > abs(best_lambda):
                best_lambda = lambda_opt_temp
                best_idx = idx

        if best_idx is None:
            continue

        # 更新选定的坐标
        u_old = u[best_idx].copy()
        u_new = u_old + best_lambda
        u[best_idx] = u_new

        delta = u_new - u_old
        Hu += H[:, best_idx].reshape(-1, 1) * delta

        # 更新已更新坐标的边界
        ub_u[best_idx] = ub[best_idx] - u_new
        lb_u[best_idx] = lb[best_idx] - u_new

    # 计算目标值
    obj_val = 0.5 * u.T @ H @ u - q.T @ u

    return {
        'x': u,
        'iterations': step + 1,
        'objective_value': obj_val.item()
    }