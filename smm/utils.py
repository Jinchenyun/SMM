import numpy as np


def Linex(u, a):
    return np.exp(u*a) - u*a - 1


def Blinex(u, miu, lambda_, a):
    return (1 - 1/(1+lambda_*Linex(u, a))) / miu


def Aen(u, tau, theta):
    return np.where(u >= 0, theta/2 * (u**2) + (1 - theta) * u, tau * (theta/2 * (u**2) - (1 - theta) * u))

def Baen(u, eta, lambda_, tau, theta):
    return (1 - 1/(1+lambda_*Aen(u, tau, theta))) / eta


def Pin(u, tau):
    return np.where(u >= 0, u, -tau * u)


def g(u, lambda_, tau):
    return lambda_ * Pin(u, tau)


def h_baen(u, eta, lambda_, tau, theta):
    return -1 * g(u, lambda_, tau) + Baen(u, eta, lambda_, tau, theta)


def h_baen_diff(u, eta, lambda_, tau, theta):
    u_sq = u ** 2
    abs_u = np.abs(u)
    denom_base = 1 + lambda_ * (theta / 2 * u_sq + (1 - theta) * abs_u)
    denom = eta * denom_base ** 2

    # 计算分子
    numerator_pos = theta * (u - 1) + 1
    numerator_neg = theta * (u + 1) - 1

    # 使用 np.where 替代 if-else
    out = np.where(
        u >= 0,
        lambda_ * numerator_pos / denom - lambda_,
        lambda_ * tau * numerator_neg / denom + lambda_ * tau
    )

    return out


def Bsp(u, eta, lambda_, tau):
    return np.where(u >= 0, 1 / eta * (1 - 1 / (1 + lambda_ * u**2)), 1 / eta * (1 - 1 / (1 + lambda_ * tau * u**2)))


def h_bsp(u, eta, lambda_, tau):
    return -1 * g(u, lambda_, tau) + Bsp(u, eta, lambda_, tau)


def h_bsp_diff(u, eta, lambda_, tau):
    return np.where(u >= 0, lambda_ * (2 * u / (1 + lambda_ * u**2)**2 - 1), lambda_ * tau * (2 * u / (1 + lambda_ * u**2)**2 + 1))
