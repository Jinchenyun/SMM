def Linex(u, a):
    return np.exp(u*a) - u*a - 1


def Blinex(u, miu, lambda_, a):
    return (1 - 1/(1+lambda_*Linex(u, a))) / miu


def Aen(u, tau, theta):
    if u >= 0:
        return theta/2 * (u**2) + (1 - theta) * u
    else:
        return tau * (theta/2 * (u**2) - (1 - theta) * u)


def Baen(u, eta, lambda_, tau, theta):
    return (1 - 1/(1+lambda_*Aen(u, tau, theta))) / eta


def Pin(u, tau):
    if u >= 0:
        return u
    else:
        return -1 * tau * u


def g(u, lambda_, tau):
    return lambda_ * Pin(u, tau)


def h_baen(u, eta, lambda_, tau, theta):
    return -1 * g(u, lambda_, tau) + Baen(u, eta, lambda_, tau, theta)


def h_baen_diff(u, eta, lambda_, tau, theta):
    if u >= 0:
        out = lambda_ * (theta*(u - 1) + 1) / (eta * (1 + (lambda_*(theta/2 * u**2 + (1-theta)*u)))**2) - lambda_
    else:
        out = lambda_*tau * (theta*(u + 1) - 1) / (eta * (1 + (lambda_*(theta/2 * u**2 - (1-theta)*u)))**2) + lambda_*tau
    return out


def Bsp(u, eta, lambda_, tau):
    if u >= 0:
        out = 1 / eta * (1 - 1 / (1 + lambda_ * u**2))
    else:
        out = 1/ eta * (1 - 1 / (1 + lambda_ * tau * u**2))
    return  out

def h_bsp(u, eta, lambda_, tau):
    return -1 * g(u, lambda_, tau) + Bsp(u, eta, lambda_, tau)


def h_bsp_diff(u, eta, lambda_, tau):
    if u >= 0:
        out = lambda_ * (2 * u / (1 + lambda_ * u**2)**2 - 1)
    else:
        out = lambda_ * tau * (2 * u / (1 + lambda_ * u**2)**2 + 1)
    return out
