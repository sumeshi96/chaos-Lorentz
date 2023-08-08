# ローレンツ方程式のリアプノフ指数を計算する

import numpy as np

# ローレンツ方程式
def lorenz_equations(x, y, z, sigma, rho, beta):
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return dx_dt, dy_dt, dz_dt

# ローレンツ方程式のヤコビアン
def lorenz_jacobian(x, y, z, sigma, rho, beta):
    J = np.zeros((3, 3))
    J[0, 0] = -sigma
    J[0, 1] = sigma
    J[1, 0] = rho - z
    J[1, 1] = -1
    J[1, 2] = -x
    J[2, 0] = y
    J[2, 1] = x
    J[2, 2] = -beta
    return J

# リアプノフ指数の計算
def lyapunov_exponent(x0, y0, z0, sigma, rho, beta, n, dt):
    x, y, z = x0, y0, z0
    L = np.zeros(3)

    for _ in range(n):
        dx_dt, dy_dt, dz_dt = lorenz_equations(x, y, z, sigma, rho, beta)
        J = lorenz_jacobian(x, y, z, sigma, rho, beta)
        L += np.log(np.abs(np.linalg.eigvals(J)))
        x += dx_dt * dt
        y += dy_dt * dt
        z += dz_dt * dt

    return L / n

# パラメータの設定
sigma = 10
rho = 28
beta = 8/3
x0, y0, z0 = 1, 1, 1
n = 10000
dt = 0.01

# リアプノフ指数の計算
lyapunov_exp = lyapunov_exponent(x0, y0, z0, sigma, rho, beta, n, dt)
print("Lyapunov Exponents:", lyapunov_exp)
