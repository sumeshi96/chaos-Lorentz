# ローレンツ方程式の解を求める

import numpy as np

# ローレンツ方程式
def lorenz_equations(x, y, z, sigma, rho, beta):
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return dx_dt, dy_dt, dz_dt

# ルンゲ・クッタ法
def runge_kutta_step(x, y, z, sigma, rho, beta, dt):
    k1_x, k1_y, k1_z = lorenz_equations(x, y, z, sigma, rho, beta)
    k2_x, k2_y, k2_z = lorenz_equations(x + 0.5 * dt * k1_x, y + 0.5 * dt * k1_y, z + 0.5 * dt * k1_z, sigma, rho, beta)
    k3_x, k3_y, k3_z = lorenz_equations(x + 0.5 * dt * k2_x, y + 0.5 * dt * k2_y, z + 0.5 * dt * k2_z, sigma, rho, beta)
    k4_x, k4_y, k4_z = lorenz_equations(x + dt * k3_x, y + dt * k3_y, z + dt * k3_z, sigma, rho, beta)

    x_next = x + (dt / 6) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
    y_next = y + (dt / 6) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
    z_next = z + (dt / 6) * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)

    return x_next, y_next, z_next

# ローレンツ方程式のシミュレーション
def lorenz_simulation(sigma, rho, beta, initial_state, timesteps, dt):
    x, y, z = initial_state
    x_data, y_data, z_data = [x], [y], [z]

    for _ in range(timesteps):
        x, y, z = runge_kutta_step(x, y, z, sigma, rho, beta, dt)
        x_data.append(x)
        y_data.append(y)
        z_data.append(z)

    return x_data, y_data, z_data

# パラメータの設定
sigma = 10
rho = 28
beta = 8/3
initial_state = (1, 1, 1)
timesteps = 10000
dt = 0.01

# ローレンツ方程式のシミュレーション
x_data, y_data, z_data = lorenz_simulation(sigma, rho, beta, initial_state, timesteps, dt)

# 結果の表示
print("x:", x_data[-1])
print("y:", y_data[-1])
print("z:", z_data[-1])
