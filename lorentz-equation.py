# ローレンツ方程式の図をプロット

import numpy as np
import matplotlib.pyplot as plt

# ローレンツ方程式
def lorenz_equations(x, y, z, sigma, rho, beta):
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return dx_dt, dy_dt, dz_dt

# ローレンツ方程式のシミュレーション
def lorenz_simulation(sigma, rho, beta, initial_state, timesteps, dt):
    x, y, z = initial_state
    x_data, y_data, z_data = [x], [y], [z]

    for _ in range(timesteps):
        dx_dt, dy_dt, dz_dt = lorenz_equations(x, y, z, sigma, rho, beta)
        x += dx_dt * dt
        y += dy_dt * dt
        z += dz_dt * dt
        x_data.append(x)
        y_data.append(y)
        z_data.append(z)

    return x_data, y_data, z_data

# パラメータの設定
sigma = 10
rho = 28
beta = 8/3
initial_state = (1, 1, 1)
timesteps = 4000
dt = 0.01

# ローレンツ方程式のシミュレーション
x_data, y_data, z_data = lorenz_simulation(sigma, rho, beta, initial_state, timesteps, dt)

# プロット
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_data, y_data, z_data)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor')
plt.show()
