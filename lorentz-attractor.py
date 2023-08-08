import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

def lorentz_equation(X, dt=0.01, rho=28.0, sigma=10.0, beta=8.0/3.0):
    x, y, z = X
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z

    x += dx * dt
    y += dy * dt
    z += dz * dt

    return np.array([x, y, z])

def lorentz_equation_simulate(initial_conditions, N, dt=0.01):
    trajectory = np.empty((N, 3))
    trajectory[0] = initial_conditions

    for i in range(N-1):
        trajectory[i+1] = lorentz_equation(trajectory[i], dt)

    return trajectory

# ローレンツ方程式の数値解を取得
N = 10000
dt = 0.01
initial_conditions = np.array([1.0, 1.0, 1.0])
trajectory = lorentz_equation_simulate(initial_conditions, N, dt)

# 各グラフの表示
fig1 = plt.figure(figsize=(5, 5))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(trajectory[:,0], trajectory[:,1], trajectory[:,2],linewidth=0.5)
ax1.set_title('ローレンツ方程式')
plt.tight_layout()
plt.show()

fig2 = plt.figure(figsize=(10, 5))
ax2 = fig2.add_subplot(111)
ax2.plot(np.arange(N)*dt, trajectory[:, 0],linewidth=0.5)
ax2.set_xlabel('t')
ax2.set_ylabel('x(t)')
ax2.set_title('x(t)の観測時系列')
plt.tight_layout()
plt.show()

# x(t)、x(t+τ)、x(t+2τ)を使った3Dグラフの作成
taus = [1, 5, 10]
for tau in taus:
    x_t = trajectory[:, 0]
    x_t_tau = np.roll(x_t, -tau)
    x_t_2tau = np.roll(x_t, -2*tau)

    fig3 = plt.figure(figsize=(5, 5))
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.plot(x_t[:-2*tau], x_t_tau[:-2*tau], x_t_2tau[:-2*tau],linewidth=0.5)
    ax3.set_xlabel('x(t)')
    ax3.set_ylabel('x(t+τ)')
    ax3.set_zlabel('x(t+2τ)')
    ax3.set_title(f'時間遅れτ={tau/100}におけるアトラクター再構成')
    plt.tight_layout()
    plt.show()