import os
import numpy as np
import matplotlib.pyplot as plt
from simulate import simulate_truth, omega_true, simulate_sensors
from ekf import predict, run_filter, attitude_errors

os.makedirs ("results", exist_ok=True)

ts, qs, ws = simulate_truth(omega_true)
dt = 0.01
gyro, accel, bias_true = simulate_sensors(qs, ws, dt)
x0 = np.array([1.0, 0, 0, 0,  0, 0, 0])
P0 = np.diag([0.1**2]*3 + [0.02**2]*3)
xs, Ps = run_filter(gyro, accel, dt, x0, P0)


xg = np.zeros_like(xs)
xg[0] = x0
for k in range(len(ts) - 1):
    xg[k + 1] = predict(xg[k], gyro[k], dt)

tilt_f, yaw_f = attitude_errors(xs[:, :4], qs)
tilt_g, yaw_g = attitude_errors(xg[:, :4], qs)


fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ts, np.hypot(tilt_g, yaw_g), color="gray", label="gyro only (total)")
ax.plot(ts, yaw_f, color="tab:orange", label="EKF yaw error")
ax.plot(ts, tilt_f, color="tab:blue", label="EKF tilt error")
ax.set_xlabel("time [s]")
ax.set_ylabel("error [deg]")
ax.set_title("Attitude error: gyro only vs EKF")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/fig1_attitude_error.png", dpi=150)



sig_b = np.sqrt(np.array([np.diag(P)[3:] for P in Ps]))
fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
for i, name in enumerate(["x", "y", "z"]):
    axs[i].plot(ts, bias_true[:, i], color="tab:green", label="true bias")
    axs[i].plot(ts, xs[:, 4 + i], color="tab:blue", label="EKF guess")
    axs[i].fill_between(ts, xs[:, 4 + i] - 2 * sig_b[:, i],
                        xs[:, 4 + i] + 2 * sig_b[:, i],
                        color="tab:blue", alpha=0.15, label="±2σ")
    axs[i].set_ylabel(f"b_{name} [rad/s]")
    axs[i].grid(alpha=0.3)
axs[0].legend(loc="upper right")
axs[2].set_xlabel("time [s]")
axs[0].set_title("Gyro bias: EKF guess vs truth")
fig.tight_layout()
fig.savefig("results/fig2_bias.png", dpi=150)


sigmas = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
tilt_rms = []
for s in sigmas:
    xs_s, _ = run_filter(gyro, accel, dt, x0, P0, sigma_accel=s)
    tilt_s, _ = attitude_errors(xs_s[:, :4], qs)
    tilt_rms.append(np.sqrt(np.mean(tilt_s[500:] ** 2)))
    print(f"sigma_accel = {s:6.3f}   tilt RMS = {tilt_rms[-1]:7.3f} deg")

fig, ax = plt.subplots(figsize=(7, 4))
ax.loglog(sigmas, tilt_rms, "o-")
ax.axvline(0.02, color="tab:green", ls="--", label="true sensor noise")
ax.set_xlabel("sigma_accel used in the filter (R = sigma²)")
ax.set_ylabel("tilt RMS error, t > 5 s [deg]")
ax.set_title("Tuning R")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("results/fig3_tuning_R.png", dpi=150)

print("saved 3 figures in results/")