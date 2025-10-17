from mj_etc import ETC
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import colormaps
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.use("tkagg")

# etc = ETC("B", "FLI", "B&C", moon_phase='bright')
# etc.time_for_snr(30, mag=16.3, plot=True)
# plt.show()
AU: float = 1.4959787 * 10 ** 11
rows_to_skip = 0
data = np.loadtxt("server_data_error.txt", delimiter=",")
true = np.array([1.356419039495192, 6.141111760545866, 1.751132828366537E+02 * np.pi / 180,
                 3.221602897089529E+02 * np.pi / 180, 1.280093129447824E+02 * np.pi / 180,  2.460977979242884E+06])
means = np.array([np.mean(data[rows_to_skip:, 0]) / (1.5 * 10 ** 11), np.mean(data[rows_to_skip:, 1]),
                  np.mean(data[rows_to_skip:, 2]), np.mean(data[rows_to_skip:, 3]), np.mean(data[rows_to_skip:, 4]),
                  np.mean(data[rows_to_skip:, 5])])
stds = np.array([np.std(data[rows_to_skip:, 0]) / (1.5 * 10 ** 11), np.std(data[rows_to_skip:, 1]),
                 np.std(data[rows_to_skip:, 2]), np.std(data[rows_to_skip:, 3]), np.std(data[rows_to_skip:, 4]),
                 np.std(data[rows_to_skip:, 5])])

print((true - means) / stds)
print(stds / means)

data = np.loadtxt("server_data_error.txt", delimiter=",")
print(np.mean(data[rows_to_skip:, 0]) / (1.5 * 10 ** 11), np.mean(data[rows_to_skip:, 1]),
      np.mean(data[rows_to_skip:, 2]), np.mean(data[rows_to_skip:, 3]), np.mean(data[rows_to_skip:, 4]),
      np.mean(data[rows_to_skip:, 5]))
print(np.std(data[rows_to_skip:, 0]) / (1.5 * 10 ** 11), np.std(data[rows_to_skip:, 1]), np.std(data[rows_to_skip:, 2]),
      np.std(data[rows_to_skip:, 3]), np.std(data[rows_to_skip:, 4]), np.std(data[rows_to_skip:, 5]))

num_bins = 40
line_height = 800
fig, axs = plt.subplots(2, 3)
axs[0, 0].hist(data[rows_to_skip:, 0], num_bins)
# axs[0, 0].vlines(true[0] * 1.5 * 10 ** 11, 0, line_height, colors="r", linestyles="dashed")
axs[0, 1].hist(scipy.stats.sigmaclip(data[rows_to_skip:, 1], low=6, high=6), num_bins)
# axs[0, 1].vlines(true[1], 0, line_height, colors="r", linestyles="dashed")
axs[0, 2].hist(data[rows_to_skip:, 2], num_bins)
axs[0, 2].vlines(true[2], 0, line_height, colors="r", linestyles="dashed")
axs[1, 0].hist(data[rows_to_skip:, 3], num_bins)
axs[1, 0].vlines(true[3], 0, line_height, colors="r", linestyles="dashed")
axs[1, 1].hist(data[rows_to_skip:, 4], num_bins)
axs[1, 1].vlines(true[4], 0, line_height, colors="r", linestyles="dashed")
axs[1, 2].hist(data[rows_to_skip:, 5], num_bins)
axs[1, 2].vlines(true[5], 0, line_height, colors="r", linestyles="dashed")

data = np.loadtxt("S:/obs_error_params.txt", delimiter=",")
# print(np.std(data[:, 6]) / (1.5 * 10 ** 11), np.std(data[:, 7]), np.std(data[:, 8]), np.std(data[:, 9]), np.std(data[:, 10]),
#       np.std(data[:, 11]))
#

print(data[np.where(data[:, 12] == np.min(data[:, 12])), :])

scaled_errors = np.log10(data[:, 12][data[:, 12] < 10])
# scaled_errors = np.log10(data[:, 12])
data = data[data[:, 12] < 10]
min_error = np.min(scaled_errors)
max_error = np.max(scaled_errors)
normalised_error = (scaled_errors - min_error) / (max_error - min_error)
color_map = colormaps['gnuplot_r']

fig_2, axs_2 = plt.subplots(2, 3)
axs_2[0, 0].scatter(data[:, 0] * AU, data[:, 1], c="b", alpha=0.01)
axs_2[0, 1].scatter(data[:, 0] * AU, data[:, 2], c="b", alpha=0.01)
axs_2[0, 2].scatter(data[:, 0] * AU, data[:, 3], c="b", alpha=0.01)
axs_2[1, 0].scatter(data[:, 0] * AU, data[:, 4], c="b", alpha=0.01)
axs_2[1, 1].scatter(data[:, 0] * AU, data[:, 5] * 7 + 2460980, c="b", alpha=0.01)
axs_2[1, 2].scatter(data[:, 1], data[:, 2], c="b", alpha=0.01)
axs_2[0, 0].scatter(data[:, 6], data[:, 7], alpha=1, c=normalised_error, cmap=color_map)
axs_2[0, 1].scatter(data[:, 6], data[:, 8], alpha=1, c=normalised_error, cmap=color_map)
axs_2[0, 2].scatter(data[:, 6], data[:, 9], alpha=1, c=normalised_error, cmap=color_map)
axs_2[1, 0].scatter(data[:, 6], data[:, 10], alpha=1, c=normalised_error, cmap=color_map)
axs_2[1, 1].scatter(data[:, 6], data[:, 11], alpha=1, c=normalised_error, cmap=color_map)
axs_2[1, 2].scatter(data[:, 7], data[:, 8], alpha=1, c=normalised_error, cmap=color_map)

fig_4, axs_4 = plt.subplots(1, 3)
im1 = axs_4[0].scatter(data[:, 6], data[:, 7], alpha=1, c=scaled_errors, cmap=color_map)
axs_4[0].set_xlabel("Perihelion distance [m]")
axs_4[1].set_xlabel("Perihelion distance [m]")
axs_4[2].set_xlabel("Perihelion distance [m]")
axs_4[0].set_ylabel("Eccentricity")
axs_4[1].set_ylabel("Inclination [rad]")
axs_4[2].set_ylabel("Time of perihelion [JD]")
axs_4[1].scatter(data[:, 6], data[:, 8], alpha=1, c=scaled_errors, cmap=color_map)
axs_4[2].scatter(data[:, 6], data[:, 11], alpha=1, c=scaled_errors, cmap=color_map)
cax = make_axes_locatable(axs_4[2]).append_axes("right", size="5%", pad=0.05)
plt.colorbar(im1, cax=cax, label="log10 of the error in the fit")
fig_4.set_size_inches(12, 5)
fig_4.tight_layout(pad=0.5)
print(data[data[:, 12] == np.min(data[:, 12])])


params = ["QR [m]", "E", "I [rad]", "$\\Omega$ [rad]", "$\\omega$ [rad]", "TP"]
fig_3, axs_3 = plt.subplots(1, 6, figsize=(10, 4))
for param in range(6):
    axs_3[param].scatter(data[:, param + 6], -1 * np.log10(data[:, 12]))
    axs_3[param].set_xlabel(params[param])
    axs_3[param].set_ylim(-2, 2)
    if param != 0:
        axs_3[param].yaxis.set_ticks([])

axs_3[0].set_ylabel("-log10 of the error in the fit")
fig_3.tight_layout()

# num_bins = 2000
# for parameter in range(6):
#     param_values = data[:, parameter + 6]
#     original_max = np.max(param_values)
#     original_min = np.min(param_values)
#     min_error_in_each = {}
#     for fit in range(param_values.size):
#         param_value = ((param_values[fit] - original_min) / ((original_max - original_min) / num_bins) // 1)
#         try:
#             if min_error_in_each[param_value] > data[fit, -1]:
#                 min_error_in_each[param_value] = data[fit, -1]
#         except KeyError:
#             min_error_in_each[param_value] = data[fit, -1]
#
#     ydata = np.array([-1 * float(value) + max(data[:, -1]) for value in min_error_in_each.values()])
#     xdata = np.array([float(value) for value in min_error_in_each.keys()])
#     parameters, _ = scipy.optimize.curve_fit(gauss, xdata * ((original_max - original_min) / num_bins) + original_min,
#                                              ydata, (0, 10, np.mean(param_values), np.std(param_values)))
#
#     axs_3[parameter].scatter(xdata * ((original_max - original_min) / num_bins) + original_min, ydata)
#     axs_3[parameter].scatter(np.linspace(original_min, original_max, 10000), gauss(np.linspace(original_min,
#                                                                                                original_max, 10000),
#                                                                                    *parameters))
#     print(f"Parameter {parameter}: {parameters}")

plt.show()


