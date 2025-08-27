from mj_etc import ETC
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import colormaps
matplotlib.use("tkagg")

# etc = ETC("B", "FLI", "B&C", moon_phase='bright')
# etc.time_for_snr(30, mag=16.3, plot=True)
# plt.show()

true = [1.356393053772311, 6.141111760545866, 1.751132828366537E+02 * np.pi / 180,
        3.221602897089529E+02 * np.pi / 180, 1.280093129447824E+02 * np.pi / 180,  2.460977979242884E+06]

rows_to_skip = -20

data = np.loadtxt("error orbit fits.txt", delimiter=",")
print(np.mean(data[rows_to_skip:, 0]) / (1.5 * 10 ** 11), np.mean(data[rows_to_skip:, 1]),
      np.mean(data[rows_to_skip:, 2]), np.mean(data[rows_to_skip:, 3]), np.mean(data[rows_to_skip:, 4]),
      np.mean(data[rows_to_skip:, 5]))
print(np.std(data[rows_to_skip:, 0]) / (1.5 * 10 ** 11), np.std(data[rows_to_skip:, 1]), np.std(data[rows_to_skip:, 2]),
      np.std(data[rows_to_skip:, 3]), np.std(data[rows_to_skip:, 4]), np.std(data[rows_to_skip:, 5]))

num_bins = 40
line_height = 2
fig, axs = plt.subplots(2, 3)
axs[0, 0].hist(data[rows_to_skip:, 0], num_bins)
axs[0, 0].vlines(true[0] * 1.5 * 10 ** 11, 0, line_height, colors="r", linestyles="dashed")
axs[0, 1].hist(data[rows_to_skip:, 1], num_bins)
axs[0, 1].vlines(true[1], 0, line_height, colors="r", linestyles="dashed")
axs[0, 2].hist(data[rows_to_skip:, 2], num_bins)
axs[0, 2].vlines(true[2], 0, line_height, colors="r", linestyles="dashed")
axs[1, 0].hist(data[rows_to_skip:, 3], num_bins)
axs[1, 0].vlines(true[3], 0, line_height, colors="r", linestyles="dashed")
axs[1, 1].hist(data[rows_to_skip:, 4], num_bins)
axs[1, 1].vlines(true[4], 0, line_height, colors="r", linestyles="dashed")
axs[1, 2].hist(data[rows_to_skip:, 5], num_bins)
axs[1, 2].vlines(true[5], 0, line_height, colors="r", linestyles="dashed")

data = np.loadtxt("error params.txt", delimiter=",")
# print(np.std(data[:, 6]) / (1.5 * 10 ** 11), np.std(data[:, 7]), np.std(data[:, 8]), np.std(data[:, 9]), np.std(data[:, 10]),
#       np.std(data[:, 11]))
#

scaled_errors = -1 * np.log10(data[:, 12])
min_error = np.min(scaled_errors)
max_error = np.max(scaled_errors)
normalised_error = (scaled_errors - min_error) / (max_error - min_error)
color_map = colormaps['hot']

fig_2, axs_2 = plt.subplots(2, 3)
axs_2[0, 0].scatter(data[:, 6], data[:, 7], alpha=1, c=normalised_error, cmap=color_map)
axs_2[0, 1].scatter(data[:, 6], data[:, 8], alpha=1, c=normalised_error, cmap=color_map)
axs_2[0, 2].scatter(data[:, 6], data[:, 9], alpha=1, c=normalised_error, cmap=color_map)
axs_2[1, 0].scatter(data[:, 6], data[:, 10], alpha=1, c=normalised_error, cmap=color_map)
axs_2[1, 1].scatter(data[:, 6], data[:, 11], alpha=1, c=normalised_error, cmap=color_map)
axs_2[1, 2].scatter(data[:, 7], data[:, 8], alpha=1, c=normalised_error, cmap=color_map)
plt.show()

