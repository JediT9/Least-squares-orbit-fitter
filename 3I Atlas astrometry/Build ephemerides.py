##
# Collect local sidereal time and save as file ready to be read into orbit fitter

# Imports
import numpy as np
from astroquery import jplhorizons
import pandas as pd


def get_horizons():
    """Get the ephemerides from horizons"""
    content = np.loadtxt("C:/Users/thkel/Documents/2025/ASTR211/3I observations/measured_ra_dec_all.txt")

    ephemerides = []
    earth_positions = []

    jds_used = content[:, 0]
    for date in jds_used:
        jpl_3iatlas = jplhorizons.Horizons(id="3I", id_type="smallbody", location="474", epochs=str(date))
        eph = jpl_3iatlas.ephemerides(quantities="7")
        jpl_earth = jplhorizons.Horizons(id="399", location="500@10", epochs=str(date))
        jpl_earth_pos = jpl_earth.vectors()
        print(date - np.min(jds_used))
        ephemerides.append(eph.to_pandas())
        earth_positions.append(jpl_earth_pos.to_pandas())

    stacked_eph = pd.concat(ephemerides, ignore_index=True)
    stacked_earth = pd.concat(earth_positions, ignore_index=True)
    print(stacked_earth)
    stacked_eph.to_csv(path_or_buf="observed ephemerides.csv", sep=",")
    stacked_earth.to_csv(path_or_buf="earth_positions_for_observations.csv")


def collate_txt_files(observed_file, jpl_file):
    """Collate the observed data with the information from jpl (LST)"""
    observed = np.loadtxt(observed_file)
    jpl = pd.read_csv(jpl_file, sep=",")
    local_sidereal_times = jpl["siderealtime"].to_numpy()
    print(local_sidereal_times)
    formatted_lst = np.zeros((1, 3))
    for time in local_sidereal_times:
        hour, remainder = (time // 1, time % 1)
        minute, remainder = ((remainder * 60) // 1, (remainder * 60) % 1)
        second = remainder * 60
        formatted_lst = np.vstack((formatted_lst, np.array([hour, minute, second])))

    formatted_ra = np.zeros((1, 3))
    for ra in observed[:, 1]:
        hour, remainder = (ra // 15, (ra / 15) % 1)
        minute, remainder = ((remainder * 60) // 1, (remainder * 60) % 1)
        second = remainder * 60
        formatted_ra = np.vstack((formatted_ra, np.array([hour, minute, second])))

    formatted_dec = np.zeros((1, 3))
    for dec in observed[:, 2]:
        sign = 1 if dec >= 0 else -1
        hour, remainder = ((abs(dec) // 1) * sign, abs(dec) % 1)
        minute, remainder = ((remainder * 60) // 1, (remainder * 60) % 1)
        second = remainder * 60
        formatted_dec = np.vstack((formatted_dec, np.array([hour, minute, second])))

    formatted_uncert = np.zeros((1, 2))
    for row in range(observed.shape[0]):
        ra_uncert = observed[row, 3] * 240
        dec_uncert = observed[row, 4] * 3600
        formatted_uncert = np.vstack((formatted_uncert, np.array([ra_uncert, dec_uncert])))

    full_data = np.hstack((observed[:, 0][:, np.newaxis], formatted_ra[1:, :], formatted_dec[1:, :], formatted_lst[1:, :],
                           formatted_uncert[1:, :]))
    np.savetxt("full_ephemerides.txt", full_data)


def build_earth_pos_txt(csv_file):
    """Convert the full csv from horizons into a txt file with time, x, y and z"""
    jpl_earth = pd.read_csv(csv_file, sep=",")
    jd, x, y, z = (jpl_earth["datetime_jd"].to_numpy(), jpl_earth["x"].to_numpy(), jpl_earth["y"].to_numpy(),
                   jpl_earth["z"].to_numpy())
    collated_array = np.transpose(np.vstack((jd, x, y, z)))
    np.savetxt("earth_pos_obs.txt", collated_array)
    print(collated_array)


def correct_inclination():
    """Correct the issues caused by allowing inclination values above pi"""
    fits = np.loadtxt("obs_error_params.txt", delimiter=",")

    for fit_index in range(fits.shape[0]):
        row = fits[fit_index, :]
        if row[8] > np.pi:
            row[8] = 2 * np.pi - row[8]
            row[9] = row[9] + np.pi
            row[10] = row[10] - np.pi
            fits[fit_index, :] = row

    np.savetxt("corrected_obs_error_params.txt", fits, delimiter=",")


correct_inclination()
# get_horizons()
# build_earth_pos_txt("earth_positions_for_observations.csv")
# collate_txt_files("C:/Users/thkel/Documents/2025/ASTR211/3I observations/measured_ra_dec_all.txt", "observed ephemerides.csv")
