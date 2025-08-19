import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
import matplotlib
import scipy.optimize
import cProfile
import pstats
import io
import copy
from datetime import datetime
from multiprocessing import Process, Lock

profiler = cProfile.Profile()
profiler.enable()
matplotlib.use('TkAgg')
lock = Lock()

# Define constants
AU: float = 1.4959787 * 10 ** 11  # https://adsabs.harvard.edu/full/1995A%26A...298..629H
G = 6.6743 * 10 ** -11  # NIST
M_SUN = 1.9891 * 10 ** 30  # https://link.springer.com/article/10.1134/S0038094612010054
true_anomalies = np.array([])
earth_tilt = 23.44 * np.pi / 180  # https://aa.usno.navy.mil/faq/asa_glossary
mt_john_lat = -43.9873942
mt_john_long = 170.465
mt_john_alt = 1026.04
earth_radius = 6371000
obs_radius = mt_john_alt + earth_radius
OCT_31_JD = 2460980
C = 299792458


@dataclass
class SkyPositionData:
    """Store the measurements for the position in the sky"""
    date: np.ndarray
    light: list[str]
    ra_meas_hour: np.ndarray
    ra_meas_min: np.ndarray
    ra_meas_sec: np.ndarray
    dec_meas_deg: np.ndarray
    dec_meas_min: np.ndarray
    dec_meas_sec: np.ndarray
    sdt_hour: np.ndarray
    sdt_min: np.ndarray
    sdt_sec: np.ndarray
    distance: np.ndarray
    range_rate: np.ndarray

    def ra_hours(self):
        """Convert the ra into an hour value"""
        ra_in_hours = self.ra_meas_hour + self.ra_meas_min / 60 + self.ra_meas_sec / 3600
        return ra_in_hours

    def ra_deg(self):
        """Convert ra into degrees"""
        ra_in_degrees = 15 * self.ra_hours()
        return ra_in_degrees

    def ra_rad(self):
        """Convert the ra into radians"""
        ra_in_rad = self.ra_deg() * np.pi / 180
        return ra_in_rad

    def dec_deg(self):
        """Convert the declination into degrees"""
        dec_in_deg = (abs(self.dec_meas_deg) + self.dec_meas_min / 60 + self.dec_meas_sec /
                      3600) * self.dec_meas_deg / abs(self.dec_meas_deg)
        return dec_in_deg

    def dec_rad(self):
        """Convert the declinations to radians"""
        dec_in_rad = self.dec_deg() * np.pi / 180
        return dec_in_rad

    def ecliptic_long(self):
        """Convert RA and dec into ecliptic longitude"""
        numerator = np.sin(self.ra_rad()) * np.cos(earth_tilt) + np.tan(self.dec_rad()) * np.sin(earth_tilt)
        denominator = np.cos(self.ra_rad())
        longitude = np.atan2(numerator, denominator)
        return longitude

    def ecliptic_latitude(self):
        """Convert RA and Dec into ecliptic latitude"""
        sin_beta = (np.sin(self.dec_rad()) * np.cos(earth_tilt)) - (np.cos(self.dec_rad()) * np.sin(self.ra_rad()) *
                                                                    np.sin(earth_tilt))
        lat = np.asin(sin_beta)
        return lat

    def lst_rad(self):
        """Calculate the local sidereal time in radians"""
        lst_hours = self.sdt_hour + self.sdt_min / 60 + self.sdt_sec / 3600
        lst_rad = lst_hours * np.pi / 12
        return lst_rad

    def get_row(self, index, rad=True):
        """Return the values associated with the given row"""
        if rad:
            values = [self.date[index], self.light[index], self.ra_rad()[index], self.dec_rad()[index],
                      self.distance[index], self.range_rate[index]]
        else:
            values = [self.date[index], self.light[index], self.ra_deg()[index], self.dec_deg()[index],
                      self.distance[index], self.range_rate[index]]
        data_frame = pd.DataFrame(values, columns=["date", "light", "ra", "dec", "distance", "range_rate"])
        return data_frame


def load_data(filename):
    """Load the data from the file into a data frame"""
    content = open(filename, "r")
    values = [[], [], [], [], [], [], [], [], [], [], [], [], []]
    for line in content:
        separated = [entry for entry in line.split(" ") if entry != ""]
        if len(separated) != 13:
            separated.insert(1, "d")
        for index in range(len(values)):
            try:
                values[index].append(float(separated[index]))
            except ValueError:
                values[index].append(separated[index])
    data_array = SkyPositionData(np.array(values[0]), values[1], np.array(values[2]), np.array(values[3]),
                                 np.array(values[4]), np.array(values[5]), np.array(values[6]), np.array(values[7]),
                                 np.array(values[8]), np.array(values[9]), np.array(values[10]), np.array(values[11]),
                                 np.array(values[12]))
    return data_array


def load_earth_pos(filename, ephemerides: SkyPositionData):
    """Load in the Earth's position from the file"""
    content = np.loadtxt(filename, delimiter=",")
    data_frame = pd.DataFrame(content, columns=["date", "x", "y", "z"])
    pos_when_observed = pd.DataFrame(columns=["date", "x", "y", "z"])
    for index, row in data_frame.iterrows():
        if row.date in ephemerides.date:
            pos_to_use = observer_pos(row[1:], ephemerides.lst_rad()[np.where(ephemerides.date == row.date)])
            pos_when_observed.loc[len(pos_when_observed)] = pos_to_use
    return pos_when_observed


def observer_pos(all_earth, local_sidereal):
    """Convert the earth center coordinates to the observatory coordinates"""

    x = obs_radius * np.cos(deg_to_rad(mt_john_lat)) * np.cos(local_sidereal[0])
    y = obs_radius * np.cos(deg_to_rad(mt_john_lat)) * np.sin(local_sidereal[0])
    z = obs_radius * np.sin(deg_to_rad(mt_john_lat))
    obs_position = np.array([x, y, z]) / 1000
    pos_to_use = all_earth + obs_position
    return pos_to_use


def xyz_to_ra_dec(xyz_pos, xyz_observer):
    """Calculate the ra/dec coordinates of from the xyz heliocentric position"""

    rel_pos = xyz_pos - xyz_observer
    long = np.atan2(rel_pos[1], rel_pos[0])
    lat = np.atan2(rel_pos[2], np.sqrt(rel_pos[1] ** 2 + rel_pos[0] ** 2))
    ra = np.atan2(np.sin(long) * np.cos(earth_tilt) - np.tan(lat) * np.sin(earth_tilt), np.cos(long))
    dec = np.asin(np.sin(lat) * np.cos(earth_tilt) + np.cos(lat) * np.sin(long) * np.sin(earth_tilt))
    return ra, dec


def ra_hours(ra_deg):
    """Convert ra from degrees into hours"""
    return ra_deg / 15


def deg_to_rad(deg_value) -> float:
    """Convert from degrees to radians"""
    return deg_value * np.pi / 180


def create_data(actual_eph: SkyPositionData):
    """Create fudged data to experiment on"""
    modified_eph = copy.deepcopy(actual_eph)
    modified_eph.ra_meas_sec = actual_eph.ra_meas_sec + np.random.normal(scale=2.0, size=actual_eph.ra_meas_sec.size)
    modified_eph.dec_meas_sec = actual_eph.dec_meas_sec + np.random.normal(scale=2.0, size=actual_eph.ra_meas_sec.size)
    return modified_eph


def calc_pos(a, e, i, omega, w, v):
    """Using the provided parameters, calculate the body's position"""
    p = a * (1 - e ** 2)
    pqw_x = p * np.cos(v) / (1 + e * np.cos(v))
    pqw_y = p * np.sin(v) / (1 + e * np.cos(v))
    pqw_z = 0
    pos_pqw = np.array([pqw_x, pqw_y, pqw_z])
    conversion_matrix = np.array([[np.cos(omega) * np.cos(w) - np.sin(omega) * np.sin(w) * np.cos(i),
                                   -np.cos(omega) * np.sin(w) - np.sin(omega) * np.cos(w) * np.cos(i),
                                   np.sin(omega) * np.sin(i)],
                                  [np.sin(omega) * np.cos(w) + np.cos(omega) * np.sin(w) * np.cos(i),
                                   -np.sin(omega) * np.sin(w) + np.cos(omega) * np.cos(w) * np.cos(i),
                                   -np.cos(omega) * np.sin(i)],
                                  [np.sin(w) * np.sin(i), np.cos(w) * np.sin(i), np.cos(i)]])
    ijk_position = np.linalg.matmul(conversion_matrix, pos_pqw)
    return ijk_position


def time_from_true_anomaly(v, e, a, m):
    """Calculate the expected time from the true anomaly"""
    if e < 1:
        phi = 2 / ((1 - e ** 2) ** (3 / 2)) * np.atan2(((1 - e) / (1 + e)) ** (1 / 2) * np.tan(v / 2), 1) - \
              (e * np.sin(v)) / ((1 - e ** 2) * (1 + e * np.cos(v)))
    else:
        phi = -2 / ((e ** 2 - 1) ** (3 / 2)) * np.arctanh((e - 1) / (e ** 2 - 1) ** (1 / 2) * np.tan(v / 2)) + \
              (e * np.sin(v)) / ((e ** 2 - 1) * (1 + e * np.cos(v)))
    mu = m * G
    h = np.sqrt(mu * a * (1 - e ** 2))
    time_since_perihelion = phi * h ** 3 / mu ** 2
    return time_since_perihelion


def root_of_anomaly(v, e, a, m, time):
    """Rearrange so the root can be found"""
    return time_from_true_anomaly(v, e, a, m) - time


def distance_to_prediction(exp_pos, earth_pos, direction):
    """Calculate the distance between the expected position and the observation line"""

    # Transform origin to the Earth
    exp_pos_wrt_earth = exp_pos - earth_pos * 1000

    # Find the closest point along visual line
    projection = abs(np.dot(exp_pos_wrt_earth, direction) / (np.dot(direction, direction))) * direction

    # Find distance squared
    distance_vec = exp_pos_wrt_earth - projection
    distance = ((distance_vec[0] / AU) ** 2 + (distance_vec[1] / AU) ** 2 + (distance_vec[2] / AU) ** 2)
    angular_distance = np.sqrt(distance) * 3600 / np.sqrt(np.dot(projection / AU, projection / AU))
    return angular_distance, exp_pos_wrt_earth


def convert_to_true_units(scaled_params):
    """Converts the scaled parameters back to true values.
    expected units:
    a - AU, e - normal, i - rad, Omega - rad, w - rad, T - weeks since October 31st
    """

    a_scaled, e, i, omega, w, t_peri_scaled = scaled_params

    # Convert units
    a = a_scaled * AU
    t_peri = 7 * t_peri_scaled + OCT_31_JD

    return a, e, i, omega, w, t_peri


def adjust_light_time(params, obs_time, earth_pos, units=False):
    """Adjust the expected time for light-travel time"""
    if not units:
        a, e, i, omega, w, perihelion_date = convert_to_true_units(params)
    else:
        a, e, i, omega, w, perihelion_date = params
    init_time = (obs_time - perihelion_date) * 86400
    curr_time = init_time
    for iteration in range(3):
        try:
            roots = scipy.optimize.root_scalar(lambda x: root_of_anomaly(x, e, a, M_SUN, curr_time),
                                               bracket=(
                                               -np.pi + np.acos(1 / e) + 0.001, np.pi - np.acos(1 / e) - 0.001))
        except ValueError:
            print(params)
        true_anomaly = roots.root
        expected_position = calc_pos(a, e, i, omega, w, true_anomaly)
        distance = np.dot(expected_position - earth_pos, expected_position - earth_pos)
        curr_time = init_time - np.sqrt(distance) / C
    return true_anomaly


def calc_error_in_orbit(params, observations: SkyPositionData, all_earth, is_minimizing=True, method="minimize"):
    """Calculate the error in an orbit"""
    global true_anomalies
    # Loop through each observation
    a, e, i, omega, w, perihelion_date = convert_to_true_units(params)
    sum_square_error = 0
    errors = np.array([])
    projections = np.zeros((1, 3))
    for index in range(observations.date.size):
        true_anomaly = adjust_light_time(params, observations.date[index], all_earth.iloc[index, 1:])
        true_anomalies = np.append(true_anomalies, true_anomaly)
        expected_position = calc_pos(a, e, i, omega, w, true_anomaly)
        earth_position = np.array([all_earth.iloc[index, 1], all_earth.iloc[index, 2], all_earth.iloc[index, 3]])
        direction = np.array(
            [np.cos(observations.ecliptic_long()[index]) * np.cos(observations.ecliptic_latitude()[index]),
             np.cos(observations.ecliptic_latitude()[index]) * np.sin(observations.ecliptic_long()[index]),
             np.sin(observations.ecliptic_latitude()[index])])
        distance_sq, projection = distance_to_prediction(expected_position, earth_position, direction)
        projections = np.vstack((projections, projection))
        errors = np.append(errors, distance_sq)
        sum_square_error += distance_sq

    if is_minimizing and method == "minimize":
        return sum_square_error
    elif is_minimizing and method == "curve_fit":
        return errors
    else:
        return projections, sum_square_error


def fit_orbit_parameters(observations: SkyPositionData, initial_guess: tuple[float, float, float, float, float, float],
                         all_earth, method="minimize"):
    """Fit the 6 keplerian parameters to the observations"""
    if method == "minimize":
        sol = scipy.optimize.minimize(calc_error_in_orbit, np.array(initial_guess),
                                      args=(observations, all_earth, True),
                                      bounds=((-10, -0.1), (1.01, 15), (0, 2 * np.pi), (0, 2 * np.pi),
                                              (0, 2 * np.pi), (-4, 4)))
        sol_params = sol.x
    # elif method == "curve_fit":
    #     obs_indexes = np.arange(0, len(observations.light), 1)
    #     ideal_values = np.zeros((1, len(observations.light)))
    #     sol_params, covar = scipy.optimize.curve_fit(lambda x, a, e, i, omega, w, t:
    #                                                  calc_error_in_orbit((a, e, i, omega, w, t), observations,
    #                                                                      all_earth, method="curve_fit"), obs_indexes,
    #                                                  ideal_values, initial_guess)
    elif method == "differential_evolution":
        sol = scipy.optimize.differential_evolution(calc_error_in_orbit, x0=np.array(initial_guess),
                                                    args=(observations, all_earth, True),
                                                    bounds=((-10, -0.1), (1.01, 15), (0, 2 * np.pi),
                                                            (0, 2 * np.pi), (0, 2 * np.pi), (-4, 4)),
                                                    workers=4)
        sol_params = sol.x
    return sol_params


def testing_on_earth():
    """Testing the orbit model"""
    jpl_positions = np.loadtxt("Earth over year.txt", delimiter=",")
    earth_orbit = (1.0000011, 0.01671022, deg_to_rad(0), deg_to_rad(360 - 11.26064), deg_to_rad(102.94719 + 11.26064))
    positions_to_calc = np.linspace(0, 360, 360)
    expected_xs = []
    expected_ys = []
    for angle in positions_to_calc:
        expected_positions = calc_pos(earth_orbit[0], earth_orbit[1], earth_orbit[2], earth_orbit[3], earth_orbit[4],
                                      deg_to_rad(angle))
        expected_xs.append(expected_positions[0])
        expected_ys.append(expected_positions[1])
    axs = plt.axes()
    axs.plot(jpl_positions[:, 1], jpl_positions[:, 2], label="JPL positions")
    axs.plot(expected_xs, expected_ys, label="Expected")
    axs.legend()
    plt.show()


def brute_error_varying_data(initial_params, observations, all_earth, iterations, lock_to_use):
    """Compute the given number of iterations with random data each time"""

    # Open file to record results as found
    for iteration in range(iterations):
        false_data = create_data(observations)
        fitted_scaled = fit_orbit_parameters(false_data, initial_params, all_earth)
        true_fitted = convert_to_true_units(fitted_scaled)
        error = calc_error_in_orbit(fitted_scaled, false_data, all_earth)
        with lock_to_use:
            with open("error orbit fits.txt", "a") as content:
                content.write(",".join([str(element) for element in true_fitted] + [str(error)]))
                content.write("\n")
        print(iteration, datetime.now())
        # print(false_data.ra_meas_sec[0])
        print(fitted_scaled - initial_params)
        print(
            f"A: {true_fitted[0]}, e: {true_fitted[1]}, (i, Omega, w): {true_fitted[2], true_fitted[3], true_fitted[4]}, Tp: {true_fitted[5]}")
        print(f"Error: {error} arc-seconds")


def brute_force_varying_parameters(true_params, observations: SkyPositionData, all_earth, iterations, file_lock: Lock):
    """Compute the given number of iterations, randomly varying one of the parameters"""

    # Possible values
    possible_a = np.linspace(-2, -0.1, 1000)
    possible_e = np.linspace(1.01, 10, 1000)
    possible_angle = np.linspace(0, 2 * np.pi, 1000)
    possible_t = np.linspace(-4, 4, 1000)

    # Open file to record fits as it goes
    for iteration in range(iterations):
        # param_to_vary = np.random.randint(0, 6)
        # new_params = np.array(copy.deepcopy(true_params))
        # scale_factor = np.random.normal()
        # possible_param_change = scale_factor * np.array([0.2, 3, 3, 3, 3, 3/7])
        # new_params[param_to_vary] = new_params[param_to_vary] + possible_param_change[param_to_vary]

        new_params = np.array([possible_a[np.random.randint(0, 1000)], possible_e[np.random.randint(0, 1000)],
                               possible_angle[np.random.randint(0, 1000)],
                               possible_angle[np.random.randint(0, 1000)],
                               possible_angle[np.random.randint(0, 1000)], possible_t[np.random.randint(0, 1000)]])

        # Check bounds
        bounds = [(-10, -0.1), (1.01, 15), (0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi), (-4, 4)]
        for index in range(new_params.size):
            if new_params[index] < bounds[index][0]:
                new_params[index] = bounds[index][0]
            elif new_params[index] > bounds[index][1]:
                new_params[index] = bounds[index][1]

        fitted_scaled = fit_orbit_parameters(observations, new_params, all_earth)
        true_fitted = convert_to_true_units(fitted_scaled)
        error = calc_error_in_orbit(fitted_scaled, observations, all_earth)
        with lock:
            with open("error params.txt", "a") as content:
                content.write(",".join([str(element) for element in new_params] +
                                       [str(element) for element in true_fitted] + [str(error)]))
                content.write("\n")
        print(iteration, datetime.now())
        print(new_params)
        print(error)


def run_multi_thread(function, args, lock_to_use, threads=1):
    """Run the supplied functions simultaneously across supplied number of threads"""

    created_threads = [Process(target=function, args=(*args, lock_to_use)) for num in range(threads)]

    for thread in created_threads:
        thread.start()

    for thread in created_threads:
        thread.join()


axs = plt.axes()
ephemeris = load_data("Ephemerides.txt")
earth_positions = load_earth_pos("Earth.txt", ephemeris)

print(convert_to_true_units((-2.638825934940408E-01, 6.141051778951341E+00, deg_to_rad(175.1132392476728),
                             deg_to_rad(322.1595230967855), deg_to_rad(128.0088682076279), -2 / 7)))

# print(adjust_light_time((-2.638825934940408E-01, 6.141051778951341E+00, deg_to_rad(175.1132392476728),
#                          deg_to_rad(322.1595230967855), deg_to_rad(128.0088682076279), -2/7), 2460915.791666667,
#                         (1.368325216108015E+08 * 1000, -6.412592053894778E+07 * 1000,  3.393047681309283E+03 * 1000)))

# brute_error_varying_data((-2.638825934940408E-01, 6.141051778951341E+00, deg_to_rad(175.1132392476728),
#                           deg_to_rad(322.1595230967855), deg_to_rad(128.0088682076279), -0.288714285), ephemeris,
#                          earth_positions, 1000)

# brute_force_varying_parameters((-2.638825934940408E-01, 6.141051778951341E+00, deg_to_rad(175.1132392476728),
#                                 deg_to_rad(322.1595230967855), deg_to_rad(128.0088682076279), -2/7), ephemeris,
#                                earth_positions, 1000, lock)

if __name__ == "__main__":
    # run_multi_thread(brute_force_varying_parameters, ((-2.638825934940408E-01, 6.141051778951341E+00,
    #                                                    deg_to_rad(175.1132392476728), deg_to_rad(322.1595230967855),
    #                                                    deg_to_rad(128.0088682076279), -2 / 7), ephemeris,
    #                                                   earth_positions, 1000), lock, threads=6)

    # fitted_scaled = fit_orbit_parameters(ephemeris, (-2.638825934940408E-01, 6.141051778951341E+00,
    #                                                  deg_to_rad(175.1132392476728), deg_to_rad(322.1595230967855),
    #                                                  deg_to_rad(128.0088682076279), -2/7),
    #                                      earth_positions, "minimize")
    # fitted = convert_to_true_units(fitted_scaled)

    xs = np.linspace(-np.pi + np.acos(1 / 6) + 0.001, np.pi - np.acos(1 / 6) - 0.001, 1000)
    ys = time_from_true_anomaly(xs, 6, -1.00000000e+02, M_SUN)
    # plt.plot(xs, ys)
    # plt.show()

    # fitted_positions = np.zeros((1, 3))
    # for true in np.linspace(-np.pi + np.acos(1 / fitted[1]) + 0.2, np.pi - np.acos(1 / fitted[1]) - 0.2, 1000):
    #     position = calc_pos(fitted[0], fitted[1], fitted[2], fitted[3], fitted[4], true)
    #     fitted_positions = np.vstack((fitted_positions, position))
    # axs.plot(fitted_positions[1:, 0], fitted_positions[1:, 1], "y-", label="Fitted orbit")
    # closest_approaches, error = calc_error_in_orbit(fitted_scaled, ephemeris, earth_positions,
    #                                                 False)
    # axs.plot(closest_approaches[1:, 0] + earth_positions.x * 1000,
    #          closest_approaches[1:, 1] + earth_positions.y * 1000, "y.")
    # print(error)
    # print(fitted_scaled)
    # print(fitted)
    # print((np.array(fitted) / np.array((-2.638825934940408E-01 * AU, 6.141051778951341E+00, deg_to_rad(175.1132392476728),
    #                                    deg_to_rad(322.1595230967855), deg_to_rad(128.0088682076279),
    #                                     2460977.9789309348))) - 1)

    fitted_orbits = np.loadtxt("error params.txt", delimiter=",")
    # fitted_orbits = fitted_orbits[fitted_orbits[:, 12] < 80, :]
    print(-np.log10(fitted_orbits[:, 12]))
    min_error = np.max(-np.log10(fitted_orbits[:, 12]))
    max_error = np.min(-np.log10(fitted_orbits[:, 12]))
    print(max_error, min_error)
    for orbit_index in range(fitted_orbits[:, 0].size):
        positions = np.zeros((1, 3))
        for true in np.linspace(-np.pi + np.acos(1 / fitted_orbits[orbit_index, 7]) + 0.001,
                                np.pi - np.acos(1 / fitted_orbits[orbit_index, 7]) - 0.001, 1000):
            position = calc_pos(fitted_orbits[orbit_index, 6], fitted_orbits[orbit_index, 7], fitted_orbits[orbit_index, 8],
                                fitted_orbits[orbit_index, 9], fitted_orbits[orbit_index, 10], true)
            positions = np.vstack((positions, position))
        axs.plot(positions[1:, 0], positions[1:, 1], color=((-np.log10(fitted_orbits[orbit_index, 12]) - max_error) /
                                                            (min_error - max_error), 0, 0),
                 alpha=(-np.log10(fitted_orbits[orbit_index, 12]) - max_error) / (min_error - max_error),
                 label=fitted_orbits[orbit_index, 12])
    axs.legend()

    # propagate_direction = np.linspace(0, 5 * 10 ** 11, 1000)
    # for index in range(len(ephemeris.light)):
    #     if index == 0:
    #         axs.plot(earth_positions.x[index] * 1000 + np.cos(ephemeris.ecliptic_long()[index]) *
    #                  np.cos(ephemeris.ecliptic_latitude()[index]) * propagate_direction,
    #                  earth_positions.y[index] * 1000 + np.sin(ephemeris.ecliptic_long()[index]) *
    #                  np.cos(ephemeris.ecliptic_latitude()[index]) * propagate_direction, "g-", label="Observation line")
    #     else:
    #         axs.plot(earth_positions.x[index] * 1000 + np.cos(ephemeris.ecliptic_long()[index]) *
    #                  np.cos(ephemeris.ecliptic_latitude()[index]) * propagate_direction,
    #                  earth_positions.y[index] * 1000 + np.sin(ephemeris.ecliptic_long()[index]) *
    #                  np.cos(ephemeris.ecliptic_latitude()[index]) * propagate_direction, "g-")
    #
    # axs.plot(AU * np.cos(np.linspace(0, 2 * np.pi, 1000)), AU * np.sin(np.linspace(0, 2 * np.pi, 1000)))
    # axs.plot(earth_positions.x * 1000 + np.cos(ephemeris.ecliptic_long()) * np.cos(ephemeris.ecliptic_latitude()) *
    #          ephemeris.distance * AU, earth_positions.y * 1000 + np.sin(ephemeris.ecliptic_long()) *
    #          np.cos(ephemeris.ecliptic_latitude()) * ephemeris.distance * AU, "m.")

    # Plot the expected positions on December 20
    # data = np.loadtxt("error orbit fits.txt", delimiter=",")
    # standard_deviations = np.array([np.std(data[:, 0]), np.std(data[:, 1]), np.std(data[:, 2]),
    #                                 np.std(data[:, 3]), np.std(data[:, 4]), np.std(data[:, 5])])
    # param_means = np.array([np.median(data[:, 0]), np.median(data[:, 1]), np.median(data[:, 2]),
    #                         np.median(data[:, 3]), np.median(data[:, 4]), np.median(data[:, 5])])
    # dec_20 = 2461029.5
    # for orbit_num in range(4000):
    #     param_modifier = np.random.normal(size=6)
    #     params_to_plot = param_means + standard_deviations * param_modifier
    #     true_anomaly_dec_20 = adjust_light_time(params_to_plot, dec_20,
    #                                             np.array([5.179287755554469E+06,  1.471008280186807E+08,
    #                                                       -8.397315859936178E+03]) * 1000, True)
    #     position = calc_pos(params_to_plot[0], params_to_plot[1], params_to_plot[2], params_to_plot[3], params_to_plot[4],
    #                         true_anomaly_dec_20)
    #     ra_obs, dec_obs = xyz_to_ra_dec(position, np.array([5.179287755554469E+06,  1.471008280186807E+08,
    #                                                         -8.397315859936178E+03]) * 1000)
    #     ra_min = ra_obs * 12 * 60 / np.pi
    #     dec_min = dec_obs * 180 * 60 / np.pi
    #     axs.plot(ra_min, dec_min, "r.", alpha=0.05)
    #     print(orbit_num)
    #
    # axs.set_xlabel("Right ascension [']")
    # axs.set_ylabel("Declination [']")

    actual_positions = np.loadtxt("actual 3I pos.txt", delimiter=",")
    axs.plot(actual_positions[:, 0] * AU, actual_positions[:, 1] * AU, "r-")
    axs.plot([0], [0], "r.")
    axis_lim = 7 * 10 ** 8
    profiler.disable()

    # Format and display the results
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats('ncalls')
    stats.print_stats(200)  # Print top 20 functions
    # print(s.getvalue())
    plt.show()
