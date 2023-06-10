import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy.special as sp
import csv
from scipy.optimize import curve_fit
from astropy.coordinates import EarthLocation,SkyCoord
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import AltAz


# Set the default text font size
plt.rc('font', size=16)
# Set the axes title font size
plt.rc('axes', titlesize=20)
# Set the axes labels font size
plt.rc('axes', labelsize=15)
# Set the font size for x tick labels
plt.rc('xtick', labelsize=15)
# Set the font size for y tick labels
plt.rc('ytick', labelsize=15)
# Set the legend font size
plt.rc('legend', fontsize=18)
# Set the font size of the figure title
plt.rc('figure', titlesize=25)

def makeFiles():
    """
    Loads the CSV files, makes conversions, and puts it all into a nice array
    """
    global runs, numbers
    numbers = []  # file numbers [38, 41, 43, 44, 45, 46 | 48, 52, 53, 54, 55, 56, 57, 58, 59]

    runs = {
        # dictionary of file Run Numbers
        # Run Number : [[number, frequency, amplitude], ...]# numbers 1-461
    }

    directory = Path('data_files_2023_05_19')  # change this path to where your csv files are
    # Loop over files in the directory
    for file in directory.iterdir():
        number = str(file)[-6:-4]  # gets the file number
        numbers.append(number)
        templist = []
        if file.is_file():
            with open(file, newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    templist.append(row)

        templist = templist[12:-1]  # removes heading the last character
        templist = np.array(templist, dtype=float)  # converts to np array of floats

        #### Conversions ####
        convert = lambda x: 10**(x/10)  # converts dB to W
        c, D = 299792458, 1.8  # speed of light, diameter of disk
        RBW = 30000.0  # 1 Resolution Bandwidth = 30000.0 Hz
        templist[:, -1] = (convert(templist[:, -1]) / 1000) / RBW  # converts dBm/RBW to W/Hz
        # templist[:, -1] is the amplitude of the measurements, the power dBm

        cable_length = 60 / 3.281  # 60 ft to m
        cable_attenuation = convert(3)  # in W/m * m = W
        templist[:, -1] = templist[:, -1] * cable_attenuation  # corrects for attenuation

        LNA_gain = convert(37)  # in W
        templist[:, -1] = templist[:, -1] / LNA_gain  # corrects for gain; this is the power hitting the antenna

        templist[:, -1] = templist[:, -1] * 2  # corrects for polarization; this is total power hitting the ground

        runs[number] = templist  # adds to dictionary
        # print(runs[number])

def fits(time):
    """
    Fits gaussian and calculates residuals, B, T and N
    -----------------------------------------------------
    time(string): 'am' or 'pm' for the two observation times.
    """
    Ns, sigNs = [], []
    if time == 'am':
        ytext1 = 0.05
        nums = numbers[:6]  # gets the morning numbers
        guess = (3.9e-18, .035, 1420.57, 3.5e-16, -4.04e-23)  # amplitude, sigma, mu, c
        PDT = np.array(['07:00:00', '07:20:00', '07:26:00',  # 38, 41, 43
                        '07:32:00', '07:38:00', '07:44:00'])  # 44, 45, 46
            
    elif time == 'pm':
        ytext1 = 0.55
        nums = numbers[6:]  # gets the afternoon numbers
        guess = (7e-18, .06, 1420.2263, 1.4e-16, -5.81e-23)  # amplitude, sigma, mu, c
        PDT = np.array(['16:40:00', '16:54:00', '17:00:00',  # 48, 52, 53
                        '17:06:00', '17:11:00', '17:17:00',  # 54, 55, 56
                        '17:23:00', '17:29:00', '17:35:00'])

    for n in nums:
        xdata = runs[n][:, 1] / 1e6  # frequecnies, converts Hz to MHz
        ydata = runs[n][:, 2]  # W / Hz

        if False:  # change this to True to plot raw data
            plt.plot(xdata, ydata)
            plt.title('Power vs. Wavelength')
            plt.ylabel('W / Hz')
            # plt.savefig('Data.pdf')
            plt.show()

        x = 2  # factor to scale figures by
        fig, ax = plt.subplots(2, 1, figsize=(6.4 * x, 4.8 * x), sharex=True)  # creates figures

        ax[0].plot(xdata, ydata)  # plots the data
        fig.text(0.5, 0.04, 'Frequency (MHz)', ha='center')
        # fig.text(0.04, 0.5, 'W / Hz', va='center', rotation='vertical')

        popt, pcov = curve_fit(gauss, xdata, ydata, p0=guess)  # fits a guassian
        print('Amp={}  sig={}  mu={}  c={}'.format(*popt))  # prints the best fit parameters
        fit = gauss(xdata, *popt)  # calculates the fit curve
        ax[0].plot(xdata, fit)  # plots the fit on top of data
        ax[0].axvline(1420.4, color='r')  # plots the expected frequency
        ax[0].set_title('Power vs. Wavelength')
        ax[0].set_ylabel('W / Hz')
        ax[0].text(ytext1, 0.8, "Baseline = {:.2e} * x + {:.2e}".format(popt[-1], popt[-2]), transform=ax[0].transAxes)

        sig_nums = 4  # number of sigmas to start counting background
        stop1 = popt[2] - sig_nums * popt[1]  # 3 sigma left
        start2 = popt[2] + sig_nums * popt[1]  # 3 sigma right
        index = find_closest_indices(xdata, [stop1, start2])  # get the index for those

        start1, stop1 = 0, index[0]
        start2, stop2 = index[1], len(xdata)  # indices for background range

        residual = ydata - fit  # gets the residuals
        residual1 = ydata[start1:stop1] - fit[start1:stop1]  # left
        residual2 = ydata[start2:stop2] - fit[start2:stop2]  # right
        res_sum = np.sum(residual)
        ax[1].plot(xdata, residual)  # plots the residuals on another plot
        ax[1].set_title('Residual')
        ax[1].set_ylabel('W / Hz')
        ax[1].text(1, 0.5, "Sum = \n{:.2e}".format(res_sum), transform=ax[1].transAxes)
        ax[1].axvline(xdata[stop1], color='r')  # plots the expected frequency
        ax[1].axvline(xdata[start2], color='r')  # plots the expected frequency

        sub = ydata - popt[-2] - xdata * popt[-1]  # subtract the background c, this is now dp/df
        B, sigB, T, sigT = BT(sub, np.sqrt(np.abs(sub)))  # calculates B and T for each point

        ymaxloc = np.where(ydata == np.max(ydata))[0][0]  # get the index of max y
        xmax = xdata[ymaxloc]  # gets the correspodning x value

        v = -0.21 / 1000 * (xdata - xmax) * 1e6  # v = -lamda * delf; 21 cm line (in km) * datapoint - central point (in Hz)
        spacing = np.abs((v[-1] - v[0]) / len(v))  # velocity spacing
        constants = 1.8224e18 * spacing
        N = constants * np.sum(T)  # Column Density im cm^-2

        rms1 = RMS(residual1);
        Terr1 = BT(1, rms1)[-1]  # left side
        rms2 = RMS(residual2);
        Terr2 = BT(1, rms2)[-1]  # right side

        length = len(residual1) + len(residual2)
        sigN = np.sqrt(length * (Terr1 ** 2 + Terr2 ** 2)) * constants

        Ns.append(N);
        sigNs.append(sigN)
        print('N = {} +- {}'.format(N, sigN))
        print('-------------------------------------------')
        # plt.savefig('Gauss fit {}.pdf'.format(n))
        plt.show()

        if n == 3:  # set the number of the run to see B plot for
            plt.plot(xdata, B)
            plt.title('Brightness Distribution')
            plt.ylabel('W / Hz / m^2')
            plt.savefig('B {}.pdf'.format(n))
            plt.show()

    Ns = np.array(Ns); sigNs = np.array(sigNs)

    lat, lon = RADEC(time, p=False)
    lat = np.array(lat); lon = np.array(lon)
    lat = np.round(lat, decimals=2); lon = np.round(lon, decimals=2)
    fig, ax1 = plt.subplots()
    # Plot with the first x-axis
    ax1.errorbar(range(len(nums)), Ns/1e21, yerr=sigNs/1e21, linestyle='') ###### Decide how you want to format these
    ax1.scatter(range(len(nums)), Ns/1e21, color='r', zorder=10)
    ax1.set_xticks(range(len(nums)))
    ax1.set_xticklabels(PDT)
    ax1.set_xlabel('Index')
    ax1.set_xlabel('Time (PDT)')
    ax1.set_ylabel('Helium Column Density (cm^-2) 1e21')
    # Create a second x-axis
    ax2 = ax1.twiny()
    # Set the second x-axis ticks and labels
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(range(len(nums)))
    ax2.set_xticklabels(lon)
    ax2.set_xlabel('Galatic Longetude (Degrees)')
    # plt.savefig('N prgression.pdf')
    plt.show()


def skymap():
    lat1, lon1 = RADEC('am', p=False)
    lat2, lon2 = RADEC('pm', p=False)

    gal1 = SkyCoord(lat1, lon1, frame='galactic', unit=u.deg)
    gal2 = SkyCoord(lat2, lon2, frame='galactic', unit=u.deg)

    plt.subplot(111, projection='aitoff')
    plt.grid(True)
    plt.plot(gal1.l.wrap_at('180d').radian, gal1.b.radian, label='Morning')
    plt.plot(gal2.l.wrap_at('180d').radian, gal2.b.radian, label='Afternoon')

    plt.legend()
    # plt.savefig('skymap.pdf')
    plt.show()


def airyintegral():
    """
    From Oliver. Don't need to change anything. This is only dependent on the telescope and we only used one.
    """
    # set up an x axis running from 0.0 to 5.0 (+0.001 to prevent a divide by zero)
    x = np.arange(5000) / 1000. + 1. / 1000.

    # Define the Airy function -- it's (J1(x)/x)**2 where J1 is a Bessel function
    airy = (sp.jv(1, x) / x) ** 2

    # Renormalize it to be =1 at the peak:
    airy = airy / np.max(airy)

    # find where it crosses zero (Airy disk), and limit the range to that point.
    # Don't ask me why it has to be w[0][0], that's Python weirdness.
    w = np.where(airy == np.min(airy))
    print(w[0][0]);
    3831
    x = x[0:w[0][0]]
    airy = airy[0:w[0][0]]

    # show what it looks like:
    plt.plot(x, airy)
    plt.plot(x, airy * 0.)
    plt.plot(x, airy * 0. + 1.0, color='r')
    plt.show()

    # Since this is falling off in 2 dimensions, we get solid angle with integral( airy*2pi*r*dr )
    # But we want this as a fraction of what it would be if it didn't fall off at all, so we can divide
    # this by integral ( 2pi*r*dr ) and the constant factors 2pi*dr cancel out, so simply:

    integralairy = np.sum(airy * x)
    integralnorm = np.sum(x)
    print(integralairy / integralnorm)


def RMS(array):
    mean = np.average(array)
    sumgrand = (array - mean) ** 2
    length = len(array)
    return np.sqrt(np.sum(sumgrand) / length)


def find_closest_indices(xdata, target_values):
    xdata = np.array(xdata)
    target_values = np.array(target_values)
    indices = np.abs(xdata[:, np.newaxis] - target_values).argmin(axis=0)
    return indices


def gauss(x, amplitude, sig, mu, c, slope):
    a = 1 / (sig * np.sqrt(2*np.pi))
    b = -0.5 * ((x - mu) / sig) ** 2
    return amplitude * (a * np.exp(b)) + slope * x + c


def BT(dpdf, sigdpdf, useB=False):
    """
    Calculates brightness distribution (B) and brightness temperature (T).
    T is calculated using either dpdf or B depending on useB.
    B must be in units of W / Hz
    Returns B,T.
    """
    beta = 0.5
    wl = 0.21  # m
    kb = 1.380649e-23  # m^2 kg s^-2 K^-1
    B = dpdf / (beta * wl ** 2)
    sigB = sigdpdf / (beta * wl ** 2)

    if useB:  # calcualtes T based on calcualted B
        T = wl ** 2 * B / (2 * kb)
    else:  # calcualtes T based on inputted dpdf
        T = dpdf / (2 * beta * kb)
        sigT = sigdpdf / (2 * beta * kb)  # fix the errors things

    return B, sigB, T, sigT
       

def RADEC(time, p=True):
    ucsc_location = EarthLocation(lat='36.9905', lon='-122.0584', height=341 * u.m)  # Location of UCSC
    azimuth = 180 + 1 / 24 * 360  # Azimuth for HA = 1

    if time == 'am':
        telelv = 77.5  # elevation of telescope, degrees
        UTC = np.array(['14:00:00', '14:20:00', '14:26:00',  # 38, 41, 43
                        '14:32:00', '14:38:00', '14:44:00'])  # 44, 45, 46
    if time == 'pm':
        telelv = 56.7  # elevation of telescope, degrees
        UTC = np.array(['23:40:00', '23:54:00', '00:00:00',  # 48, 52, 53
                        '00:06:00', '00:11:00', '00:17:00',  # 54, 55, 56
                        '00:23:00', '00:29:00', '00:35:00'])  # 57, 58, 59  #### change these times to UTC

    lat, lon = [], []
    for t in UTC:  # this stuff from Olivier
        if t[:2] == '00': day = 20
        else: day = 19
        observing_time = Time('2023-05-{} {}'.format(day, t)) #time in UTC (example for afternoon)
        coord = SkyCoord(alt=telelv*u.degree, az=azimuth*u.degree, location=ucsc_location,
        obstime=observing_time, frame='altaz')
        gal = coord.transform_to("galactic")
        if p:
            print("Galactic coordinates:", gal)
            print('------------------------')

        lat.append(gal.l.degree)
        lon.append(gal.b.degree)
    return lat, lon

if __name__ == "__main__":
    makeFiles()
    # RADEC('am')  # morning coordinates
    # RADEC('pm')  # evening Coordinates  # 45 deg w/ galactic plane can explain why it still goes up
    fits('am')  # morning plots
    # fits('pm')  # evening plots
    # skymap()
