"""
CMod Ex2: Symplectic Euler time integration of
a particle moving in a double well potential.

Produces plots of the position of the particle
and its energy, both as function of time. Also
saves both to file.

The potential is given by:

U(r1, r2) = d_e * ((1 - exp(-alpha(r12 - r_e))) ** 2) - 1)

where the parameters are read in from a data file
and passed to the functions that
calculate force and potential energy.
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as pyplot
from scipy.signal import find_peaks
from particle3D import Particle3D


def morse_force(particle, different_particle, r_e, d_e, alpha) :
    """
    Method to return the force on a particle
    in a double well potential using Morse Potential.
    Force is given by:

    F(r1, r2) = 2 * alpha * d_e * (1 - exp(-alpha(r12 - r_e))) * exp(-alpha(r12 - r_e)) * r12_hat

    :param particle: Particle3D instance
    :param different_particle: Particle3D instance
    :param r_e: parameter r_e, controls position of the potential minimum
    :param d_e: parameter d_e, controls curvature of the potential minimum
    :param alpha: parameter alpha, controls depth of the potential minimum

    :return: force acting on particle as Numpy array
    """
    r12 = different_particle.pos - particle.pos
    mag_r12 = np.linalg.norm(r12)
    r12_hat = r12 / mag_r12
    force = 2 * alpha * d_e * (1 - math.exp(-alpha * (mag_r12 - r_e))) * math.exp(-alpha * (mag_r12 - r_e)) * r12_hat
    return force


def morse_potential(particle, different_particle, r_e, d_e, alpha) :
    """
    Method to return Morse Potential
    of particle in double-well potential using Morse Potential:

    U(r1, r2) = d_e * ((1 - exp(-alpha(r12 - r_e))) ** 2) - 1)

    :param particle: Particle3D instance
    :param different_particle: Particle3D instance
    :param r_e: parameter r_e, controls position of the potential minimum
    :param d_e: parameter d_e, controls curvature of the potential minimum
    :param alpha: parameter alpha, controls depth of the potential minimum

    :return: Morse Potential of particle as float
    """
    r12 = different_particle.pos - particle.pos
    mag_r12 = np.linalg.norm(r12)
    potential = d_e * (((1 - math.exp(-alpha * (mag_r12 - r_e))) ** 2) - 1)
    return potential

def oscillations(pos_list, dt) :
    """
    Method to calculate the period of the oscillating particle
    and then obtain the wave-number given by:

    wave-number = 1 / wavelength

    :param pos_list: position list along the time-axis
    :param dt: time-step for simulation

    :return: wave_number
    """
    wave_peaks = find_peaks(pos_list)                           # Uses scipy.signal find_peaks function to index all peaks
    peak_pos = wave_peaks[0]                                    # Creates an array of the indices of all peaks

    first_peak = peak_pos[0]                                    # First peak
    second_peak = peak_pos[1]                                   # Second peak

    period = (len(pos_list[first_peak : second_peak])) * dt     # Calculates period
    period_in_seconds = period * 1.018050571e-14                # Converts period to seconds

    frequency = 1 / period_in_seconds                           # Units : Hz
    speed_of_light = 299792458                                  # Units : m/s
    wavelength = (speed_of_light / frequency) * 100             # Units : m (multiply by 100 to convert to cm) -> cm

    wave_number = 1 / wavelength
    return wave_number                                          # Units : cm^-1

# Begin main code
def main() :
    """
    The main method carries out the simulation in a few parts:

    1.) Reads in data file from the command line
    2.) Specifies initial conditions
    3.) Initialises data lists for plotting later
    4.) Starts a time integration loop
    5.) Plots particle trajectory to screen
    6.) Plots particle energy to screen
    7.) Measures the frequency of oscillations and prints the wave-number and it's inaccuracy to screen
    8.) Measures the energy inaccuracy of the simulation and prints it to the screen

    """

    with open(sys.argv[1], "r") as infile :

        # Part 1.) Reads in data file from the command line

        # Read name of files from command line which needs 3 parts
        if len(sys.argv) != 3 :

            # Helpful error message if the format is incorrect
            print("Wrong number of arguments.")
            print("Usage: " + sys.argv[0] + "<input file>" + "<output file>")
            quit()
        else :
            outfile_name = sys.argv[2]

            # Open output file
            outfile = open(outfile_name, "w")

            line1 = infile.readline()           # Processes line 1 of input file
            line2 = infile.readline()           # Processes line 2 of input file
            line2 = line2.split()               # Separates the parameters in line 2

            # Helpful error message if there is not 2 parameters in line 2
            if len(line2) != 2 :
                print("Wrong number of arguments in line 2 of input data file, i.e. simulation parameters. ")

            else :
                dt = float(line2[0])            # Reads in time-step for simulation
                numstep = int(line2[1])         # Reads in number of steps for simulation

            line3 = infile.readline()           # Processes line 3 of input file
            line4 = infile.readline()           # Processes line 4 of input file
            line4 = line4.split()               # Separates the parameters in line 4

            # Helpful error message if there is not 3 parameters in line 4
            if len(line4) != 3 :
                print("Wrong number of arguments in line 4 of input data file, i.e. Morse potential parameters. ")

            else :
                d_e = float(line4[0])           # Reads in first Morse potential parameter
                r_e = float(line4[1])           # Reads in second Morse potential parameter
                alpha = float(line4[2])         # Reads in third Morse potential parameter

            line5 = infile.readline()           # Processes line 5 of input file
            line5 = line5.split()               # Separates the parameters in line 5
            line6 = infile.readline()           # Processes line 6 of input file
            line6 = line6.split()               # Separates the parameters in line 6


            # Helpful error message if there is not 8 parameters in lines 5 and 6
            if len(line5) and len(line6) != 8 :
                print("Wrong number of arguments in line 5 and 6, i.e. initial conditions of particles. ")

            else :

                pos1 = np.array([float(line5[2]), float(line5[3]), float(line5[4])])    # Sets up Particle 1 position
                vel1 = np.array([float(line5[5]), float(line5[6]), float(line5[7])])    # Sets up Particle 1 velocity

                p1 = Particle3D(str(line5[0]), float(line5[1]), pos1, vel1)             # Sets up Particle 1 as a particle3D instance

                pos2 = np.array([float(line6[2]), float(line6[3]), float(line6[4])])    # Sets up Particle 2 position
                vel2 = np.array([float(line6[5]), float(line6[6]), float(line6[7])])    # Sets up Particle 2 velocity

                p2 = Particle3D(str(line6[0]), float(line6[1]), pos2, vel2)             # Sets up Particle 2 as a particle3D instance

    infile.close()

    # Part 2.) Specifies initial conditions

    time = 0.0
    p1_to_p2 = np.linalg.norm(p2.pos - p1.pos)
    energy = p1.kinetic_e() + p2.kinetic_e() + morse_potential(p1, p2, r_e, d_e, alpha)     # Sums up initial energy total of system
    outfile.write("{0:f} {1:f} {2:12.8f}\n".format(time, p1_to_p2, energy))                 # Formats output file being written

    # Part 3.) Initialises data lists for plotting later

    time_list = [time]
    pos1_list = [p1.pos]
    pos2_list = [p2.pos]
    pos_list = [np.linalg.norm(p2.pos - p1.pos)]            # Position list is | r2 - r1 | from particle positions
    energy_list = [energy]

    # Part 4.) Starts a time integration loop

    for i in range(numstep) :

        # Update particle position
        p1.update_pos(dt)
        p2.update_pos(dt)
        p1_to_p2 = np.linalg.norm(p2.pos - p1.pos)

        # Calculate Morse force
        force1 = morse_force(p1, p2, r_e, d_e, alpha)
        force2 = - force1

        # Update particle velocity 
        p1.update_vel(dt, force1)
        p2.update_vel(dt, force2)

        # Increase time
        time += dt

        # Output particle information
        energy = p1.kinetic_e() + p2.kinetic_e() + morse_potential(p1, p2, r_e, d_e, alpha)
        outfile.write("{0:f} {1:f} {2:12.8f}\n".format(time, p1_to_p2, energy))

        # Append information to data lists
        time_list.append(time)
        pos1_list.append(p1.pos)
        pos2_list.append(p2.pos)
        pos_list.append(p1_to_p2)
        energy_list.append(energy)

    # Post-simulation:
    # Close output file
    outfile.close()

    # Part 5.) Plots particle trajectory to screen

    pyplot.title('Symplectic Euler : Position vs Time')
    pyplot.xlabel('Time (* 1.018050571e-14 s) : ')
    pyplot.ylabel('Position (Angstroms) : ')
    pyplot.plot(time_list, pos1_list)
    pyplot.plot(time_list, pos2_list)
    pyplot.show()

    # Part 6.) Plots particle energy to screen

    pyplot.title('Symplectic Euler : Total Energy vs Time')
    pyplot.xlabel('Time (* 1.018050571e-14 s) : ')
    pyplot.ylabel('Energy (eV) : ')
    pyplot.plot(time_list, energy_list)
    pyplot.show()

    # Part 7.) Measures the frequency of oscillations and prints the wave-number and it's inaccuracy to screen

    wave_number = oscillations(pos_list, dt)
    print("Wave-number :", wave_number, "cm ^ -1 .")
    v_nought = 1525.5847124925656
    delta_v = wave_number - v_nought
    wave_number_inaccuracy = delta_v / v_nought
    print("Wave-number inaccuracy : +/-", wave_number_inaccuracy, "cm ^ -1 .")

    # Part 8.) Measures the energy inaccuracy of the simulation and prints it to the screen

    initial_energy = p1.kinetic_e() + p2.kinetic_e() + morse_potential(p1, p2, r_e, d_e, alpha)
    max_energy = max(energy_list)
    min_energy = min(energy_list)

    delta_energy = max_energy - min_energy
    energy_inaccuracy = delta_energy / initial_energy

    print("Energy inaccuracy : +/-", energy_inaccuracy, "eV ")


# Execute main method, but only when directly invoked
if __name__ == "__main__" :
    main()
