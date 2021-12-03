"""
CMod Ex2: velocity Verlet time integration of
a particle moving in a double well potential.

Produces plots of the position of the particle
and its energy, both as function of time. Also
saves both to file.

The potential is U(r1, r2) = d_e * ((1 - exp(-alpha(r12 - r_e))) ** 2) - 1)
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
    Force is given by
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
    of particle in double-well potential using Morse Potential
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
    and then obtain the wave-number.

    wave-number = 1 / wavelength

    :param pos_list: position list along the time-axis
    :param dt: time-step for simulation

    :return: wave_number

    """

    wave_peaks = find_peaks(pos_list)
    print(wave_peaks[0])
    peak_pos = [wave_peaks[0]]

    first_peak = peak_pos[0]
    last_peak = peak_pos[1]

    period = peak_pos[first_peak : last_peak]
    period_in_seconds = period * 1.018050571e-14

    frequency = 1 / period_in_seconds       # Hz
    speed_of_light = 299792458              # m/s
    wavelength = (speed_of_light / frequency) / 100         # converts to cm

    wave_number = 1 / wavelength
    return wave_number


# Begin main code
def main() :

    with open(sys.argv[1], "r") as infile :

        # Read name of output file from command line
        if len(sys.argv) != 3 :

            print("Wrong number of arguments.")
            print("Usage: " + sys.argv[0] + "<input file>" + "<output file>")
            quit()
        else :
            outfile_name = sys.argv[2]

            # Open output file
            outfile = open(outfile_name, "w")

            line1 = infile.readline()
            line2 = infile.readline()
            line2 = line2.split()

            if len(line2) != 2 :
                print("Wrong number of arguments in line 2 of input data file, i.e. simulation parameters. ")

            else :
                dt = float(line2[0])
                numstep = int(line2[1])

            line3 = infile.readline()
            line4 = infile.readline()
            line4 = line4.split()

            if len(line4) != 3 :
                print("Wrong number of arguments in line 4 of input data file, i.e. Morse potential parameters. ")

            else :
                d_e = float(line4[0])
                r_e = float(line4[1])
                alpha = float(line4[2])

            line5 = infile.readline()
            line5 = line5.split()
            line6 = infile.readline()
            line6 = line6.split()

            if len(line5) and len(line6) != 8 :
                print("Wrong number of arguments in line 5 and 6, i.e. initial conditions of particles. ")

            else :

                pos1 = np.array([float(line5[2]), float(line5[3]), float(line5[4])])
                vel1 = np.array([float(line5[5]), float(line5[6]), float(line5[7])])

                p1 = Particle3D(str(line5[0]), float(line5[1]), pos1, vel1)

                pos2 = np.array([float(line6[2]), float(line6[3]), float(line6[4])])
                vel2 = np.array([float(line6[5]), float(line6[6]), float(line6[7])])

                p2 = Particle3D(str(line6[0]), float(line6[1]), pos2, vel2)

    infile.close()

    time = 0.0
    p1_to_p2 = np.linalg.norm(p2.pos - p1.pos)


    # Write out initial conditions
    energy = p1.kinetic_e() + p2.kinetic_e() + morse_potential(p1, p2, r_e, d_e, alpha)
    outfile.write("{0:f} {1:f} {2:12.8f}\n".format(time, p1_to_p2, energy))

    # Get initial force
    force1 = morse_force(p1, p2, r_e, d_e, alpha)
    force2 = - force1

    # Initialise data lists for plotting later
    time_list = [time]
    pos1_list = [p1.pos]
    pos2_list = [p2.pos]
    pos_list = [np.linalg.norm(p2.pos - p1.pos)]
    energy_list = [energy]

    # Start the time integration loop
    for i in range(numstep) :
        # Update particle position
        p1.update_pos_2nd(dt, force1)
        p2.update_pos_2nd(dt, force2)
        p1_to_p2 = np.linalg.norm(p2.pos - p1.pos)
        
        # Update force
        force1_new = morse_force(p1, p2, r_e, d_e, alpha)
        force2_new = - force1_new
        # Update particle velocity by averaging
        # current and new forces
        p1.update_vel(dt, 0.5*(force1 + force1_new))
        p2.update_vel(dt, 0.5*(force2 + force2_new))
        
        # Re-define force value
        force1 = force1_new
        force2 = force2_new

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

    # Plot particle trajectory to screen
    pyplot.title('Velocity Verlet : Position vs Time')
    pyplot.xlabel('Time : ')
    pyplot.ylabel('Position : ')
    pyplot.plot(time_list, pos1_list)
    pyplot.plot(time_list, pos2_list)
    pyplot.show()

    # Plot particle energy to screen
    pyplot.title('Velocity Verlet : Total Energy vs Time')
    pyplot.xlabel('Time : ')
    pyplot.ylabel('Energy : ')
    pyplot.plot(time_list, energy_list)
    pyplot.show()

    print("Wavenumber : ", oscillations(pos_list, dt), "cm ^ -1 .")

# Execute main method, but only when directly invoked
if __name__ == "__main__" :
    main()
