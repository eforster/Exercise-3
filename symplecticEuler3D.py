"""
CMod Ex2: Symplectic Euler time integration of
a particle moving in a double well potential.

Produces plots of the position of the particle
and its energy, both as function of time. Also
saves both to file.

The potential is V(x) = a*x^4 - b*x^2, where
a and b are hard-coded in the main() method
and passed to the functions that
calculate force and potential energy.
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as pyplot
from particle3D import Particle3D


def morse_force(particle, different_particle, r_e, alpha, d_e):
    """
    Method to return the force on a particle
    in a double well potential using Morse Potential.
    Force is given by
    F(r1, r2) = 2 * alpha * d_e * (1 - exp(-alpha(r12 - r_e))) * exp(-alpha(r12 - r_e)) * r12_hat

    :param particle: Particle3D instance
    :param different_particle: Particle3D instance
    :param r_e: parameter r_e, controls position of the potential minimum
    :param alpha: parameter alpha, controls depth of the potential minimum
    :param d_e: parameter d_e, controls curvature of the potential minimum
    :return: force acting on particle as Numpy array
    """
    r12 = different_particle.pos - particle.pos
    r12_hat = r12 / np.linalg.norm(r12)
    force = 2 * alpha * d_e * (1 - math.exp(-alpha(r12 - r_e))) * math.exp(-alpha(r12 - r_e)) * r12_hat
    return force


def morse_potential(particle, different_particle, r_e, alpha, d_e):
    """
    Method to return Morse Potential
    of particle in double-well potential using Morse Potential
    U(r1, r2) = d_e * ((1 - exp(-alpha(r12 - r_e))) ** 2) - 1)

    :param particle: Particle3D instance
    :param different_particle: Particle3D instance
    :param r_e: parameter r_e, controls position of the potential minimum
    :param alpha: parameter alpha, controls depth of the potential minimum
    :param d_e: parameter d_e, controls curvature of the potential minimum

    :return: Morse Potential of particle as float
    """
    r12 = different_particle.pos - particle.pos
    potential = d_e * (((1 - math.exp(-alpha(r12 - r_e))) ** 2) - 1)
    return potential


# Begin main code
def main():
    # Read name of output file from command line
    if len(sys.argv) != 2:
        print("Wrong number of arguments.")
        print("Usage: " + sys.argv[0] + " <output file>")
        quit()
    else:
        outfile_name = sys.argv[1]

    # Open output file
    outfile = open(outfile_name, "w")

    # Set up simulation parameters
    dt = 0.01
    numstep = 2000
    time = 0.0
    a = 0.1
    b = 1.0

    # Set up particle initial conditions:
    #  position x0 = 0.0
    #  velocity v0 = 1.0
    #  mass      m = 1.0
    p1 = Particle3D(0.0, 1.0, 1.0)

    # Write out initial conditions
    energy = p1.kinetic_energy() + pot_energy_dw(p1, a, b)
    outfile.write("{0:f} {1:f} {2:12.8f}\n".format(time, p1.position, energy))

    # Initialise data lists for plotting later
    time_list = [time]
    pos_list = [p1.position]
    energy_list = [energy]

    # Start the time integration loop
    for i in range(numstep):
        # Update particle position
        p1.leap_pos1st(dt)

        # Calculate force
        force = force_dw(p1, a, b)
        # Update particle velocity 
        p1.leap_velocity(dt, force)

        # Increase time
        time += dt

        # Output particle information
        energy = p1.kinetic_energy() + pot_energy_dw(p1, a, b)
        outfile.write("{0:f} {1:f} {2:12.8f}\n".format(time, p1.position, energy))

        # Append information to data lists
        time_list.append(time)
        pos_list.append(p1.position)
        energy_list.append(energy)

    # Post-simulation:
    # Close output file
    outfile.close()

    # Plot particle trajectory to screen
    pyplot.title('Symplectic Euler: position vs time')
    pyplot.xlabel('Time')
    pyplot.ylabel('Position')
    pyplot.plot(time_list, pos_list)
    pyplot.show()

    # Plot particle energy to screen
    pyplot.title('Symplectic Euler: total energy vs time')
    pyplot.xlabel('Time')
    pyplot.ylabel('Energy')
    pyplot.plot(time_list, energy_list)
    pyplot.show()


# Execute main method, but only when directly invoked
if __name__ == "__main__":
    main()
