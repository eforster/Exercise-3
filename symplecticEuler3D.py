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


def morse_force(particle, different_particle, r_e, d_e, alpha):
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


def morse_potential(particle, different_particle, r_e, d_e, alpha):
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
    numstep = 10000
    time = 0.0
    r_e = 1.20752
    d_e = 5.21322
    alpha = 2.65374

    # Set up particle initial conditions:
    #  position x0 = 0.0
    #  velocity v0 = 1.0
    #  mass      m = 1.0
    p1 = Particle3D('Oxygen', 16, np.array([0.65661, 0, 0]), np.array([0.05, 0, 0]))
    p2 = Particle3D('Oxygen', 16, np.array([-0.65661, 0, 0]), np.array([-0.05, 0, 0]))
    p1_to_p2 = np.linalg.norm(p2.pos - p1.pos)

    # Write out initial conditions
    energy = p1.kinetic_e() + p2.kinetic_e() + morse_potential(p1, p2, r_e, d_e, alpha)
    outfile.write("{0:f} {1:f} {2:12.8f}\n".format(time, p1_to_p2, energy))

    # Initialise data lists for plotting later
    time_list = [time]
    pos1_list = [p1.pos]
    pos2_list = [p2.pos]
    pos_list = [np.linalg.norm(p2.pos - p1.pos)]
    energy_list = [energy]

    # Start the time integration loop
    for i in range(numstep):
        # Update particle position
        p1.update_pos(dt)
        p2.update_pos(dt)
        p1_to_p2 = np.linalg.norm(p2.pos - p1.pos)

        # Calculate force
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
        energy_list.append(energy)

    # Post-simulation:
    # Close output file
    outfile.close()

    # Plot particle trajectory to screen
    pyplot.title('Symplectic Euler: Position vs Time')
    pyplot.xlabel('Time')
    pyplot.ylabel('Position')
    pyplot.plot(time_list, pos1_list)
    pyplot.plot(time_list, pos2_list)
    pyplot.show()

    # Plot particle energy to screen
    pyplot.title('Symplectic Euler: Total Energy vs Time')
    pyplot.xlabel('Time')
    pyplot.ylabel('Energy')
    pyplot.plot(time_list, energy_list)
    pyplot.show()


# Execute main method, but only when directly invoked
if __name__ == "__main__":
    main()
