import sympy as sp
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
import scipy as scpy


from sympy.physics.vector import init_vprinting
Axes3D = Axes3D

init_vprinting(use_latex='mathjax', pretty_print=False)


a, alpha, d, theta, theta1, theta2, theta3, theta4, theta5, theta6, l1, a2, a3, l4, l5, l6 = sp.symbols(
    'a alpha d theta theta1 theta2 theta3 theta4 theta5 theta6 l1 a2 a3, l4, l5 l6')
# Helper functions


def scos(x): return sp.cos(x).evalf()
def ssin(x): return sp.sin(x).evalf()



# Cross product function


def cross(A, B):
    return [A[1]*B[2] - A[2]*B[1], A[2]*B[0] - A[0]*B[2], A[0]*B[1] - A[1]*B[0]]

# DH Transformation


def dh_trans(q):

    # Constant D-H parameters
    l1 = 128
    a2 = 612.7
    a3 = 571.6
    l4 = 163.9
    l5 = 115.7
    l6 = 192.2

    theta1 = q[0]
    theta2 = q[1]
    theta3 = q[2]
    theta4 = q[3]
    theta5 = q[4]
    theta6 = q[5]
    rot = sp.Matrix([[scos(theta), -ssin(theta)*scos(alpha), ssin(theta)*ssin(alpha)],
                     [ssin(theta), scos(theta)*scos(alpha), -
                      scos(theta)*ssin(alpha)],
                     [0, ssin(alpha), scos(alpha)]])
    trans = sp.Matrix([a*scos(theta), a*ssin(theta), d])
    last_row = sp.Matrix([[0, 0, 0, 1]])
    t = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), last_row)

    t01 = t.subs({a: 0, alpha: -sp.pi/2, d: l1, theta: theta1})
    t12 = t.subs({a: a2, alpha: sp.pi, d: 0, theta: theta2-sp.pi/2})
    t23 = t.subs({a: a3, alpha: sp.pi, d: 0, theta: theta3})
    t34 = t.subs({a: 0, alpha: sp.pi/2, d: l4, theta: theta4+sp.pi/2})
    t45 = t.subs({a: 0, alpha: -sp.pi/2, d: l5, theta: theta5})
    t56 = t.subs({a: 0, alpha: 0, d: l6, theta: theta6})

    t06 = t01 * t12 * t23 * t34 * t45 * t56
    return t06

# DH for Jacobian


def dh_for_jacobian(q):
    # Constant D-H parameters
    l1 = 128
    a2 = 612.7
    a3 = 571.6
    l4 = 163.9
    l5 = 115.7
    l6 = 192.2

    theta1 = q[0]
    theta2 = q[1]
    theta3 = q[2]
    theta4 = q[3]
    theta5 = q[4]
    theta6 = q[5]

    rot = sp.Matrix([[scos(theta), -ssin(theta)*scos(alpha), ssin(theta)*ssin(alpha)],
                     [ssin(theta), scos(theta)*scos(alpha), -
                      scos(theta)*ssin(alpha)],
                     [0, ssin(alpha), scos(alpha)]])
    trans = sp.Matrix([a*scos(theta), a*ssin(theta), d])
    last_row = sp.Matrix([[0, 0, 0, 1]])
    t = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), last_row)

    t01 = t.subs({a: 0, alpha: -sp.pi/2, d: l1, theta: theta1})
    t12 = t.subs({a: a2, alpha: sp.pi, d: 0, theta: theta2-sp.pi/2})
    t23 = t.subs({a: a3, alpha: sp.pi, d: 0, theta: theta3})
    t34 = t.subs({a: 0, alpha: sp.pi/2, d: l4, theta: theta4+sp.pi/2})
    t45 = t.subs({a: 0, alpha: -sp.pi/2, d: l5, theta: theta5})
    t56 = t.subs({a: 0, alpha: 0, d: l6, theta: theta6})

    T = [t01, t01*t12, t01*t12*t23, t01*t12*t23*t34,
         t01*t12*t23*t34*t45, t01*t12*t23*t34*t45*t56]
    return T

# Jacobian calculation


def jacobian(T):

    # Z vectors
    z = [sp.Matrix([0, 0, 1])]
    for i in T:
        z.append((i[:3, 2]))

    # Origins
    o = [sp.Matrix([0, 0, 0])]
    for i in T:
        o.append((i[:3, 3]))

    # Build the Jacobian matrix
    J = sp.zeros(6, 6)

    # The first three rows of the Jacobian are the cross product of z vectors and difference of end-effector and joint origins
    for i in range(6):
        J[0, i] = sp.Matrix(
            cross(z[i], [o[-1][0] - o[i][0], o[-1][1] - o[i][1], o[-1][2] - o[i][2]]))

    # The last three rows of the Jacobian are simply the z vectors for rotational joints
    for i in range(6):
        J[3:6, i] = z[i]
        # sp.pprint(J)
    return J


def print_J_O_Z_vectors():

    # Constant D-H parameters
    l1 = 128
    a2 = 612.7
    a3 = 571.6
    l4 = 163.9
    l5 = 115.7
    l6 = 192.2

    rot = sp.Matrix([[scos(theta), -ssin(theta)*scos(alpha), ssin(theta)*ssin(alpha)],
                     [ssin(theta), scos(theta)*scos(alpha), -
                      scos(theta)*ssin(alpha)],
                     [0, ssin(alpha), scos(alpha)]])
    trans = sp.Matrix([a*scos(theta), a*ssin(theta), d])
    last_row = sp.Matrix([[0, 0, 0, 1]])
    t = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), last_row)

    t01 = t.subs({a: 0, alpha: -sp.pi/2, d: l1, theta: theta1})
    t12 = t.subs({a: a2, alpha: sp.pi, d: 0, theta: theta2-sp.pi/2})
    t23 = t.subs({a: a3, alpha: sp.pi, d: 0, theta: theta3})
    t34 = t.subs({a: 0, alpha: sp.pi/2, d: l4, theta: theta4+sp.pi/2})
    t45 = t.subs({a: 0, alpha: -sp.pi/2, d: l5, theta: theta5})
    t56 = t.subs({a: 0, alpha: 0, d: l6, theta: theta6})

    T = [t01, t01*t12, t01*t12*t23, t01*t12*t23*t34,
         t01*t12*t23*t34*t45, t01*t12*t23*t34*t45*t56]

  # Z vectors
    z = [sp.Matrix([0, 0, 1])]
    for i in T:
        z.append((i[:3, 2]))

    print("****************************************************************************************************")
    print("THE Z-AXES UNIT VECTORS OF LOCAL FRAMES WITH RESPECT TO BASE FRAME: ")
    print("z0:")
    sp.pprint(z[0])

    print("z1:")
    sp.pprint(z[1])

    print("z2:")
    sp.pprint(z[2])

    print("z3:")
    sp.pprint(z[3])

    print("z4:")
    sp.pprint(z[4])

    print("z5:")
    sp.pprint(z[5])

    print("z6:")
    sp.pprint(z[6])

    # Origins
    o = [sp.Matrix([0, 0, 0])]
    for i in T:
        o.append((i[:3, 3]))
    print("****************************************************************************************************")
    print("THE ORIGIN VECTORS ARE: ")
    print("O0:")
    sp.pprint(o[0])

    print("O1:")
    sp.pprint(o[1])

    print("O2:")
    sp.pprint(o[2])

    print("O3:")
    sp.pprint(o[3])

    print("O4:")
    sp.pprint(o[4])

    print("O5:")
    sp.pprint(o[5])

    print("O6:")
    sp.pprint(o[6])

    # Build the Jacobian matrix
    J = sp.zeros(6, 6)

    # The first three rows of the Jacobian are the cross product of z vectors and difference of end-effector and joint origins
    for i in range(6):
        J[0, i] = sp.Matrix(
            cross(z[i], [o[-1][0] - o[i][0], o[-1][1] - o[i][1], o[-1][2] - o[i][2]]))

    # The last three rows of the Jacobian are simply the z vect5ors for rotational joints
    for i in range(6):
        J[3:6, i] = z[i]
        # sp.pprint(J)
    print("GENERIC JACOBIAN MATRIX: ")
    sp.pprint(J)
    print("--------------------------------------------------------------------------------------------------------------")


def print_DH_MATRIX():
    rot = sp.Matrix([[scos(theta), -ssin(theta)*scos(alpha), ssin(theta)*ssin(alpha)],
                     [ssin(theta), scos(theta)*scos(alpha), -
                      scos(theta)*ssin(alpha)],
                     [0, ssin(alpha), scos(alpha)]])
    trans = sp.Matrix([a*scos(theta), a*ssin(theta), d])
    last_row = sp.Matrix([[0, 0, 0, 1]])
    t = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), last_row)

    t01 = t.subs({a: 0, alpha: -sp.pi/2, d: l1, theta: theta1})
    t12 = t.subs({a: a2, alpha: sp.pi, d: 0, theta: theta2-sp.pi/2})
    t23 = t.subs({a: a3, alpha: sp.pi, d: 0, theta: theta3})
    t34 = t.subs({a: 0, alpha: sp.pi/2, d: l4, theta: theta4+sp.pi/2})
    t45 = t.subs({a: 0, alpha: -sp.pi/2, d: l5, theta: theta5})
    t56 = t.subs({a: 0, alpha: 0, d: l6, theta: theta6})

    t06 = t01 * t12 * t23 * t34 * t45 * t56

    print("***********************************************************************************************************")
    print(" Homogenous Transformation Matrix T_06: ")
    sp.pprint(t06)
def print_updated_trajectory():
    x_dot = -sp.N(100*sp.pi/100*sp.sin(sp.N((sp.pi*theta)/100)+sp.pi/2))
    y_dot = 0.0
    z_dot = sp.N(100*sp.pi/100*sp.cos(sp.N((sp.pi*theta)/100)+sp.pi/2))

    epsilon = sp.Matrix([x_dot, y_dot, z_dot, 0, 0, 0])
    print("THE UPDATED CIRCULAR TRAJECTORY EQUATIONS: ")
    sp.pprint(epsilon)
    print('-------------------------------------------------------------------------------------------------------------')

def print_generic_gravity_matrix():
  l1 = 128
  a2 = 612.7
  a3 = 571.6
  l4 = 163.9
  l5 = 115.7
  l6 = 192.2

  m1 = 7.1
  m2 = 12.7
  m3 = 4.27
  m4 = 2
  m5 = 2
  m6 = 0.365

  h1 = l1/2
  h2 = l1 + (a2*ssin(theta2-sp.pi/2))/2
  h3 = l1 + (a2*ssin(theta2-sp.pi/2)) + (a3*ssin(theta2-sp.pi/2+theta3))/2
  h4 = l1 + (a2*ssin(theta2-sp.pi/2)) + (a3*ssin(theta2-sp.pi/2+theta3))
  h5 = l1 + (a2*ssin(theta2-sp.pi/2)) + (a3*ssin(theta2-sp.pi/2+theta3)) + (l5*ssin(theta2-sp.pi/2+theta3+theta4+sp.pi/2))/2
  h6 = l1 + (a2*ssin(theta2-sp.pi/2)) + (a3*ssin(theta2-sp.pi/2+theta3)) + (l5*ssin(theta2-sp.pi/2+theta3+theta4+sp.pi/2))

  p = 9.8*((m1*h1)+(m2*h2)+(m3*h3)+(m4*h4)+(m5*h5)+(m6*h6))

  g = sp.Matrix([0, 0, 0, 0, 0, 0])

  g[0] = sp.diff(p,theta1)
  g[1] = sp.diff(p,theta2)
  g[2] = sp.diff(p,theta3)
  g[3] = sp.diff(p,theta4)
  g[4] = sp.diff(p,theta5)
  g[5] = sp.diff(p,theta6)

  print("TOTAL POTENTIAL ENERGY EQUATION: ")
  print('P = ',p)

  print("----------------------------------------------------------------------------------------------------------------")

  print("THE GRAVITY MATRIX: ")
  sp.pprint(g)

def generic_gravity_matrix():
  l1 = 128
  a2 = 612.7
  a3 = 571.6
  l4 = 163.9
  l5 = 115.7
  l6 = 192.2

  # Mass of the LINKS
  m1 = 7.1
  m2 = 12.7
  m3 = 4.27
  m4 = 2
  m5 = 2
  m6 = 0.365



  # Height of Centre of Mass of all the LINKS
  h1 = l1/2
  h2 = l1 + (a2*ssin(theta2-sp.pi/2))/2
  h3 = l1 + (a2*ssin(theta2-sp.pi/2)) + (a3*ssin(theta2-sp.pi/2+theta3))/2
  h4 = l1 + (a2*ssin(theta2-sp.pi/2)) + (a3*ssin(theta2-sp.pi/2+theta3))
  h5 = l1 + (a2*ssin(theta2-sp.pi/2)) + (a3*ssin(theta2-sp.pi/2+theta3)) + (l5*ssin(theta2-sp.pi/2+theta3+theta4+sp.pi/2))/2
  h6 = l1 + (a2*ssin(theta2-sp.pi/2)) + (a3*ssin(theta2-sp.pi/2+theta3)) + (l5*ssin(theta2-sp.pi/2+theta3+theta4+sp.pi/2))


  #Total Potential Energy
  p = 9.8*((m1*h1)+(m2*h2)+(m3*h3)+(m4*h4)+(m5*h5)+(m6*h6))

  g = sp.Matrix([0, 0, 0, 0, 0, 0])

  g[0] = sp.diff(p,theta1)
  g[1] = sp.diff(p,theta2)
  g[2] = sp.diff(p,theta3)
  g[3] = sp.diff(p,theta4)
  g[4] = sp.diff(p,theta5)
  g[5] = sp.diff(p,theta6)

  return g

def gravity_matrix(q):
    g = generic_gravity_matrix()
    g[0] = g[0].subs(theta1, q[0]).subs(theta2, q[1]).subs(theta3, q[2]).subs(theta4, q[3]).subs(theta5, q[4]).subs(theta6, q[5]).evalf()
    g[1] = g[1].subs(theta1, q[0]).subs(theta2, q[1]).subs(theta3, q[2]).subs(theta4, q[3]).subs(theta5, q[4]).subs(theta6, q[5]).evalf()
    g[2] = g[2].subs(theta1, q[0]).subs(theta2, q[1]).subs(theta3, q[2]).subs(theta4, q[3]).subs(theta5, q[4]).subs(theta6, q[5]).evalf()
    g[3] = g[3].subs(theta1, q[0]).subs(theta2, q[1]).subs(theta3, q[2]).subs(theta4, q[3]).subs(theta5, q[4]).subs(theta6, q[5]).evalf()
    g[4] = g[4].subs(theta1, q[0]).subs(theta2, q[1]).subs(theta3, q[2]).subs(theta4, q[3]).subs(theta5, q[4]).subs(theta6, q[5]).evalf()
    g[5] = g[5].subs(theta1, q[0]).subs(theta2, q[1]).subs(theta3, q[2]).subs(theta4, q[3]).subs(theta5, q[4]).subs(theta6, q[5]).evalf()

    return g


def print_joint_torques(q):

    f = sp.Matrix([0, -5, 0, 0, 0, 0])
    dt = 0.1
    total_time = np.arange(0, 200, dt)

    x_points = list()
    y_points = list()
    z_points = list()

    t_joint1 = list()
    t_joint2 = list()
    t_joint3 = list()
    t_joint4 = list()
    t_joint5 = list()
    t_joint6 = list()
    tau = list()
    for i in total_time:

        x_dot = -sp.N(100*sp.pi/100*sp.sin(sp.N((sp.pi*i)/100)+sp.pi/2))
        y_dot = 0.0
        z_dot = sp.N(100*sp.pi/100*sp.cos(sp.N((sp.pi*i)/100)+sp.pi/2))

        epsilon = sp.Matrix([x_dot, y_dot, z_dot, 0, 0, 0])

        T = dh_for_jacobian(q)
        J = jacobian(T)
        J_trans = J.transpose()

        tau = gravity_matrix(q) - (J_trans*f)

        t_joint1.append(tau[0])
        t_joint2.append(tau[1])
        t_joint3.append(tau[2])
        t_joint4.append(tau[3])
        t_joint5.append(tau[4])
        t_joint6.append(tau[5])

        j_inv = np.linalg.pinv(np.array(J, dtype=float))
        q_dot = j_inv * epsilon
        q = q + q_dot * dt

        T_end_effector = dh_trans(q)

        x_points.append(T_end_effector[0, 3])
        y_points.append(356.1)
        z_points.append(T_end_effector[2, 3])

    plt.rcParams['figure.figsize'] = [20, 10]
    fig, axs = plt.subplots(2,3)
    axs[0,0].plot(total_time, t_joint1)
    axs[0,0].set_title("JOINT 1 TORQUE")


    axs[0,1].plot(total_time, t_joint2)
    axs[0,1].set_title("JOINT 2 TORQUE")

    axs[0,2].plot(total_time, t_joint3)
    axs[0,2].set_title("JOINT 3 TORQUE")

    axs[1,0].plot(total_time, t_joint4)
    axs[1,0].set_title("JOINT 4 TORQUE")

    axs[1,1].plot(total_time, t_joint5)
    axs[1,1].set_title("JOINT 5 TORQUE")

    axs[1,2].plot(total_time, t_joint6)
    axs[1,2].set_title("JOINT 6 TORQUE")
    axs[1,2].set_ylim([-5, 5])
    fig.tight_layout(pad=3.0)

    for ax in axs.flat:
      ax.set(xlabel='TIME', ylabel='TORQUE')

    plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(x_points, y_points, z_points)
    ax.set_title("End-Effector Traced Path (ISOMETRIC VIEW)")
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    plt.show()


# Print Homogenous Transformation Matrix
print_DH_MATRIX()

# Print Origin Vectors, Origin Vectors and Jacobian Matrix
print_J_O_Z_vectors()
print_updated_trajectory()

#Print GENERIC GRAVITY MATRIX
print_generic_gravity_matrix()

# Home Position Of End Effector
q_initial = sp.Matrix([0, 0, 0, 0, 0, 0])

#Plots JOINT TORQUES and PRINTS THE CIRCLE TRACED
print_joint_torques(q_initial)
