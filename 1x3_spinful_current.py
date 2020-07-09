
# Prerequisites
from __future__ import division, print_function
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np


import qmeq
import random
from sympy import *

# Quantum dot parameters


cL1 = 1.6
cR1 = 0.3
cm12 = 1.0
cm13 = 0.5
cU1 = 0.4
cD1 = 0.4

cL2 = 0.8
cR2 = 0.8
cm21 = 1.0
cm23 = 1.0
cU2 = 0.4
cD2 = 0.4

cL3 = 0.3
cR3 = 1.6
cm31 = 0.5
cm32 = 1.0
cU3 = 0.4
cD3 = 0.4


q10 = 0.0
q20 = 0.0
q30 = 0.0

c11 = cL1 + cR1 + cm12 + cm13 + cU1 + cD1
c22 = cL2 + cR2 + cm23 + cm21 + cU2 + cD2
c33 = cL3 + cR3 + cm32 + cm31 + cU3 + cD3

vGU, vGD, vL, vR = symbols("vGU, vGD, vL, vR")
n1, n2, n3 = symbols("n1, n2, n3")

q1 = (cL1*vL+cR1*vR+cU1*vGU+cD1*vGD)*6.24/1000+q10
q2 = (cL2*vL+cR2*vR+cU2*vGU+cD2*vGD)*6.24/1000+q20
q3 = (cL3*vL+cR3*vR+cU3*vGU+cD3*vGD)*6.24/1000+q30

C = Matrix([[c11, -cm12, -cm13],
            [-cm21, c22, -cm23],
            [-cm31, -cm32, c33]])*6.24

Q = Matrix([[q1 - n1],
            [q2 - n2],
            [q3 - n3]])

U=0.5*((C.inv()*Q).T*Q)*1000

U0x0 = U.subs([(n1,0),(n2,0),(n3,0)])[0]
U1x1 = U.subs([(n1,1),(n2,0),(n3,0)])[0]
U2x1 = U.subs([(n1,0),(n2,1),(n3,0)])[0]
U3x1 = U.subs([(n1,0),(n2,0),(n3,1)])[0]
U1x2 = U.subs([(n1,2),(n2,0),(n3,0)])[0]
U2x2 = U.subs([(n1,0),(n2,2),(n3,0)])[0]
U3x2 = U.subs([(n1,0),(n2,0),(n3,2)])[0]

mu1 = U1x1 - U0x0
mu2 = U2x1 - U0x0
mu3 = U3x1 - U0x0

U12x1x1 = U.subs([(n1,1),(n2,1),(n3,0)])[0]
U23x1x1 = U.subs([(n1,0),(n2,1),(n3,1)])[0]
U13x1x1 = U.subs([(n1,1),(n2,0),(n3,1)])[0]

Um12 = U12x1x1 - U0x0 - mu1 - mu2
Um23 = U23x1x1 - U0x0 - mu2 - mu3
Um13 = U13x1x1 - U0x0 - mu1 - mu3

U11 = U1x2 - U1x1 - mu1
U22 = U2x2 - U2x1 - mu2
U33 = U3x2 - U3x1 - mu3


omegapres, omegaflip = 1.0, 0.0
vgateup, vgatedown, vbiasL, vbiasR = 0.0, 0.0, 8.0, -8.0

V0=20 # unit meV*nm
a=11  # unit: nm
Ev1=-V0*(1/a+0.5/a)
Ev2=-V0*(1/a+1/a)
Ev3=-V0*(1/a+0.5/a)

Je = 0.00
Jp = 0.00
Jt1 = 0.00
Jt2 = 0.00


Eq1 = 7.0
Eq2 = -4.0
Eq3 = 3.0

# Lead parameters
temp = 0.5
dband = 1200
# Tunneling amplitudes
gam = 0.005
t0 = np.sqrt(gam/(2*np.pi))
t00 = 0.0*t0




nsingle = 6
nstate = 2**nsingle

hsingle =  {(0,0): Eq1+Ev1+mu1.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (1,1): Eq1+Ev1+mu1.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (2,2): Eq2+Ev2+mu2.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (3,3): Eq2+Ev2+mu2.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (4,4): Eq3+Ev3+mu3.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (5,5): Eq3+Ev3+mu3.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (0,2): -omegapres,
            (0,4): -omegapres/5,
            (2,4): -omegapres,
            (1,3): -omegapres,
            (1,5): -omegapres/5,
            (3,5): -omegapres
}

# 0 is up, 1 is down


coulomb = {(0,1,1,0):U11.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (2,3,3,2):U22.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (4,5,5,4):U33.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,2,2,0):Um12.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,3,3,0):Um12.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,2,2,1):Um12.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,3,3,1):Um12.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,4,4,0):Um13.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,5,5,0):Um13.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,4,4,1):Um13.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,5,5,1):Um13.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (2,4,4,2):Um23.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (2,5,5,2):Um23.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (3,4,4,3):Um23.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (3,5,5,3):Um23.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)])
}



tleads = {(0, 0):-t0, # L, up   <-- up
          (1, 4):-t0, # R, up   <-- up
          (2, 1):-t0, # L, down <-- down
          (3, 5):-t0}




nleads = 4



#        L,up        R,up         L,down      R,down
mulst = {0: -vbiasL, 1: -vbiasR, 2: -vbiasL, 3: -vbiasR}
tlst =  {0: temp,    1: temp,     2: temp,    3: temp}



system = qmeq.Builder(nsingle, hsingle, coulomb,
                      nleads, tleads, mulst, tlst, dband,
                      kerntype='Lindblad')


# Here we have chosen to use **Pauli master equation** (*kerntype='Pauli'*) to describe the stationary state. Let's calculate the current through the system:



system.solve()

print("System indexing: ")
print(system.indexing)
print("Dots Eigenenergies: ")
print(system.Ea)

##################################  density matrix in eigenstate basis  #######################################

dm_eigen = np.zeros((nstate,nstate),dtype=np.complex_)
for i in range(0, nstate):
  for j in range(0, nstate):
    dm_eigen[i,j] = system.get_phi0(i,j)
print("Reduced density matrix: ")
print(dm_eigen)

##################################  eigenstates in fock basis full 1x16 array  #################################

def fock_coeff():
  result = np.zeros((nstate,nstate),dtype=np.complex_)
  for i in range(0, nstate):
    temp = system.coeff_state(i)
    for j in range(0, len(temp)):
      result[i,j] = temp[j]
  return result

eigenstate_in_fock = fock_coeff()
eig_T = np.transpose(eigenstate_in_fock)
eig_C = np.conjugate(eigenstate_in_fock)
print("Eigenstates in fock basis: ")
print(eigenstate_in_fock)

#################################  density matrix in fock basis  #########################################

dm_fock = np.matmul(eig_T, np.matmul(dm_eigen, eig_C))  # transform from eigenstate basis to fock basis
print("Dot density matrix in fock basis: ")
print(dm_fock)


for i in range(0,nstate):
  system.print_state(i)

#################################  construct the current operater here |1d> to |2d>  ############################
current_1dx2d = np.zeros((nstate,nstate),dtype=np.complex_)
current_1dx2d[1,2] = 1
current_1dx2d[2,1] = -1
current_1dx2d[5,6] = 1
current_1dx2d[6,5] = -1

current_1dx3d = np.zeros((nstate,nstate),dtype=np.complex_)
current_1dx3d[1,4] = 1
current_1dx3d[4,1] = -1
current_1dx3d[3,6] = 1
current_1dx3d[6,3] = -1

current_2dx3d = np.zeros((nstate,nstate),dtype=np.complex_)
current_2dx3d[2,4] = 1
current_2dx3d[4,2] = -1
current_2dx3d[3,5] = 1
current_2dx3d[5,3] = -1

#print(current_1dx2d)


################################ The current is the trace of product dm_fock and current_1dx2d  ##############################
#product = np.matmul(dm_fock,current_1dx2d)
#print("Current 1d->2d is: ")
#print(np.trace(product))





for i in range(0,4):
  print("#####################")
  


def stab_calc(system, bfield, vlst, vglst, dV=0.0001):
    vpnt, vgpnt = vlst.shape[0], vglst.shape[0]

    stab = np.zeros((vpnt, vgpnt))
    stab_cond = np.zeros((vpnt, vgpnt))
    print(vpnt)
    for j1 in range(vgpnt):
        vgateup = vglst[j1]
        print(j1)


        for j2 in range(vpnt):
            vgatedown = vlst[j2]

            system.change(hsingle =  {(0,0): Eq1+Ev1+mu1.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (1,1): Eq1+Ev1+mu1.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (2,2): Eq2+Ev2+mu2.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (3,3): Eq2+Ev2+mu2.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (4,4): Eq3+Ev3+mu3.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (5,5): Eq3+Ev3+mu3.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (0,2): -omegapres,
            (0,4): -omegapres/5,
            (2,4): -omegapres,
            (1,3): -omegapres,
            (1,5): -omegapres/5,
            (3,5): -omegapres})

            system.change(coulomb = {(0,1,1,0):U11.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (2,3,3,2):U22.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (4,5,5,4):U33.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,2,2,0):Um12.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,3,3,0):Um12.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,2,2,1):Um12.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,3,3,1):Um12.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,4,4,0):Um13.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,5,5,0):Um13.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,4,4,1):Um13.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,5,5,1):Um13.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (2,4,4,2):Um23.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (2,5,5,2):Um23.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (3,4,4,3):Um23.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (3,5,5,3):Um23.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)])})

            system.solve(masterq=False)

            #################  ploting the expectation value of 1d->2d current in gate-gate map  ########################  

            system.solve(qdq=False)
            #for i in range(0, nstate):
            #  for j in range(0, nstate):
            #    dm_eigen[i,j] = system.get_phi0(i,j)

            #eigenstate_in_fock = fock_coeff()
            #eig_T = np.transpose(eigenstate_in_fock)
            #eig_C = np.conjugate(eigenstate_in_fock)
            #dm_fock = np.matmul(eig_T, np.matmul(dm_eigen, eig_C))
            #product = np.matmul(dm_fock,current_1dx2d)

            #stab[j1, j2] = abs(np.trace(product))
            #temp = 0
            #mini = 0
            #for j in range(0,nstate):
            #  if (system.Ea[j] < mini):
            #    mini = system.Ea[j]
            #    temp = j
            #result = 0
            #if temp == 0:
            #  result = 0
            #elif temp > 0 and temp < 4:
            #  result = 1
            #elif temp > 3 and temp < 7:
            #  result = 2
            #else:
            #  result = 3
            #stab[j1, j2] = result
            #for i in range(0, nstate):
            #  for j in range(0, nstate):
            #    dm_eigen[i,j] = system.get_phi0(i,j)

            #eigenstate_in_fock = fock_coeff()
            #eig_T = np.transpose(eigenstate_in_fock)
            #eig_C = np.conjugate(eigenstate_in_fock)
            #dm_fock = np.matmul(eig_T, np.matmul(dm_eigen, eig_C))
            #product = np.matmul(dm_fock,(current_2dx3d + current_1dx3d))
            #stab[j1, j2] = abs(np.trace(product))
            stab[j1, j2] = system.current[0] + system.current[2]
            stab[j1, j2] = abs(stab[j1, j2])
    return stab, stab_cond

# We changed the single particle Hamiltonian by calling the function **system.change** and specifying which matrix elements to change. The function **system.add** adds a value to a specified parameter. Also the option *masterq=False* in **system.solve** indicates just to diagonalise the quantum dot Hamiltonian and the master equation is not solved. Similarly, the option *qdq=False* means that the quantum dot Hamiltonian is not diagonalized (it was already diagonalized previously) and just master equation is solved.


#system.kerntype = 'Lindblad'
vpnt, vgpnt = 101, 101
vlst = np.linspace(-200, 600, vpnt)
vglst = np.linspace(-200, 600, vgpnt)
stab, stab_cond = stab_calc(system, 0, vlst, vglst)


# The stability diagram has been produced. Let's see how it looks like:


def stab_plot(stab, stab_cond, vlst, vglst, gam):
    (xmin, xmax, ymin, ymax) = np.array([vglst[0], vglst[-1], vlst[0], vlst[-1]])
    fig = plt.figure(figsize=(8,6))
    #
    p1 = plt.subplot(1, 1, 1)
    p1.set_xlabel('$V_{g1}(mV)$', fontsize=20)
    p1.set_ylabel('$V_{g2}(mV)$', fontsize=20)
    p1_im = plt.imshow(stab.T, extent=[xmin, xmax, ymin, ymax], aspect='auto', origin='lower', cmap = plt.get_cmap('Spectral'))
    cbar1 = plt.colorbar(p1_im)
    cbar1.set_label('Current [unit]', fontsize=20)

    plt.tight_layout()
    plt.show()

stab_plot(stab, stab_cond, vlst, vglst, gam)