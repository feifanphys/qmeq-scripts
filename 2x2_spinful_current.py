
# Prerequisites
from __future__ import division, print_function
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np


import qmeq
import random
from sympy import *
import time

# Quantum dot parameters


cL1 = 1.6
cR1 = 0.3
cm12 = 1.0
cm13 = 0.7
cm14 = 1.0
cU1 = 0.2
cD1 = 0.6

cL2 = 0.3
cR2 = 1.6
cm21 = 1.0
cm23 = 1.0
cm24 = 0.7
cU2 = 0.2
cD2 = 0.6

cL3 = 0.3
cR3 = 1.6
cm31 = 0.7
cm32 = 1.0
cm34 = 1.0
cU3 = 0.6
cD3 = 0.2

cL4 = 1.6
cR4 = 0.3
cm41 = 1.0
cm42 = 0.7
cm43 = 1.0
cU4 = 0.6
cD4 = 0.2


q10 = 0.0
q20 = 0.0
q30 = 0.0
q40 = 0.0

c11 = cL1 + cR1 + cm12 + cm13 + cm14 + cU1 + cD1
c22 = cL2 + cR2 + cm23 + cm21 + cm24 + cU2 + cD2
c33 = cL3 + cR3 + cm32 + cm31 + cm34 + cU3 + cD3
c44 = cL4 + cR4 + cm41 + cm42 + cm43 + cU4 + cD4

vGU, vGD, vL, vR = symbols("vGU, vGD, vL, vR")
n1, n2, n3, n4 = symbols("n1, n2, n3, n4")

q1 = (cL1*vL+cR1*vR+cU1*vGU+cD1*vGD)*6.24/1000+q10
q2 = (cL2*vL+cR2*vR+cU2*vGU+cD2*vGD)*6.24/1000+q20
q3 = (cL3*vL+cR3*vR+cU3*vGU+cD3*vGD)*6.24/1000+q30
q4 = (cL4*vL+cR4*vR+cU4*vGU+cD4*vGD)*6.24/1000+q30

C = Matrix([[c11, -cm12, -cm13, -cm14],
            [-cm21, c22, -cm23, -cm24],
            [-cm31, -cm32, c33, -cm34],
            [-cm41, -cm42, -cm43, c44]])*6.24

Q = Matrix([[q1 - n1],
            [q2 - n2],
            [q3 - n3],
            [q4 - n4]])

U=0.5*((C.inv()*Q).T*Q)*1000

U0x0 = U.subs([(n1,0),(n2,0),(n3,0),(n4,0)])[0]
U1x1 = U.subs([(n1,1),(n2,0),(n3,0),(n4,0)])[0]
U2x1 = U.subs([(n1,0),(n2,1),(n3,0),(n4,0)])[0]
U3x1 = U.subs([(n1,0),(n2,0),(n3,1),(n4,0)])[0]
U4x1 = U.subs([(n1,0),(n2,0),(n3,0),(n4,1)])[0]
U1x2 = U.subs([(n1,2),(n2,0),(n3,0),(n4,0)])[0]
U2x2 = U.subs([(n1,0),(n2,2),(n3,0),(n4,0)])[0]
U3x2 = U.subs([(n1,0),(n2,0),(n3,2),(n4,0)])[0]
U4x2 = U.subs([(n1,0),(n2,0),(n3,0),(n4,2)])[0]


mu1 = U1x1 - U0x0
mu2 = U2x1 - U0x0
mu3 = U3x1 - U0x0
mu4 = U4x1 - U0x0

U11 = U1x2 - U1x1 - mu1
U22 = U2x2 - U2x1 - mu2
U33 = U3x2 - U3x1 - mu3
U44 = U4x2 - U4x1 - mu4


U12x1x1 = U.subs([(n1,1),(n2,1),(n3,0),(n4,0)])[0]
U23x1x1 = U.subs([(n1,0),(n2,1),(n3,1),(n4,0)])[0]
U13x1x1 = U.subs([(n1,1),(n2,0),(n3,1),(n4,0)])[0]
U14x1x1 = U.subs([(n1,1),(n2,0),(n3,0),(n4,1)])[0]
U24x1x1 = U.subs([(n1,0),(n2,1),(n3,0),(n4,1)])[0]
U34x1x1 = U.subs([(n1,0),(n2,0),(n3,1),(n4,1)])[0]


Um12 = U12x1x1 - U0x0 - mu1 - mu2
Um23 = U23x1x1 - U0x0 - mu2 - mu3
Um13 = U13x1x1 - U0x0 - mu1 - mu3
Um14 = U14x1x1 - U0x0 - mu1 - mu4
Um24 = U24x1x1 - U0x0 - mu2 - mu4
Um34 = U34x1x1 - U0x0 - mu3 - mu4

V0=18 # unit meV*nm
a=11  # unit: nm
b=np.sqrt(a**2+a**2)
Ev1=-V0*(1/a+1/a+1/b)
Ev2=-V0*(1/a+1/a+1/b)
Ev3=-V0*(1/a+1/a+1/b)
Ev4=-V0*(1/a+1/a+1/b)


omegapres, omegaflip = 0.2, 0.0
vgateup, vgatedown, vbiasL, vbiasR = 0.0, 0.0, 5.0, 0.0


Je = 0.00
Jp = 0.00
Jt1 = 0.00
Jt2 = 0.00


Eq1 = 1.0
Eq2 = -1.5
Eq3 = 0.0
Eq4 = 3.0

# Lead parameters
temp = 0.5
dband = 1200
# Tunneling amplitudes
gam = 0.005
t0 = np.sqrt(gam/(2*np.pi))
t00 = 0.0*t0




nsingle = 8
nstate = 2**nsingle

hsingle =  {(0,0): Eq1+Ev1+mu1.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (1,1): Eq1+Ev1+mu1.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (2,2): Eq2+Ev2+mu2.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (3,3): Eq2+Ev2+mu2.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (4,4): Eq3+Ev3+mu3.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (5,5): Eq3+Ev3+mu3.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (6,6): Eq4+Ev4+mu4.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (7,7): Eq4+Ev4+mu4.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (0,2): -omegapres,
            (0,4): -omegapres/2,
            (0,6): -omegapres,
            (1,3): -omegapres,
            (1,5): -omegapres/2,
            (1,7): -omegapres,
            (2,4): -omegapres,
            (2,6): -omegapres/2,
            (3,5): -omegapres,
            (3,7): -omegapres/2,
            (4,6): -omegapres,
            (5,7): -omegapres}

# 0 is up, 1 is down


coulomb = {(0,1,1,0):U11.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (2,3,3,2):U22.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (4,5,5,4):U33.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (6,7,7,6):U44.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,2,2,0):Um12.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,3,3,0):Um12.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,2,2,1):Um12.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,3,3,1):Um12.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,4,4,0):Um13.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,5,5,0):Um13.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,4,4,1):Um13.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,5,5,1):Um13.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,6,6,0):Um14.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,7,7,0):Um14.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,6,6,1):Um14.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,7,7,1):Um14.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (2,4,4,2):Um23.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (2,5,5,2):Um23.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (3,4,4,3):Um23.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (3,5,5,3):Um23.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (2,6,6,2):Um24.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (2,7,7,2):Um24.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (3,6,6,3):Um24.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (3,7,7,3):Um24.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (4,6,6,4):Um34.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (4,7,7,4):Um34.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (5,6,6,5):Um34.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (5,7,7,5):Um34.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)])}



tleads = {(0, 0):-t0, # L, up   <-- up
          (1, 1):-t0, # R, up   <-- up
          (2, 6):-t0, # L, down <-- down
          (3, 7):-t0,
          (4, 2):-t0,
          (5, 3):-t0,
          (6, 4):-t0,
          (7, 5):-t0}



nleads = 8


mulst = {0: -vbiasL, 1: -vbiasL, 2: -vbiasL, 3: -vbiasL,
          4: -vbiasR, 5: -vbiasR, 6: -vbiasR, 7: -vbiasR}
tlst =  {0: temp,    1: temp,     2: temp,    3: temp,
          4: temp,  5: temp, 6: temp, 7: temp}



system = qmeq.Builder(nsingle, hsingle, coulomb,
                      nleads, tleads, mulst, tlst, dband,
                      kerntype='Pauli')


# Here we have chosen to use **Pauli master equation** (*kerntype='Pauli'*) to describe the stationary state. Let's calculate the current through the system:



system.solve()

print("System indexing: ")
print(system.indexing)
print("Dots Eigenenergies: ")
print(system.Ea)
"""
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
"""

for i in range(0,nstate):
  system.print_state(i)

#################################  construct the current operater here |1d> to |2d>  ############################
current_1dx2d = np.zeros((nstate,nstate),dtype=np.complex_)
current_1dx2d[8,4] = 1
current_1dx2d[4,8] = -1
current_1dx2d[9,5] = 1
current_1dx2d[5,9] = -1
current_1dx2d[10,6] = 1
current_1dx2d[6,10] = -1
current_1dx2d[11,7] = 1
current_1dx2d[7,11] = -1

current_4dx3d = np.zeros((nstate,nstate),dtype=np.complex_)
current_4dx3d[1,2] = 1
current_4dx3d[2,1] = -1
current_4dx3d[5,6] = 1
current_4dx3d[6,5] = -1
current_4dx3d[9,10] = 1
current_4dx3d[10,9] = -1
current_4dx3d[13,14] = 1
current_4dx3d[14,13] = -1

current_1dx4d = np.zeros((nstate,nstate),dtype=np.complex_)
current_1dx4d[8,1] = 1
current_1dx4d[1,8] = -1
current_1dx4d[10,3] = 1
current_1dx4d[3,10] = -1
current_1dx4d[12,5] = 1
current_1dx4d[5,12] = -1
current_1dx4d[14,7] = 1
current_1dx4d[7,14] = -1

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
            (6,6): Eq4+Ev4+mu4.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (7,7): Eq4+Ev4+mu4.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
            (0,2): -omegapres,
            (0,4): -omegapres/10,
            (0,6): -omegapres,
            (1,3): -omegapres,
            (1,5): -omegapres/10,
            (1,7): -omegapres,
            (2,4): -omegapres,
            (2,6): -omegapres/10,
            (3,5): -omegapres,
            (3,7): -omegapres/10,
            (4,6): -omegapres,
            (5,7): -omegapres})

            system.change(coulomb = {(0,1,1,0):U11.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (2,3,3,2):U22.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (4,5,5,4):U33.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (6,7,7,6):U44.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,2,2,0):Um12.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,3,3,0):Um12.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,2,2,1):Um12.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,3,3,1):Um12.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,4,4,0):Um13.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,5,5,0):Um13.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,4,4,1):Um13.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,5,5,1):Um13.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,6,6,0):Um14.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (0,7,7,0):Um14.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,6,6,1):Um14.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (1,7,7,1):Um14.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (2,4,4,2):Um23.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (2,5,5,2):Um23.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (3,4,4,3):Um23.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (3,5,5,3):Um23.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (2,6,6,2):Um24.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (2,7,7,2):Um24.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (3,6,6,3):Um24.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (3,7,7,3):Um24.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (4,6,6,4):Um34.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (4,7,7,4):Um34.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (5,6,6,5):Um34.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)]),
          (5,7,7,5):Um34.subs([(vL, vbiasL),(vR, vbiasR),(vGU, vgateup),(vGD, vgatedown)])})

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
            #product = np.matmul(dm_fock,(current_1dx4d))
            #stab[j1, j2] = abs(np.trace(product))
            stab[j1, j2] = system.current[0] + system.current[1] + system.current[2] + system.current[3]
            stab[j1, j2] = abs(stab[j1, j2])
        print("--- %s seconds ---" % (time.time() - start_time))
    return stab, stab_cond

# We changed the single particle Hamiltonian by calling the function **system.change** and specifying which matrix elements to change. The function **system.add** adds a value to a specified parameter. Also the option *masterq=False* in **system.solve** indicates just to diagonalise the quantum dot Hamiltonian and the master equation is not solved. Similarly, the option *qdq=False* means that the quantum dot Hamiltonian is not diagonalized (it was already diagonalized previously) and just master equation is solved.


system.kerntype = 'Pauli'
vpnt, vgpnt = 51, 51
vlst = np.linspace(-200, 600, vpnt)
vglst = np.linspace(-200, 600, vgpnt)
start_time = time.time()
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