
# Prerequisites
from __future__ import division, print_function
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np


import qmeq
import random

# Quantum dot parameters
vgate = 0.0
bfield = 0.0
omega = 0.0
vbias = 20

cL1,cG1,cm = 1.6, 0.6, 1.0
cL2,c12,c21,cG2 = 0.3, 0.2, 0.2, 0.5
cR2,cR1 = 1.6, 0.3

q10 = 0.0 
q20 = 0.0

csigma1 = cL1 + cG1 + cm + c12 + cR1
csigma2 = cR2 + cG2 + cm + c21 + cL2

U1 = (csigma2/(csigma1*csigma2-cm**2))/6.24*1000
U2 = (csigma1/(csigma1*csigma2-cm**2))/6.24*1000
Um = (cm/(csigma1*csigma2 - cm**2))/6.24*1000

print(U1)
print(U2)
print(Um)

vgate1, vgate2, vbiasL, vbiasR = 285.0, 60.0, 10.0, 0.0

q1 = (c12*vgate2 + cG1*vgate1 + cL1*vbiasL + cR1*vbiasR)*6.24/1000 + q10
q2 = (c21*vgate1 + cG2*vgate2 + cL2*vbiasL + cR2*vbiasR)*6.24/1000 + q10

mu1 = Um*q2 + U1*q1 - U1/2
mu2 = Um*q1 + U2*q2 - U2/2

omegapres, omegaflip = 0.01*U1, 0.00*U1

Jev = 0.01 * random.random()
Jt1v = 0.01 * random.random()
Jt2v = 0.01 * random.random()

Je = 0.00*U1
Jp = 0.00*U1
Jt1 = 0.00*U1
Jt2 = 0.00*U1


Eq1 = 0.5 * U1
Eq2 = 0.0 * U1
Eq1h = 0.5 * U1
Eq2h = 0.0 * U1

# Lead parameters
temp = 0.5
dband = 1200
# Tunneling amplitudes
gam = 0.005
t0 = np.sqrt(gam/(2*np.pi))
t00 = 0.0*t0




nsingle = 4
nstate = 2**nsingle

hsingle =  {(0,0): Eq1-mu1+bfield/2,
            (1,1): Eq1-mu1-bfield/2,
            (2,2): Eq2-mu2+bfield/2,
            (3,3): Eq2-mu2-bfield/2,
            (0,2): -omegapres,
            (1,3): -omegapres,
            (0,3): -omegaflip,
            (1,2): -omegaflip
}

# 0 is up, 1 is down

print(Eq1-mu1)
print(Eq2-mu2)



coulomb = {(0,1,1,0):U1,
          (1,2,2,1):Um,
          (0,2,2,0):Um-Je,
          (1,3,3,1):Um-Je,
          (0,3,3,0):Um,
          (2,3,3,2):U2,
          (1,2,3,0):-Je,
          (0,3,2,1):-Je,
          (2,3,0,1):-Jp,
          (0,1,2,3):-Jp,
          (0,1,3,0):-Jt1,
          (0,1,1,2):-Jt1,
          (1,2,2,3):-Jt2,
          (1,3,3,2):-Jt2,
          (0,3,1,0):-Jt1,
          (1,2,0,1):-Jt2,
          (2,3,0,3):-Jt2}



tleads = {(0, 0):-t0, # L, up   <-- up
          (1, 0):-t00, # R, up   <-- up
          (2, 1):-t0, # L, down <-- down
          (3, 1):-t00,
          (4, 2):-t00,
          (5, 2):-t0,
          (6, 3):-t00,
          (7, 3):-t0} # R, down <-- down



nleads = 8

vbiasL = vbias/2
vblasR = -vbias/2

#        L,up        R,up         L,down      R,down
mulst = {0: -vbiasL, 1: -vbiasR, 2: -vbiasL, 3: -vbiasR,
          4: -vbiasL, 5: -vbiasR, 6: -vbiasL, 7: -vbiasR}
tlst =  {0: temp,    1: temp,     2: temp,    3: temp,
          4: temp,  5: temp,  6: temp,  7: temp}



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



#################################  construct the current operater here |1d> to |2d>  ############################
current_1dx2d = np.zeros((nstate,nstate),dtype=np.complex_)
current_1dx2d[1,4] = 1
current_1dx2d[4,1] = -1
current_1dx2d[3,6] = 1
current_1dx2d[6,3] = -1
current_1dx2d[9,12] = 1
current_1dx2d[12,9] = -1
current_1dx2d[11,14] = 1
current_1dx2d[14,11] = -1
print(current_1dx2d)


################################ The current is the trace of product dm_fock and current_1dx2d  ##############################
product = np.matmul(dm_fock,current_1dx2d)
print("Current 1d->2d is: ")
print(np.trace(product))





for i in range(0,4):
  print("#####################")
  


def stab_calc(system, bfield, vlst, vglst, dV=0.0001):
    vpnt, vgpnt = vlst.shape[0], vglst.shape[0]

    stab = np.zeros((vpnt, vgpnt))
    stab_cond = np.zeros((vpnt, vgpnt))
    print(vpnt)
    for j1 in range(vgpnt):
        vgate1 = vglst[j1]
        print(j1)


        for j2 in range(vpnt):
            vgate2 = vlst[j2]
            q1 = (c12*vgate2 + cG1*vgate1 + cL1*vbiasL + cR1*vbiasR)*6.24/1000 + q10
            q2 = (c21*vgate1 + cG2*vgate2 + cL2*vbiasL + cR2*vbiasR)*6.24/1000 + q20


            mu1 = Um*q2 + U1*q1 - U1/2
            mu2 = Um*q1 + U2*q2 - U2/2
            gg = 0.01*U1
            hh = 0.00*U1


            system.change(hsingle={(0,0): Eq1-mu1+bfield/2,
            (1,1): Eq1h-mu1-bfield/2,
            (2,2): Eq2-mu2+bfield/2,
            (3,3): Eq2h-mu2-bfield/2,
            (0,2): -gg,
            (1,3): -gg,
            (0,3): -hh,
            (1,2): -hh})
            system.solve(masterq=False)

            #################  ploting the expectation value of 1d->2d current in gate-gate map  ########################  

            system.solve(qdq=False,currentq=False)
            for i in range(0, nstate):
              for j in range(0, nstate):
                dm_eigen[i,j] = system.get_phi0(i,j)

            eigenstate_in_fock = fock_coeff()
            eig_T = np.transpose(eigenstate_in_fock)
            eig_C = np.conjugate(eigenstate_in_fock)
            dm_fock = np.matmul(eig_T, np.matmul(dm_eigen, eig_C))
            product = np.matmul(dm_fock,current_1dx2d)

            stab[j1, j2] = abs(np.trace(product))

    return stab, stab_cond

# We changed the single particle Hamiltonian by calling the function **system.change** and specifying which matrix elements to change. The function **system.add** adds a value to a specified parameter. Also the option *masterq=False* in **system.solve** indicates just to diagonalise the quantum dot Hamiltonian and the master equation is not solved. Similarly, the option *qdq=False* means that the quantum dot Hamiltonian is not diagonalized (it was already diagonalized previously) and just master equation is solved.


#system.kerntype = 'Lindblad'
vpnt, vgpnt = 101, 101
vlst = np.linspace(-200, 600, vpnt)
vglst = np.linspace(-200, 600, vgpnt)
stab, stab_cond = stab_calc(system, bfield, vlst, vglst)


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
    cbar1.set_label('Energy [meV]', fontsize=20)

    plt.tight_layout()
    plt.show()

stab_plot(stab, stab_cond, vlst, vglst, gam)