
# Prerequisites
from __future__ import division, print_function
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np


import qmeq


# Quantum dot parameters
vgate = 0.0
bfield = 0.0
omega = 0.0

cL1,cG1,cm = 1.6, 0.4, 1.2
cL2,c12,c21,cG2 = 0.5, 0.2, 0.2, 0.4
cR2,cR1 = 1.6, 0.5

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

vgate1, vgate2, vbiasL, vbiasR = 0.0, 0.0, 10.0, 0.0

q1 = (c12*vgate2 + cG1*vgate1 + cL1*vbiasL + cR1*vbiasR)*6.24/1000 + q10
q2 = (c21*vgate1 + cG2*vgate2 + cL2*vbiasL + cR2*vbiasR)*6.24/1000 + q10

mu1 = Um*q2 + U1*q1 - U1/2
mu2 = Um*q1 + U2*q2 - U2/2

omegapres, omegaflip = 0.01*U1, 0.00*U1

Je = 0.00*U1
Jp = 0.00*U1
Jt1 = 0.00*U1
Jt2 = 0.00*U1

Eq1 = 0.0 * U1
Eq2 = 0.0 * U1

# Lead parameters
temp = 0.5
dband = 1200
# Tunneling amplitudes
gam = 0.005
t0 = np.sqrt(gam/(2*np.pi))
t00 = 0.0*t0




nsingle = 4

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
                     # lead label, lead spin <-- level spin



nleads = 8

#        L,up        R,up         L,down      R,down
mulst = {0: -vbiasL, 1: -vbiasR, 2: -vbiasL, 3: -vbiasR,
          4: -vbiasL, 5: -vbiasR, 6: -vbiasL, 7: -vbiasR}
tlst =  {0: temp,    1: temp,     2: temp,    3: temp,
          4: temp,  5: temp,  6: temp,  7: temp}



system = qmeq.Builder(nsingle, hsingle, coulomb,
                      nleads, tleads, mulst, tlst, dband,
                      kerntype='Lindblad')


# Here we have chosen to use **Pauli master equation** (*kerntype='Pauli'*) to describe the stationary state. Let's calculate the current through the system:

# In[11]:

#system.solve()
#print('Current:')
#print(system.current)
#print(system.energy_current)


# The four entries correspond to current in $L\uparrow$, $R\uparrow$, $L\downarrow$, $R\downarrow$ lead channels. We see that the current through left lead and right lead channels is conserved up to numerical errors:

# In[12]:

#print('Current continuity:')
#print(np.sum(system.current))
#print(system.indexing)
for i in range(0,4):
  print("#####################")
  #system.print_state(i)


def stab_calc(system, bfield, vlst, vglst, dV=0.0001):
    vpnt, vgpnt = vlst.shape[0], vglst.shape[0]

    stab = np.zeros((vpnt, vgpnt))
    stab_cond = np.zeros((vpnt, vgpnt))
    print(vpnt)
    for j1 in range(vgpnt):
        vgate1 = vglst[j1]
        vgate2 = vglst[j1]
        print(j1)
        #print(vgate1)
        #print(vgate2)

        for j2 in range(vpnt):
            vbiasL = vlst[j2]/2
            vbiasR = -vlst[j2]/2
            q1 = (c12*vgate2 + cG1*vgate1 + cL1*vbiasL + cR1*vbiasR)*6.24/1000 + q10
            q2 = (c21*vgate1 + cG2*vgate2 + cL2*vbiasL + cR2*vbiasR)*6.24/1000 + q20

            #print(q1)
            #print(q2)

            mu1 = Um*q2 + U1*q1 - U1/2
            mu2 = Um*q1 + U2*q2 - U2/2

            system.change(hsingle={(0,0): Eq1-mu1+bfield/2,
            (1,1): Eq1-mu1-bfield/2,
            (2,2): Eq2-mu2+bfield/2,
            (3,3): Eq2-mu2-bfield/2,
            (0,2): -omegapres,
            (1,3): -omegapres,
            (0,3): -omegaflip,
            (1,2): -omegaflip})
            system.solve(masterq=False)
            system.change(mulst={0: vlst[j2]/2, 1: -vlst[j2]/2,
                                 2: vlst[j2]/2, 3: -vlst[j2]/2,
                                 4: vlst[j2]/2, 5: -vlst[j2]/2,
                                 6: vlst[j2]/2, 7: -vlst[j2]/2})
            system.solve(qdq=False)
            stab[j1, j2] = system.current[0] + system.current[2] + system.current[4] + system.current[6]
            #stab[j1,j2] = mu1-mu2
            #
            #system.add(mulst={0: dV/2, 1: -dV/2,
                              #2: dV/2, 3: -dV/2})
            #system.solve(qdq=False)
            #stab_cond[j1, j2] = (system.current[0] + system.current[2] - stab[j1, j2])/dV
    #
    return stab, stab_cond

# We changed the single particle Hamiltonian by calling the function **system.change** and specifying which matrix elements to change. The function **system.add** adds a value to a specified parameter. Also the option *masterq=False* in **system.solve** indicates just to diagonalise the quantum dot Hamiltonian and the master equation is not solved. Similarly, the option *qdq=False* means that the quantum dot Hamiltonian is not diagonalized (it was already diagonalized previously) and just master equation is solved.

# In[19]:

system.kerntype = 'Lindblad'
vpnt, vgpnt = 201, 201
vlst = np.linspace(-100, 100, vpnt)
vglst = np.linspace(-300, 300, vgpnt)
stab, stab_cond = stab_calc(system, bfield, vlst, vglst)


# The stability diagram has been produced. Let's see how it looks like:

# In[20]:

def stab_plot(stab, stab_cond, vlst, vglst, gam):
    (xmin, xmax, ymin, ymax) = np.array([vglst[0], vglst[-1], vlst[0], vlst[-1]])
    fig = plt.figure(figsize=(8,6))
    #
    p1 = plt.subplot(1, 1, 1)
    p1.set_xlabel('$V_{g}(mV)$', fontsize=20)
    p1.set_ylabel('$V(mV)$', fontsize=20)
    p1_im = plt.imshow(stab.T/gam, extent=[xmin, xmax, ymin, ymax], aspect='auto', origin='lower', cmap = plt.get_cmap('Spectral'))
    cbar1 = plt.colorbar(p1_im)
    cbar1.set_label('Current [$\Gamma$]', fontsize=20)
    #
    #p2 = plt.subplot(1, 2, 2)
    #p2.set_xlabel('$V_{g}/U$', fontsize=20);
    #p2.set_ylabel('$V/U$', fontsize=20);
    #p2_im = plt.imshow(stab_cond.T, extent=[xmin, xmax, ymin, ymax], aspect='auto', origin='lower', cmap = plt.get_cmap('Spectral'))
    #cbar2 = plt.colorbar(p2_im)
    #cbar2.set_label('Conductance $\mathrm{d}I/\mathrm{d}V$', fontsize=20)
    #
    plt.tight_layout()
    plt.show()

stab_plot(stab, stab_cond, vlst, vglst, gam)