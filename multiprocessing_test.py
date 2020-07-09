
# Prerequisites
from __future__ import division, print_function
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np


import qmeq
import random
import time
import multiprocessing
from itertools import product


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


vgate1, vgate2, vbiasL, vbiasR = 0.0, 0.0, 10.0, 0.0


omegapres, omegaflip = 0.01*U1, 0.00*U1


Je = 0.03*U1
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

hsingle =  {}

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

vbiasL = vbias/2
vblasR = -vbias/2

mulst = {0: -vbiasL, 1: -vbiasR, 2: -vbiasL, 3: -vbiasR,
        4: -vbiasL, 5: -vbiasR, 6: -vbiasL, 7: -vbiasR}
tlst =  {0: temp,    1: temp,     2: temp,    3: temp,
        4: temp,  5: temp,  6: temp,  7: temp}

system = qmeq.Builder(nsingle, hsingle, coulomb,
                      nleads, tleads, mulst, tlst, dband,
                      kerntype='Lindblad')
system.kerntype = 'Lindblad'
    
gg = 0.01*U1
# The four entries correspond to current in $L\uparrow$, $R\uparrow$, $L\downarrow$, $R\downarrow$ lead channels. We see that the current through left lead and right lead channels is conserved up to numerical errors:

# In[12]:

#print('Current continuity:')
#print(np.sum(system.current))
#print(system.indexing)


def stab_calc(vlst, vglst, dV=0.0001):

    vgate1 = vglst 
    vgate2 = vlst

    q1 = (c12*vgate2 + cG1*vgate1 + cL1*vbiasL + cR1*vbiasR)*6.24/1000 + q10
    q2 = (c21*vgate1 + cG2*vgate2 + cL2*vbiasL + cR2*vbiasR)*6.24/1000 + q20


    mu1 = Um*q2 + U1*q1 - U1/2
    mu2 = Um*q1 + U2*q2 - U2/2
    system.change(hsingle={(0,0): Eq1-mu1,
            (1,1): Eq1h-mu1,
            (2,2): Eq2-mu2,
            (3,3): Eq2h-mu2,
            (0,2): -gg,
            (1,3): -gg,
            (0,3): -omegaflip,
            (1,2): -omegaflip})
    system.solve(masterq=False)

    system.solve(qdq=False)
    stab = system.current[0] + system.current[2] + system.current[4] + system.current[6]
    stab = abs(stab)
    print("done")
    #return stab, stab_cond

# We changed the single particle Hamiltonian by calling the function **system.change** and specifying which matrix elements to change. The function **system.add** adds a value to a specified parameter. Also the option *masterq=False* in **system.solve** indicates just to diagonalise the quantum dot Hamiltonian and the master equation is not solved. Similarly, the option *qdq=False* means that the quantum dot Hamiltonian is not diagonalized (it was already diagonalized previously) and just master equation is solved.

# In[19]:

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




if __name__ == '__main__':
  # Quantum dot parameters
  for i in range(0,4):
    print("#####################")
  #system.print_state(i)

  start_time = time.time()
  vpnt, vgpnt = 101, 101
  stab = np.zeros((vgpnt,vpnt))
  stab_cond = np.zeros((vgpnt,vpnt))
  vlst = np.linspace(-200, 600, vpnt)
  vglst = np.linspace(-200, 600, vgpnt)
  pool = multiprocessing.Pool(multiprocessing.cpu_count())
  stab = pool.starmap(stab_calc,product(vlst,vglst))
  pool.close()
  print("--- %s seconds ---" % (time.time() - start_time))
  stab_plot(stab, stab_cond, vlst, vglst, gam)
#stab_calc(system, bfield, vlst, vglst)


# The stability diagram has been produced. Let's see how it looks like:
