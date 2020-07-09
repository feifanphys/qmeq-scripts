

# Prerequisites
from __future__ import division, print_function
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
#from IPython.display import Image

import qmeq
from sympy import *


Cg = 2.6
Cs = 5.6
Cd = 5.6
Csigma = 2 * Cg + Cs + Cd

E_c = (1/Csigma)/6.24*1000
print(E_c)

Q_0 = 0

# Quantum dot parameters
vgate = 0.0
bfield = 0.0
omega = 0.0



# In[5]:

# Lead parameters
vbias = 2
temp = 0.5
dband = 1000.0
# Tunneling amplitudes
gam = 0.5
t0 = np.sqrt(gam/(2*np.pi))


vGU, vGD, vL, vR = symbols("vGU, vGD, vL, vR")
n1 = symbols("n1")

q1 = (Cs*vL+Cd*vR+Cg*vGU+Cg*vGD)*6.24/1000

C = Matrix([[Csigma]])*6.24

Q = Matrix([[q1-n1]])

U=0.5*((C.inv()*Q).T*Q)*1000
print(U)
U0 = U.subs([(n1,0)])[0]
U1 = U.subs([(n1,1)])[0]
U2 = U.subs([(n1,2)])[0]
mu1 = U1 - U0
U_m = U2 - U1 - mu1

print(U_m)


E_0 = -150
E_1 = -75
E_2 = -30
E_3 = -4

nsingle = 8

# 0 is up, 1 is down
hsingle = {(0, 0): 0.5*E_c - 3 * E_c,
           (1, 1): 0.5*E_c - 2 * E_c,
           (2, 2): 0.5*E_c - 2 * E_c,
           (3, 3): 0.5*E_c + 0 * E_c,
           (4, 4): 0.5*E_c + 0 * E_c,
           (5, 5): 0.5*E_c + 0 * E_c,
           (6, 6): 0.5*E_c + 0 * E_c,
           (7, 7): 0.5*E_c + 0 * E_c}



#

coulomb = {(0,1,1,0): U_m,
            (0,2,2,0): U_m,
            (0,3,3,0): U_m,
            (0,4,4,0): U_m,
            (0,5,5,0): U_m,
            (0,6,6,0): U_m,
            (0,7,7,0): U_m,
            (1,2,2,1): U_m,
            (1,3,3,1): U_m,
            (1,4,4,1): U_m,
            (1,5,5,1): U_m,
            (1,6,6,1): U_m,
            (1,7,7,1): U_m,
            (2,3,3,2): U_m,
            (2,4,4,2): U_m,
            (2,5,5,2): U_m,
            (2,6,6,2): U_m,
            (2,7,7,2): U_m,
            (3,4,4,3): U_m,
            (3,5,5,3): U_m,
            (3,6,6,3): U_m,
            (3,7,7,3): U_m,
            (4,5,5,4): U_m,
            (4,6,6,4): U_m,
            (4,7,7,4): U_m,
            (5,6,6,5): U_m,
            (5,7,7,5): U_m,
            (6,7,7,6): U_m}


tleads = {(0, 0):t0, # L, up   <-- up
          (1, 0):t0, # R, up   <-- up
          (0, 1):t0, # L, down <-- down
          (1, 1):t0,
          (0, 2):t0,
          (1, 2):t0,
          (0, 3):t0,
          (1, 3):t0,
          (0, 4):t0, # L, up   <-- up
          (1, 4):t0, # R, up   <-- up
          (0, 5):t0, # L, down <-- down
          (1, 5):t0,
          (0, 6):t0,
          (1, 6):t0,
          (0, 7):t0,
          (1, 7):t0} # R, down <-- down
                     # lead label, lead spin <-- level spin





nleads = 2

#        L,up        R,up         L,down      R,down
mulst = {0: vbias/2, 1: -vbias/2}
tlst =  {0: temp,    1: temp}




system = qmeq.Builder(nsingle, hsingle, coulomb,
                      nleads, tleads, mulst, tlst, dband,
                      kerntype='Pauli')




system.solve()
print('Pauli current:')
print(system.current)
print(system.energy_current)


# The four entries correspond to current in $L\uparrow$, $R\uparrow$, $L\downarrow$, $R\downarrow$ lead channels. We see that the current through left lead and right lead channels is conserved up to numerical errors:

# In[12]:

print('Current continuity:')
print(np.sum(system.current))


# If we want to change the approach we could redefine the system with the **qmeq.Builder** by specifying the new *kerntype*. It is also possible just to change the value of *system.kerntype*:





def stab_calc(system, bfield, vlst, vglst, dV=0.0001):
    vpnt, vgpnt = vlst.shape[0], vglst.shape[0]
    stab = np.zeros((vpnt, vgpnt))
    stab_cond = np.zeros((vpnt, vgpnt))
    #
    for j1 in range(vgpnt):
        print(j1)
        for j2 in range(vpnt):
            system.change(hsingle={(0,0):E_0 + mu1.subs([(vL, vlst[j2]/2),(vR, -vlst[j2]/2),(vGU, vglst[j1]),(vGD, vglst[j1])]),
                          (1,1):E_0 + mu1.subs([(vL, vlst[j2]/2),(vR, -vlst[j2]/2),(vGU, vglst[j1]),(vGD, vglst[j1])]),
                          (2,2):E_1 + mu1.subs([(vL, vlst[j2]/2),(vR, -vlst[j2]/2),(vGU, vglst[j1]),(vGD, vglst[j1])]),
                          (3,3):E_1 + mu1.subs([(vL, vlst[j2]/2),(vR, -vlst[j2]/2),(vGU, vglst[j1]),(vGD, vglst[j1])]),
                          (4,4):E_2 + mu1.subs([(vL, vlst[j2]/2),(vR, -vlst[j2]/2),(vGU, vglst[j1]),(vGD, vglst[j1])]),
                          (5,5):E_2 + mu1.subs([(vL, vlst[j2]/2),(vR, -vlst[j2]/2),(vGU, vglst[j1]),(vGD, vglst[j1])]),
                          (6,6):E_3 + mu1.subs([(vL, vlst[j2]/2),(vR, -vlst[j2]/2),(vGU, vglst[j1]),(vGD, vglst[j1])]),
                          (7,7):E_3 + mu1.subs([(vL, vlst[j2]/2),(vR, -vlst[j2]/2),(vGU, vglst[j1]),(vGD, vglst[j1])])})
            tmp = U_m.subs([(vL, vlst[j2]/2),(vR, -vlst[j2]/2),(vGU, vglst[j1]),(vGD, vglst[j1])])
            system.change(coulomb = {(0,1,1,0): tmp,
            (0,2,2,0): tmp,
            (0,3,3,0): tmp,
            (0,4,4,0): tmp,
            (0,5,5,0): tmp,
            (0,6,6,0): tmp,
            (0,7,7,0): tmp,
            (1,2,2,1): tmp,
            (1,3,3,1): tmp,
            (1,4,4,1): tmp,
            (1,5,5,1): tmp,
            (1,6,6,1): tmp,
            (1,7,7,1): tmp,
            (2,3,3,2): tmp,
            (2,4,4,2): tmp,
            (2,5,5,2): tmp,
            (2,6,6,2): tmp,
            (2,7,7,2): tmp,
            (3,4,4,3): tmp,
            (3,5,5,3): tmp,
            (3,6,6,3): tmp,
            (3,7,7,3): tmp,
            (4,5,5,4): tmp,
            (4,6,6,4): tmp,
            (4,7,7,4): tmp,
            (5,6,6,5): tmp,
            (5,7,7,5): tmp,
            (6,7,7,6): tmp})
            system.solve(masterq=False)
            system.change(mulst={0: vlst[j2]/2, 1: -vlst[j2]/2})
            system.solve(qdq=False)
            stab[j1, j2] = abs(system.current[0])

    #
    return stab, stab_cond

# We changed the single particle Hamiltonian by calling the function **system.change** and specifying which matrix elements to change. The function **system.add** adds a value to a specified parameter. Also the option *masterq=False* in **system.solve** indicates just to diagonalise the quantum dot Hamiltonian and the master equation is not solved. Similarly, the option *qdq=False* means that the quantum dot Hamiltonian is not diagonalized (it was already diagonalized previously) and just master equation is solved.

# In[19]:

system.kerntype = 'Pauli'
vpnt, vgpnt = 201, 201
vlst = np.linspace(-100, 100, vpnt)
vglst = np.linspace(-800, 200, vgpnt)
stab, stab_cond = stab_calc(system, bfield, vlst, vglst)


# The stability diagram has been produced. Let's see how it looks like:

# In[20]:

def stab_plot(stab, stab_cond, vlst, vglst, gam):
    (xmin, xmax, ymin, ymax) = np.array([vglst[0], vglst[-1], vlst[0], vlst[-1]])
    fig = plt.figure(figsize=(6,4.2))
    #
    p1 = plt.subplot(1, 1, 1)
    p1.set_xlabel('$V_{g}(meV)$', fontsize=20)
    p1.set_ylabel('$V_{b}(meV)$', fontsize=20)
    p1_im = plt.imshow(stab.T/gam, extent=[xmin, xmax, ymin, ymax], aspect='auto', origin='lower', cmap = plt.get_cmap('Spectral'))
    cbar1 = plt.colorbar(p1_im)
    cbar1.set_label('Current [$\Gamma$]', fontsize=20)
    tt = "2 donor levels = " + str(nsingle) + " Single island"
    plt.title(tt, loc='center')
    #
    #p2 = plt.subplot(1, 2, 2)
    #p2.set_xlabel('$V_{g}$', fontsize=20);
    #p2.set_ylabel('$V_{b}$', fontsize=20);
    #p2_im = plt.imshow(stab_cond.T, extent=[xmin, xmax, ymin, ymax], aspect='auto', origin='lower')
    #cbar2 = plt.colorbar(p2_im)
    #cbar2.set_label('Conductance $\mathrm{d}I/\mathrm{d}V$', fontsize=20)
    #
    plt.tight_layout()
    plt.show()

stab_plot(stab, stab_cond, vlst, vglst, gam)