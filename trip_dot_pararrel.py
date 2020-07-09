

# Prerequisites
from __future__ import division, print_function
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
#from IPython.display import Image




import qmeq




# Quantum dot parameters
vgate = 0.0
bfield = 0.0
omega = 0.0
U = 6
interval = 25


# In[5]:

# Lead parameters
vbias = 0.5
temp = 1.0
dband = 100.0
# Tunneling amplitudes
gam = 0.5
t0 = np.sqrt(gam/(2*np.pi))
tL = t0*1.1
tR = t0*1.2


alpha_1 = 0.8
alpha_2 = 0.2
beta_1 = 0.3
beta_2 = 0.7
offset = 3

nsingle = 3
mu = 3
tc = 1


# 0 is up, 1 is down
hsingle = {(0, 0): offset,
           (1, 1): 0.6*offset,
           (2, 2): 0,
           (0, 1): tc,
           (1, 2): tc}


#

coulomb = {(0,1,1,0): U,
            (1,2,2,1): U}



tleads = {(0, 0):tL, # L, up   <-- up
          (1, 0):tR, # L, down <-- down
          (0, 1):tL,
          (1, 1):tR,
          (0, 2):tL,
          (1, 2):tR} # R, down <-- down
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
        system.change(hsingle={(0,0):-0.9*vglst[j1]+offset,
                              (1,1):-0.8*vglst[j1]+ 0.6*offset,
                              (2,2):-0.85*vglst[j1],
                              (0,1): tc,
                              (1,2): tc})
        system.solve(masterq=False)
        for j2 in range(vpnt):
            system.change(mulst={0: vlst[j2]/2, 1: -vlst[j2]/2})
            system.solve(qdq=False)
            stab[j1, j2] = system.current[0]
            #
            system.add(mulst={0: dV/2, 1: -dV/2})
            system.solve(qdq=False)
            stab_cond[j1, j2] = (system.current[0] - stab[j1, j2])/dV
    #
    return stab, stab_cond

# We changed the single particle Hamiltonian by calling the function **system.change** and specifying which matrix elements to change. The function **system.add** adds a value to a specified parameter. Also the option *masterq=False* in **system.solve** indicates just to diagonalise the quantum dot Hamiltonian and the master equation is not solved. Similarly, the option *qdq=False* means that the quantum dot Hamiltonian is not diagonalized (it was already diagonalized previously) and just master equation is solved.

# In[19]:

system.kerntype = 'Pauli'
vpnt, vgpnt = 201, 201
vlst = np.linspace(-3*U, 3*U, vpnt)
vglst = np.linspace(-4*U, 4*U, vgpnt)
stab, stab_cond = stab_calc(system, bfield, vlst, vglst)


# The stability diagram has been produced. Let's see how it looks like:

# In[20]:

def stab_plot(stab, stab_cond, vlst, vglst, U, gam):
    (xmin, xmax, ymin, ymax) = np.array([vglst[0], vglst[-1], vlst[0], vlst[-1]])/U
    fig = plt.figure(figsize=(12,4.2))
    #
    p1 = plt.subplot(1, 2, 1)
    p1.set_xlabel('$V_{g}/U$', fontsize=20)
    p1.set_ylabel('$V/U$', fontsize=20)
    p1_im = plt.imshow(stab.T/gam, extent=[xmin, xmax, ymin, ymax], aspect='auto', origin='lower')
    cbar1 = plt.colorbar(p1_im)
    cbar1.set_label('Current [$\Gamma$]', fontsize=20)
    #
    p2 = plt.subplot(1, 2, 2)
    p2.set_xlabel('$V_{g}/U$', fontsize=20);
    p2.set_ylabel('$V/U$', fontsize=20);
    p2_im = plt.imshow(stab_cond.T, extent=[xmin, xmax, ymin, ymax], aspect='auto', origin='lower')
    cbar2 = plt.colorbar(p2_im)
    cbar2.set_label('Conductance $\mathrm{d}I/\mathrm{d}V$', fontsize=20)
    #
    plt.tight_layout()
    plt.show()

stab_plot(stab, stab_cond, vlst, vglst, U, gam)