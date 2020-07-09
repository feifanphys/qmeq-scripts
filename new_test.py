
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
U = 20.0


# In[5]:

# Lead parameters
vbias = 0.5
temp = 1.0
dband = 300.0
# Tunneling amplitudes
gam = 0.5
t0 = np.sqrt(gam/(2*np.pi))




nsingle = 2

# 0 is up, 1 is down
hsingle = {(0, 0): vgate+bfield/2,
           (1, 1): vgate-bfield/2,
           (0, 1): omega}



coulomb = {(0,1,1,0):U}



tleads = {(0, 0):t0, # L, up   <-- up
          (1, 0):t0, # R, up   <-- up
          (2, 1):t0, # L, down <-- down
          (3, 1):t0} # R, down <-- down
                     # lead label, lead spin <-- level spin



nleads = 4

#        L,up        R,up         L,down      R,down
mulst = {0: vbias/2, 1: -vbias/2, 2: vbias/2, 3: -vbias/2}
tlst =  {0: temp,    1: temp,     2: temp,    3: temp}



system = qmeq.Builder(nsingle, hsingle, coulomb,
                      nleads, tleads, mulst, tlst, dband,
                      kerntype='Lindblad')


# Here we have chosen to use **Pauli master equation** (*kerntype='Pauli'*) to describe the stationary state. Let's calculate the current through the system:

# In[11]:

system.solve()
print('Current:')
print(system.current)
print(system.energy_current)


# The four entries correspond to current in $L\uparrow$, $R\uparrow$, $L\downarrow$, $R\downarrow$ lead channels. We see that the current through left lead and right lead channels is conserved up to numerical errors:

# In[12]:

print('Current continuity:')
print(np.sum(system.current))
print(system.indexing)
for i in range(0,4):
  print("#####################")
  system.print_state(i)
# If we want to change the approach we could redefine the system with the **qmeq.Builder** by specifying the new *kerntype*. It is also possible just to change the value of *system.kerntype*:

# In[13]:

#kernels = ['Redfield', '1vN', 'Lindblad', 'Pauli']
#for kerntype in kernels:
#    system.kerntype = kerntype
#    system.solve()

# ### Stability diagram
#
# Usually in experiments there is control of bias $V$ and gate voltage $V_{g}$. A contour plot of current or conductance in $(V, V_{g})$ plane is called a stability diagram. Let's produce such stability diagram for our quantum dot using **Pauli master equation**:

# In[18]:

def stab_calc(system, bfield, vlst, vglst, dV=0.0001):
    vpnt, vgpnt = vlst.shape[0], vglst.shape[0]
    stab = np.zeros((vpnt, vgpnt))
    stab_cond = np.zeros((vpnt, vgpnt))
    #
    for j1 in range(vgpnt):
        system.change(hsingle={(0,0):vglst[j1]+bfield/2,
                               (1,1):vglst[j1]-bfield/2})
        system.solve(masterq=False)
        for j2 in range(vpnt):
            system.change(mulst={0: vlst[j2]/2, 1: -vlst[j2]/2,
                                 2: vlst[j2]/2, 3: -vlst[j2]/2})
            system.solve(qdq=False)
            stab[j1, j2] = system.current[0] + system.current[2]
            #
            system.add(mulst={0: dV/2, 1: -dV/2,
                              2: dV/2, 3: -dV/2})
            system.solve(qdq=False)
            stab_cond[j1, j2] = (system.current[0] + system.current[2] - stab[j1, j2])/dV
    #
    return stab, stab_cond

# We changed the single particle Hamiltonian by calling the function **system.change** and specifying which matrix elements to change. The function **system.add** adds a value to a specified parameter. Also the option *masterq=False* in **system.solve** indicates just to diagonalise the quantum dot Hamiltonian and the master equation is not solved. Similarly, the option *qdq=False* means that the quantum dot Hamiltonian is not diagonalized (it was already diagonalized previously) and just master equation is solved.

# In[19]:

system.kerntype = 'Lindblad'
vpnt, vgpnt = 201, 201
vlst = np.linspace(-2*U, 2*U, vpnt)
vglst = np.linspace(-2.5*U, 1.5*U, vgpnt)
stab, stab_cond = stab_calc(system, bfield, vlst, vglst)


# The stability diagram has been produced. Let's see how it looks like:

# In[20]:

def stab_plot(stab, stab_cond, vlst, vglst, U, gam):
    (xmin, xmax, ymin, ymax) = np.array([vglst[0], vglst[-1], vlst[0], vlst[-1]])/U
    fig = plt.figure(figsize=(8,6))
    #
    p1 = plt.subplot(1, 1, 1)
    p1.set_xlabel('$V_{g}/U$', fontsize=20)
    p1.set_ylabel('$V/U$', fontsize=20)
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

stab_plot(stab, stab_cond, vlst, vglst, U, gam)