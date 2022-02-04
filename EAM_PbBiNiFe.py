# This Python program can be used to generate a LAMMPS-compatible SETFL file,
# for EAM-based MD simulations of liquid Bi, Pb, Ni and Fe and their mixtures.
# Guido Raos, Politecnico di Milano, Italy:  guido.raos@polimi.it 

# If you publish work based on this code, please cite:
# Gao, Y., Takahashi, M., Cavallotti, C. and Raos, G., 2018.
# Molecular dynamics simulation of metallic impurity diffusion in liquid lead-bismuth eutectic (LBE).
# Journal of Nuclear Materials, 501, pp.253-260.
# https://www.sciencedirect.com/science/article/pii/S0022311517316525
# and
# Gao, Y., Raos, G., Cavallotti, C. and Takahashi, M., 2016.
# Molecular dynamics simulation on physical properties of liquid lead, bismuth and lead-bismuth eutectic (LBE).
# Procedia Engineering, 157, pp.214-221.
# https://www.sciencedirect.com/science/article/pii/S1877705816325346

# The EAM potentials for Bi and Pb are based on:
# Belashchenko, D. K. (2012).
# Computer simulation of the properties of liquid metals: Gallium, lead, and bismuth.
# Russian Journal of Physical Chemistry A, 86(5), 779-790.
# https://link.springer.com/content/pdf/10.1134/S0036024412050056.pdf 
# The EAM potentials for Ni and Fe are based on:
# Zhou, X. W.; Johnson, R. A.; Wadley, H. N. G.
# Misfit-Energy-Increasing Dislocations in Vapor-Deposited CoFe/NiFe Multilayers.
# Phys. Rev. B - Condens. Matter Mater. Phys. 2004, 69 (14), 1–10.
# https://doi.org/10.1103/PhysRevB.69.144113.

# The SETFL file format is described here: http://lammps.sandia.gov/doc/pair_eam.html
# In the simulations we emply "metal units": http://lammps.sandia.gov/doc/units.html

# Inportant note no. 0:
# Below, I have changed the definitions of nrho, drho, nr, dr and cutoff, which control
# the density and range of the density and embedding energy functions in the setfl file.

# Inportant note no. 1:
# In the the setfl file, the numerical values of the electron density and of the pair potentials
# should be given given at r=0, dr, 2*dr, 3*dr, ... (i.e., starting at 0, not at dr!).
# Similarly, the values of the embedding energies should be given at rho=0, drho, 2*drhom, 3*drho, ...

# Important note no. 2:
# In Zhou's Fortran code (distributed with LAMMPS), the electron densities and pair potentials
# for r<rst=0.5 (Angs) are simply set to the value at r=rst.
# We can probably apply the same short-distance cutoff also to Belashchencko's potentials,
# without unwanted consequences (no atoms should ever be found at such small distances).
rst = 0.5

from math import exp, log, pi

# Parameters for Pb and Bi, from Belashchenko's paper.

# Belashchencko's embedding energy
def PHIB(rho, rhoX, aaX, bbX, ccX):
   if rho >= rhoX[1]:
      y = aaX[0] + ccX[0]*(rho-rhoX[0])**2
   elif rho>rhoX[5]:
      for i in range(1,5):
         if rho>=rhoX[i+1]:
            drho = rho - rhoX[i]
            y = aaX[i] + bbX[i]*drho + ccX[i]*drho**2
            break
   else:
      i = 5
      drho = rho - rhoX[i]
      y = (aaX[i] + bbX[i]*drho + ccX[i]*drho**2)* (2.0*rho/rhoX[i] - (rho/rhoX[i])**2)
   return y

# Belashchencko's electron density function
def psiB(r, pX): 
   y = pX[0] * exp(-pX[1] * r)
   return y

# Belashchencko's pair potential (single element)
def phiB(r, aX, rX, bX):
   if r <= rX[0]:
      y = bX[0]
      y += bX[1] * ( rX[0] - r )
      y += bX[2] * ( exp(bX[3]*(rX[0]-r)) - 1. )
   elif r > rX[-1]:
      y = 0.0
#      y = '%f' % num  
   else:
      n = len(rX)
      k = len(aX)
      y = 0.
      for i in range(n-1):
         if r >= rX[i] and r <= rX[i+1]:
            dr = r-rX[i+1]
            for m in range(k):
               y += aX[m][i] * dr**m
   return y

# Parameters for Ni and Fe, from Zhou, Johnson and Wadley.
# Note that Johnson's "mixing rule" is used not only for Ni-Fe interactions,
# but also at all other heteroatomic cross-interactions involving Pb and Ni.

# Johnson's embedding energy 
def PHIJ(rho,rhoX,Fn,F,Fe,eta):
   rhon = 0.85*rhoX[0]
   rho0 = 1.15*rhoX[0]
   if rho >=0.0 and rho < rhon:
      n = len(Fn)
      y = Fn[0]
      for i in range(1,n):
         y += Fn[i]*(rho/rhon-1.)**i
   elif rho >= rhon and rho <= rho0:
      n = len(F)
      y = F[0]
      for i in range(1,n):
         y += F[i]*(rho/rhoX[0]-1.)**i
   else:
      y = Fe * (1.0-eta*log(rho/rhoX[1])) * (rho/rhoX[1])**eta
   return y  

# Johnson's electron density function
def psiJ(r,fe,beta,re,lam):
   y = fe * exp(-beta*(r/re-1.0))
   y /= (1.0+(r/re-lam)**20)
   return y

# Johnson's pair potential (single element)
def phiJ(r,A,alpha,re,kappa,B,beta,lam):
   y = psiJ(r,A,alpha,re,kappa) - psiJ(r,B,beta,re,lam)
   return y

# Pair potential (cross term between Pb and Bi elements)
def phicross1(r, pX,aX,rX,bX, \
                 pY,aY,rY,bY):
   fa = psiB(r,pX)
   fb = psiB(r,pY)
   phiaa = phiB(r,aX,rX,bX)
   phibb = phiB(r,aY,rY,bY)
   phiab = 0.5 * (fb*phiaa/fa + fa*phibb/fb)
   return phiab

# Pair potential (cross term between Pb, Bi, Ni and Fe elements)
def phicross2(r, AX,BX,alphaX,feX,betaX,reX,lamX,kappaX, \
                 pY,aY,rY,bY):
   fa = psiJ(r,feX,betaX,reX,lamX)
   fb = psiB(r,pY)
   phiaa = phiJ(r,AX,alphaX,reX,kappaX,BX,betaX,lamX)
   phibb = phiB(r,aY,rY,bY)
   phiab = 0.5 * (fb*phiaa/fa + fa*phibb/fb)
   return phiab

# Pair potential (cross term between Ni and Fe elements)
def phicross3(r, AX,BX,alphaX,feX,betaX,reX,lamX,kappaX, \
                 AY,BY,alphaY,feY,betaY,reY,lamY,kappaY):
   fa = psiJ(r,feX,betaX,reX,lamX)
   fb = psiJ(r,feY,betaY,reY,lamY)
   phiaa = phiJ(r,AX,alphaX,reX,kappaX,BX,betaX,lamX)
   phibb = phiJ(r,AY,alphaY,reY,kappaY,BY,betaY,lamY)
   phiab = 0.5 * (fb*phiaa/fa + fa*phibb/fb)
   return phiab

# Parameters for embedding energy of Pb
rhoPb = [1.0, 0.90, 0.81, 0.77, 0.71, 0.46, 1.40]
aaPb = [-1.5186, -1.5010, -1.4693, -1.4488, -1.4253, -1.2976]
bbPb = [ 0., -0.3524, -0.3524, -0.6724, -0.1084, -0.9134,]
ccPb = [1.7622, 0.0, 4.00, -4.70, 1.61, -5.70]

# Parameters for electron density of Pb
pPb = [5.1531, 1.2200]

# Parameters for the pair potential of Pb
aPb = [
       [-0.60930526815355E-02,-0.44151442125440E-02, 0.00000000000000E+00],
       [-0.13621575199068E-01,-0.61437641270459E-02, 0.00000000000000E+00],
       [-0.45660788997838E-01,-0.10177246335680E+00, 0.16082816702565E+00],
       [ 0.11230757774433E+00,-0.25452335322626E+00, 0.13772915239242E+01],
       [ 0.36172101010484E+00,-0.32596219577719E+00, 0.46100352215597E+01],
       [ 0.23710940431335E+00,-0.23344635267770E+00, 0.77406092052939E+01],
       [-0.14300929222822E+00,-0.92940364296357E-01, 0.69328765962609E+01],
       [-0.18657853386373E+00,-0.19153257528366E-01, 0.31673915035111E+01],
       [-0.44652604375371E-01,-0.15905291914747E-02, 0.58055948305632E+00],
      ]
rPb = [2.6, 4.6, 7.6, 9.01]
bPb = [0.438472, -3.99326, 2.8, 1.96]

# Parameters for embedding energy of Bi
rhoBi = [1.0, 0.90, 0.80, 0.70, 0.60, 0.28, 1.40]
aaBi = [-1.57509, -1.563872, -1.537416, -1.502190, -1.428144, -1.053501]
bbBi = [0., -0.224360, -0.304760, -0.399760, -1.081160, -1.260360]
ccBi = [1.1218, 0.402, 0.475, 3.407, 0.28, 0.0]

# Parameters for electron density of Bi
pBi = [5.28909, 1.200]

# Parameters for the pair potential of Bi
aBi = [
       [ 0.23032123222947E-01,-0.58089114725590E-01,-0.38583852350712E-01,-0.63335783779621E-02, 0.00000000000000E+00],
       [-0.26618006825447E+00, 0.45752499252558E-01, 0.78495843335986E-02, 0.10303066810593E-02, 0.00000000000000E+00],
       [-0.62050552158296E+01, 0.68828017622705E+00,-0.10112647441308E+01, 0.42480747246449E-01,-0.91599761109836E-02],
       [-0.92775522451735E+02, 0.22976325825531E+01,-0.10668364762145E+02, 0.81690273704111E-01,-0.29685481254410E-01],
       [-0.44789947487489E+03, 0.49502135238428E+01,-0.43008987321300E+02, 0.40740331582690E-01,-0.29950729803211E-01],
       [-0.87044414901813E+03,-0.72236788574834E+00,-0.73991049540494E+02, 0.59046334275219E-02,-0.11676661892341E-01],
       [-0.55341290239556E+03,-0.72211727467775E+01,-0.46109980129960E+02,-0.13747612449246E-03,-0.15697393361539E-02],
      ]
rBi = [2.50, 3.00, 3.50, 4.00, 6.50, 9.05]
bBi = [0.762385, -26.94609, 17.8, 1.96]

# Scaling factor for the densities of Ni and Fe
sfac = 0.0974586
#sfac = 1.0

# Parameters for embedding energy, electron density and pair potential of Ni
reNi = 2.488746
feNi = 2.007018*sfac
rhoNi = [27.562015*sfac, 27.930410*sfac] # The value of rhoe and rhos
ArNi = 8.383453
BeNi  = 4.471175
ANi = 0.429046
BNi = 0.633531
kNi = 0.443599
raNi = 0.820658
FnNi = [-2.693513, -0.076445, 0.241442, -2.375626]
FNi = [-2.70, 0, 0.265390, -0.152856]
etaNi = 0.469000
FaNi = -2.699486

# Parameters for embedding energy, electron density and pair potential of Fe
reFe = 2.481987
feFe = 1.885957*sfac
rhoFe = [20.041463*sfac, 20.041463*sfac] # The value of rhoe and rhos
ArFe = 9.818270
BeFe  = 5.236411
AFe = 0.392811
BFe = 0.646243
kFe = 0.170306
raFe = 0.340613
FnFe = [-2.534992, -0.059605, 0.193065, -2.282322]
FFe = [-2.54, 0, 0.200269, -0.148770]
etaFe = 0.391750
FaFe = -2.539945

# GUIDO'S parameters
nrho = 2000
rhomax = max(rhoBi[-1], rhoPb[-1], 4.*rhoNi[1], 4.*rhoFe[1])
drho = rhomax / (nrho-1)
nr = 2000
cutoff = max(rPb[-1], rBi[-1], reNi*(20.**0.5)/2, reFe*(20.**0.5)/2 ) 
dr = cutoff / (nr-1)

# GAO'S parameters
#drho = 0.001
#nrho = int(round(rhoBi[-1]/drho))
#dr = 0.005
#nr = int(round(rBi[5]/dr))
#cutoff = min(rPb[-1], rBi[-1]) 

# ZHOU'S parameters for Fe.
#nrho = 2000
#drho = 0.6176373362541199E-01 
#nr = 2000
#dr = 0.2776333829388022E-02
#cutoff = 0.5549891471862793E+01

# Print data in SETFL format for LAMMPS
print 'EAM parameters for Pb and Bi from Belashchenko, for Ni and Fe from Zhu-Johnson-Wadley.\nThe cross potentials are defined by Johnsons mixing rule.\n(17/02/16, Polimi, Milan, Italy)'
print '4   Pb   Bi   Ni   Fe'
#print '1   Fe'
print nrho, drho, nr, dr, cutoff

# Pb
# Atomic number, mass, and unimportant stuff
print '82   207.2   4.9095   fcc'
# Embedding energy for Pb
for i in range(nrho):
   rho = i*drho
   print PHIB(rho, rhoPb, aaPb, bbPb, ccPb)
# Electron density function for Pb
for i in range(nr):
   r = max(i*dr,rst)
   print psiB(r, pPb) 

# Bi
# Atomic number, mass, and unimportant stuff
print '83   208.98   3.8800   bcc'
# Embedding energy for Bi
for i in range(nrho):
   rho = i*drho
   print PHIB(rho, rhoBi, aaBi, bbBi, ccBi)
# Electron density function for Bi
for i in range(nr):
   r = max(i*dr,rst)
   print psiB(r, pBi)

# Ni
# Atomic number, mass, and unimportant stuff
print '28   58.693   3.5196   fcc'
# Embedding energy for Ni
for i in range(nrho):
   rho = i*drho
   print PHIJ(rho, rhoNi, FnNi, FNi, FaNi, etaNi)
# Electron density function for Ni
for i in range(nr):
   r = max(i*dr,rst)
   print psiJ(r, feNi, BeNi, reNi, raNi)

# Fe
# Atomic number, mass, and unimportant stuff
print '26   55.847   3.5101   fcc'
# Embedding energy for Fe
for i in range(nrho):
   rho = i*drho
   print PHIJ(rho, rhoFe, FnFe, FFe, FaFe, etaFe)
# Electron density function for Fe
for i in range(nr):
   r = max(i*dr,rst)
   print psiJ(r, feFe, BeFe, reFe, raFe)

# Pair potential for Pb-Pb
for i in range(nr):
   r = max(i*dr,rst)
   print r*phiB(r, aPb,rPb,bPb)

# Pair potential for Bi-Pb
for i in range(nr):
   r = max(i*dr,rst)
   print r*phicross1(r, pBi,aBi,rBi,bBi,
                        pPb,aPb,rPb,bPb)

# Pair potential for Bi-Bi
for i in range(nr):
   r = max(i*dr,rst)
   print r*phiB(r, aBi,rBi,bBi)

# Pair potential for Ni-Pb
for i in range(nr):
   r = max(i*dr,rst)
   print r*phicross2(r, ANi,BNi,ArNi,feNi,BeNi,reNi,raNi,kNi,
                        pPb,aPb,rPb,bPb)

# Pair potential for Ni-Bi
for i in range(nr):
   r = max(i*dr,rst)
   print r*phicross2(r, ANi,BNi,ArNi,feNi,BeNi,reNi,raNi,kNi,
                        pBi,aBi,rBi,bBi)

# Pair potential for Ni-Ni
for i in range(nr):
   r = max(i*dr,rst)
   print r*phiJ(r, ANi,ArNi,reNi,kNi,BNi,BeNi,raNi)

# Pair potential for Fe-Pb
for i in range(nr):
   r = max(i*dr,rst)
   print r*phicross2(r, AFe,BFe,ArFe,feFe,BeFe,reFe,raFe,kFe,
                        pPb,aPb,rPb,bPb)

# Pair potential for Fe-Bi
for i in range(nr):
   r = max(i*dr,rst)
   print r*phicross2(r, AFe,BFe,ArFe,feFe,BeFe,reFe,raFe,kFe,
                        pBi,aBi,rBi,bBi)

# Pair potential for Fe-Ni
for i in range(nr):
   r = max(i*dr,rst)
   print r*phicross3(r, AFe,BFe,ArFe,feFe,BeFe,reFe,raFe,kFe,
                        ANi,BNi,ArNi,feNi,BeNi,reNi,raNi,kNi)

# Pair potential for Fe-Fe
for i in range(nr):
   r = max(i*dr,rst)
   print r*phiJ(r, AFe,ArFe,reFe,kFe,BFe,BeFe,raFe)

