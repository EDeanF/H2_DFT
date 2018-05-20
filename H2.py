"""
solves Schrodinger's Equation for the hydrogen molecule
using Kohn-Sham Equations under LDA
Author: E-Dean Fung (ef2486) Date: Nov 2015

comparison with Hydrogen
potential is multiplied by 2
must distinguish between total density and density of single electron


Naming conventions
WAVE = wavefunction in vector form
WAVEF = wavefunction in functional form
DENS = density function
"""


import os , sys, math, corr, fortsum, numpy as np, matplotlib.pyplot as plt
from scipy import linalg , pi , arange , zeros ,e , sqrt, log

        #=======================#
	# Initialize parameters #
	#=======================#

alp = [0.298073, 1.242567, 5.782948, 38.474970]
A = 0.0311
B = -0.048
C = 0.002
D = -0.0116
gamma = -0.1423
beta1 = 1.0529
beta2 = 0.3334

# define basis

numRpnts = 51
numZpnts = 151
numDpnts = 101
Zmax = 4
Rmax = 2
Dmin = 0.5
Dmax = 2.0

def gaus (x,y,n,m): return e**(-n*(x**2+(y-m)**2))

r = np.linspace(0,Rmax,numRpnts)
z = np.linspace(-Zmax,Zmax,numZpnts)
sep = np.linspace(Dmin,Dmax,numDpnts)
dr = r[1]-r[0]
dz = z[1]-z[0]

	#==================#
	# Define Functions #
	#==================#

# define functions for calculating Hamiltonian

def fn1(n,i,m,j): return n*m*(i-j)**2/(n+m)
def Rp(n,i,m,j): return (n*i+m*j)/(n+m)
def fn2(n,i,m,j,pos): return sqrt((n+m)*abs(Rp(n,i,m,j)-pos)**2)
def fn3(i,a,j,b,k,c,l,d): return sqrt((i+k)*(j+l)/(i+j+k+l)*abs(Rp(i,a,k,c)-Rp(j,b,l,d))**2)
def gaussint(n,i,m,j,pos):
	if i==j and i==pos: return 2/sqrt(pi)
	else: return math.erf(fn2(n,i,m,j,pos))/fn2(n,i,m,j,pos)
def gaussint2(i,a,j,b,k,c,l,d):
	if fn3(i,a,j,b,k,c,l,d)==0: return 2/sqrt(pi)
	else: return math.erf(fn3(i,a,j,b,k,c,l,d))/fn3(i,a,j,b,k,c,l,d)

# define overlap matrix
def ol (n,i,m,j): return ( pi /( n + m ))**1.5*e**(-fn1(n,i,m,j))

# define hamiltonian

def ke (n,i,m,j): return (pi/(m+n))**1.5*e**(-fn1(n,i,m,j))*(n*m/(n+m))*(3-2*fn1(n,i,m,j))
def pel(n,i,m,j): return -pi*sqrt(pi)/(n+m)*e**(-fn1(n,i,m,j))*gaussint(n,i,m,j,Dtemp[0]) 
def per(n,i,m,j): return -pi*sqrt(pi)/(n+m)*e**(-fn1(n,i,m,j))*gaussint(n,i,m,j,Dtemp[1])

# define interaction energy in the basis

def calcIE (i,a,j,b,k,c,l,d):
	return pi**3/((i+k)*(j+l)*sqrt(i+j+k+l))*e**(-fn1(i,a,k,c)-fn1(j,b,l,d))*\
		gaussint2(i,a,j,b,k,c,l,d)

# convert Hartree to eV
def Hart2eV(Hart): return 27.21138602 * Hart

# convert WAVE vector to Density Function

def calcDENS(fbasis,fWAVE): 
	ket = np.einsum('abcd,cd',fbasis,fWAVE)
	bra = np.conj(ket)
	out = 2*np.multiply(bra,ket)
	return out


def expandarray(array):
	dim=array.shape
	out = np.zeros((dim[0],dim[1]*2-1))

	for i in range(0,dim[0]):
		for j in range(0,dim[1]):
			out[i,dim[1]-1+j] = array[i,j]
			out[i,dim[1]-1-j] = array[i,j]
	return out

# calculate Rs
def Dens2Rs(fDENS):
	return np.power((3.0/(4.0*pi))*np.power(fDENS,-1.0),1.0/3.0)

# 3D integration for radially symmetric integrand in cylindrical coordinates
# and mirror symmetry across x-y plane
def int3D(integrand):
	return (2.0*pi*dr*dz)*fortsum.arr(np.multiply(integrand,r))

# define exchange terms

def Ex(fDENS):
	return -3.0/4.0*(3.0/pi)**(1.0/3.0)*int3D(np.power(fDENS,4.0/3.0))

def Vx (fDENS):
	return -1.0*np.power(3.0/pi*fDENS,1.0/3.0)

def calcHx (basis,fDENS):
	out = np.zeros([4,2,4,2])
	for n in range(0,4):
		for i in range(0,2):
			braVECT = np.zeros([4,2])
			braVECT[n,i] = 1
			bra = np.einsum('abcd,cd',basis,braVECT)
			for m in range(0,4):
				for j in range(0,2):
					if ((2*m+j)>=(2*n+i)):
						ketVECT = np.zeros([4,2])
						ketVECT[m,j] = 1
						ket = np.einsum('abcd,cd',basis,ketVECT)
						fVx = Vx(fDENS)
						# perform integral
						out[n,i,m,j] = int3D(np.multiply(bra,np.multiply(fVx,ket)))
					else: out[n,i,m,j] = np.conj(out[m,j,n,i])
	return out

# define correlation terms

#def EPSc(fDENS):
#	Rs = Dens2Rs(fDENS)
#	out = np.zeros([numZpnts,numRpnts])
#	for i in range(0,numZpnts):
#		for j in range(0,numRpnts):
#			if Rs[i,j]>=1:
#				out[i,j] = gamma/(1+beta1*sqrt(Rs[i,j])+beta2*Rs[i,j])
#			else:
#				out[i,j] = A*log(Rs[i,j])+B+Rs[i,j]*C*log(Rs[i,j])+D*Rs[i,j]
#	return out

def Ec(fDENS):
	Rs = Dens2Rs(fDENS)
	fEPSc = corr.eps(Rs)
	return int3D(np.multiply(fDENS,fEPSc))

#def Vc (fDENS):
#	Rs = Dens2Rs(fDENS)
#	out = np.zeros([numZpnts,numRpnts])
#	for i in range(0,numZpnts):
#		for j in range(0, numRpnts):
#			if Rs[i,j]>=1:
#				out[i,j] = gamma*(1.0+7.0/6.0*beta1*sqrt(Rs[i,j])+4.0/3.0*\
#					beta2*Rs[i,j])/((1.0+beta1*sqrt(Rs[i,j])+beta2*Rs[i,j])**2.0)
#			else:
#				out[i,j] = A*log(Rs[i,j])+B-A/3.0+2.0/3.0*Rs[i,j]*C*log(Rs[i,j])+\
#					(2.0*D-C)*Rs[i,j]/3.0
#	return out

def calcHc (basis,fDENS):
	Rs = Dens2Rs(fDENS)
	out = np.zeros([4,2,4,2])
	for n in range(0,4):
		for i in range(0,2):
			braVECT = np.zeros([4,2])
			braVECT[n,i] = 1
			bra = np.einsum('abcd,cd',basis,braVECT)
			for m in range(0,4):
				for j in range(0,2):
					if ((2*m+j)>=(2*n+i)):
						ketVECT = np.zeros([4,2])
						ketVECT[m,j] = 1
						ket = np.einsum('abcd,cd',basis,ketVECT)
						fVc = corr.vc(Rs)
						# perform integral
						out[n,i,m,j] = int3D(np.multiply(bra,np.multiply(fVc,ket)))
					else: out[n,i,m,j] = np.conj(out[m,j,n,i])
	return out

# total energy calculations

def etotHF(feigE,ftemp,fieB): return 2*feigE-np.einsum('a,b,abcd,c,d',ftemp,ftemp,fieB,ftemp,ftemp)

def etotx(feigE, fWAVE, fDENS, fieB, dis):
	out = 2*feigE+Ex(fDENS)+1/dis
	out -= 0.5*np.einsum('ab,cd,abcdefgh,ef,gh',np.conj(fWAVE),np.conj(fWAVE),fieB,fWAVE,fWAVE)
	out -= int3D(np.multiply(Vx(fDENS),fDENS))
	return out

def etotxc(feigE, fWAVE, fDENS, fieB, dis):
	out = 2*feigE+Ex(fDENS)+Ec(fDENS)+1/dis
	out -= 0.5*np.einsum('ab,cd,abcdefgh,ef,gh',np.conj(fWAVE),np.conj(fWAVE),fieB,fWAVE,fWAVE)
	out -= int3D(np.multiply(Vx(fDENS),fDENS))
	out -= int3D(np.multiply(corr.vc(Dens2Rs(fDENS)),fDENS))
        return out

Hx = np.zeros([4,2,4,2])
Hc = np.zeros([4,2,4,2])

# energy for each separation distance
Etotal_noint = np.zeros(numDpnts)
Etotal_x=np.zeros(numDpnts)
Etotal_xc=np.zeros(numDpnts)
# minimum energy
optE_noint = float("inf")
optE_x=float("inf")
optE_xc=float("inf")

for itD in range(0,numDpnts):	# for each value of D
	print 'Calculating Distance # %s' % (itD+1)

	# initialize Hamiltonian	
	Dtemp = [-0.5*sep[itD], 0.5*sep[itD]]
	
	ovl =[[[[ ol (m,i,n,j) for j in Dtemp] for n in alp ] for i in Dtemp] for m in alp ]
	ovl=np.array(ovl).reshape([8,8])
	
	H0 = [[[[ ke(n,i,m,j) + pel(n,i,m,j) + per(n,i,m,j) for j in Dtemp] 
		for m in alp] for i in Dtemp] for n in alp]
		
	ieB =[[[[[[[[ calcIE(i,a,j,b,k,c,l,d) for d in Dtemp]
		for l in alp] for c in Dtemp] for k in alp]
		for b in Dtemp] for j in alp] for a in Dtemp] for i in alp]

	basis = [[[[gaus(x,y,n,m) for m in Dtemp] for n in alp] for x in r] for y in z]

	#=========================#
	# calculate initial guess #
	#=========================#

	H = np.array(H0)
	# solve
	eigE,wavefunc = linalg.eigh( H.reshape([8,8]) , ovl )
	# update
	Etotal_noint[itD]=2*eigE[0]+1/sep[itD]
	if Etotal_noint[itD]<optE_noint: 
		optI_noint = eigE[0]
		optE_noint = Etotal_noint[itD]
		optD_noint = sep[itD]
		optWAVE_noint = np.array(wavefunc[:,0]).reshape([4,2])
		basis_noint = basis
	
	initWAVE = np.array(wavefunc[:,0]).reshape([4,2])
	
	#=========================#
	# LDA without correlation #
	#=========================#

	tempWAVE = initWAVE	
	tempDENS=calcDENS(basis,tempWAVE)

	it=0
	dE=float("inf")
	E1=Etotal_noint[itD]
	while ((it<50) and (dE>1e-11)):
		it+=1
		# calculate Hartree term and Hxc
		Hie = np.einsum('cd,abcdefgh,gh',tempWAVE,np.multiply(ieB,2.0),tempWAVE)
		Hx = calcHx(basis,tempDENS)
		H=np.array(H0+Hie+Hx)
		# solve
		eigE,wavefunc = linalg.eigh( H.reshape([8,8]) , ovl )
		# update
		tempWAVE = np.array(wavefunc[:,0]).reshape([4,2])
		tempDENS = calcDENS(basis,tempWAVE)
		E2=etotx(eigE[0],tempWAVE,tempDENS,np.multiply(ieB,4.0),sep[itD])
		dE=abs(E1-E2)
		E1=E2
	print '\t num of interations is %s and dE is %s' % (it, dE)
	# save to Etotal
	Etotal_x[itD]=E1
	
	if Etotal_x[itD]<optE_x:
		optI_x = eigE[0]
		optE_x = Etotal_x[itD]
		optD_x = sep[itD]
		optWAVE_x = tempWAVE
		basis_x = basis
	
	# check normalization
	#print int3D(calcDENS(basis_x,optWAVE_x))
	
	#======================#
	# LDA with correlation #
	#======================#

	tempWAVE = initWAVE	
	tempDENS=calcDENS(basis,tempWAVE)

	it=0
	dE=float("inf")
	E1=Etotal_noint[itD]
	while ((it<50) and (dE>1e-11)):
		it+=1
		# calculate Hartree term and Hxc
		Hie = np.einsum('cd,abcdefgh,gh',tempWAVE,np.multiply(ieB,2.0),tempWAVE)
		Hx = calcHx(basis,tempDENS)
		Hc = calcHc(basis,tempDENS)		
		H=H0+Hie+Hx+Hc
		# solve
		eigE,wavefunc = linalg.eigh( H.reshape([8,8]) , ovl )
		# update
		tempWAVE = np.array(wavefunc[:,0]).reshape([4,2])
		tempDENS = calcDENS(basis,tempWAVE)
		E2=etotxc(eigE[0],tempWAVE,tempDENS,np.multiply(ieB,4.0),sep[itD])
		dE=abs(E1-E2)
		E1=E2
	print '\t num of interations is %s and dE is %s' % (it, dE)
	# save to Etotal
	Etotal_xc[itD]=E1
	
	if Etotal_xc[itD]<optE_xc:
		optI_xc = eigE[0]
		optE_xc = Etotal_xc[itD]
		optD_xc = sep[itD]
		optWAVE_xc = tempWAVE
		basis_xc = basis

	# check normalization
	#print int3D(calcDENS(basis_xc,optWAVE_xc))

print 'No interaction\n\tequilibrium distance %s with energy %s and ionization energy %s' %(optD_noint, Hart2eV(optE_noint), Hart2eV(optI_noint))

print 'No correlation potential\n\tequilibrium distance %s with energy %s and ionization energy %s' %(optD_x, Hart2eV(optE_x), Hart2eV(optI_x))

print 'With correlation potential\n\tequilibrium distance %s with energy %s and ionization energy %s' %(optD_xc, Hart2eV(optE_xc), Hart2eV(optI_xc))


	#==========#
	# Graphing #
	#==========#

# plot densities

def plot2Ddens(fbasis,fDENS):
	zaxis,raxis = np.mgrid[-Zmax:Zmax+dz:dz,-Rmax:Rmax+2*dr:dr]
	z_min,z_max=0,np.abs(fDENS).max()
	plt.pcolor(raxis,zaxis,fDENS,cmap='Blues',vmin=z_min,vmax=z_max)
	plt.axis([raxis.min(),raxis.max(),zaxis.min(),zaxis.max()])
	plt.colorbar()

DENS_noint = expandarray(calcDENS(basis_noint,optWAVE_noint))
DENS_x  = expandarray(calcDENS(basis_x,optWAVE_x))
DENS_xc = expandarray(calcDENS(basis_xc,optWAVE_xc))

plot2Ddens(basis_noint,DENS_noint)
plt.savefig("DENS_noint")
plt.close()

plot2Ddens(basis_x,DENS_x)
plt.savefig("DENS_x")
plt.close()

plot2Ddens(basis_xc,DENS_xc)
plt.savefig("DENS_xc")
plt.close()

l1, = plt.plot(z, DENS_noint[:,50])
l2, = plt.plot(z, DENS_x[:,50])
l3, = plt.plot(z, DENS_xc[:,50])
plt.legend((l1,l2,l3),('no interaction', 'no correlation', 'with correlation'),loc='upper right', shadow=True)
plt.xlabel('Axial Distance from COM (Bohr)')
plt.ylabel('Density')
plt.title('H2 Molecular Orbital Electron Density')
plt.savefig('DENS_1D')
plt.close()

# plot energy v distance

l1, = plt.plot(sep,Etotal_noint)
l2, = plt.plot(sep,Etotal_x)
l3, = plt.plot(sep,Etotal_xc)
plt.plot(optD_noint,optE_noint, 'o', color='b')
plt.plot(optD_x,optE_x, 'o', color='g')
plt.plot(optD_xc,optE_xc, 'o', color='r')
plt.legend((l1,l2,l3),('no interaction','no correlation','with correlation'),loc='upper right', shadow=True)
plt.xlabel('Distance (Bohr)')
plt.ylabel('Total Energy (Hartree)')
plt.title('Hydrogen Molecule, Bonding Molecular Orbital\nTotal Energy vs Separation Distance')
plt.grid(True)
plt.savefig("EtotalvSep")

