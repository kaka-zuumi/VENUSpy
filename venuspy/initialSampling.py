from ase import units

#from scipy.optimize import minimize, root
from scipy.interpolate import interp1d
from ase.optimize import QuasiNewton, LBFGS
from ase.vibrations import Vibrations
import random

import numpy as np

########################################################################################

class initialSampling:

    # Some conversion constants
    kB = 8.617281e-5                  # eV
    freq2energy = 1.000               # cm^-1
    rotConstant2energy = 0.00209008   # eV

    # Distance to separate the centers of mass of
    # the two molecules so that hey have little
    # to no interactions (in Angstrom)
    dSEPARATED = 200.0  # 200.0

    # A larger number of points will make the
    # semiclassical sampling converge smoother
    NorderPolynomial = 50    # 100

    # Threshold for error in rovibrational energy
    # (in quanta) for the semiclassical sampling
    Nvib_error_threshold = 1.0e-1 # 1.0e-5

    # If "debug" is true, then more information
    # is printed out during the sampling
    debug=True

    def __init__(self,mol,atomsInFirstGroup,optimize=False,optimization_file="optimization.traj",
                      samplingMethodA="thermal",vibrationalSampleA=298,rotationalSampleA=298,
                      samplingMethodB="thermal",vibrationalSampleB=298,rotationalSampleB=298):

        self.mol = mol
        self.optimize = optimize
        self.optimization_file = optimization_file

        self.samplingMethodA = samplingMethodA
        self.vibrationalSampleA = vibrationalSampleA
        self.rotationalSampleA = rotationalSampleA
        self.samplingMethodB = samplingMethodB
        self.vibrationalSampleB = vibrationalSampleB
        self.rotationalSampleB = rotationalSampleB

        self.Natoms = len(self.mol)
        self.atomsInFirstGroup = atomsInFirstGroup
        self.atomsInSecondGroup = []
        for i in range(self.Natoms):
            if (i not in atomsInFirstGroup): self.atomsInSecondGroup.append(i)

    ##########################################################################################################

    # First, basic functions are defined below:

    # Get the kinetic energy of the molecule
    def getKineticEnergy(self,m,p):
        KE = 0.0
        for i in range(len(m)):
            KE += np.inner(p[i],p[i])/m[i]

        return 0.5e0 * KE

    # Separate the two molecules
    def separateMolecules(self):
        masses = self.mol.get_masses()
        q = self.mol.get_positions()
        if (len(self.atomsInFirstGroup) > 0):
            mass1 = sum([masses[i] for i in self.atomsInFirstGroup])
            r1 = sum([masses[i]*q[i] for i in self.atomsInFirstGroup]) / mass1
        else:
            r1 = np.array([0.0, 0.0, 0.0])
        if (len(self.atomsInSecondGroup) > 0):
            mass2 = sum([masses[i] for i in self.atomsInSecondGroup])
            r2 = sum([masses[i]*q[i] for i in self.atomsInSecondGroup]) / mass2
        else:
            r2 = np.array([0.0, 0.0, 0.0])

        for i in range(self.Natoms):
            if i in self.atomsInFirstGroup:
                q[i] -= r1
#               q[i][0] += self.dSEPARATED
                q[i][0] += 2*self.dSEPARATED    # Temporary fix to make parsing NWChem output easier
            else:
                q[i] -= r2
#               q[i][0] -= self.dSEPARATED
                q[i][0] -= 0.0                  # Temporary fix to make parsing NWChem output easier
    
        self.mol.set_positions(q)
    
    # Get the center-of-mass of the molecule
    def getCenterOfMass(self,m,q):
        qCM = np.zeros((3))
        for i in range(len(m)):
            qCM += m[i]*q[i]
    
        return qCM / sum(m)
    
    # Place the center-of-mass of the molecule at the origin
    def centerMolecule(self,reactantIndexes):
        masses = self.mol.get_masses()
        q = self.mol.get_positions()
        r = self.getCenterOfMass(masses[reactantIndexes],q[reactantIndexes])
        for i in range(self.Natoms):
            if i in reactantIndexes:
                q[i] -= r
    
        self.mol.set_positions(q)
    
    # Rotate the molecule's coordinates by a rotationa matrix
    def rotateMolecule(self,reactantIndexes,rotationMatrix):
        q = self.mol.get_positions()
        p = self.mol.get_momenta()
        for i in reactantIndexes:
            q[i] = np.dot(rotationMatrix.transpose(),q[i])
            p[i] = np.dot(rotationMatrix.transpose(),p[i])
    
        self.mol.set_positions(q)
        self.mol.set_momenta(p)
    
    # Rotate a set of normal modes or velocities by a rotation matrix
    def rotateNormalModes(self,nmodes,rotationMatrix):
        newmodes = np.copy(nmodes)
        for i in range(len(nmodes)):
            nmode = nmodes[i]
            for Natom in range(len(nmode)):
                newmodes[i][Natom] = np.dot(rotationMatrix.transpose(),nmode[Natom])
    
        return newmodes
    
    # Calculate the moment of inertia tensor from the
    # mass, m, and coordinates, q
    def getMomentOfInertia(self,m,q):
        I = np.zeros((3,3))
        for i in range(len(m)):
            qi = q[i]
            I += m[i] * (np.diag(np.tile(np.inner(qi,qi),3)) - np.outer(qi,qi))
    
        return I
    
    # Do an SVD to get the principal axes of a 3x3 matrix
    def getPrincipalAxes(self,I):
        U, S, VH = np.linalg.svd(I)
    
        return S, U
    
    # Choose a random rotation in 3D space
    def chooseRandomSpatialRotation(self):
    
        # First, get a random unit vector for the x-axis
        while True:
            u = np.array([random.random(), random.random(), random.random()])
            u -= 0.5e0
            u2 = sum(u**2)
            if (u2 > 1.0e-4):
                u = u / np.sqrt(u2)
                break
    
        # Second, get a random unit vector for the y-axis
        # orthogonal to the already-chosen x-axis
        while True:
            v = np.array([random.random(), random.random(), random.random()])
            v -= 0.5e0
            v = np.cross(u,v)
            v2 = sum(v**2)
            if (v2 > 1.0e-4):
                v = v / np.sqrt(v2)
                break
    
        # Third, define the z-axis to be the unit vector
        # orthogonal to both the x- and y-axis, with the
        # sign chosen randomly
        w = np.cross(u,v)
        if (random.random() > 0.5e0):
            w = -w
    
        # Three orthogonal unit vectors are one way to
        # construct a random rotation matrix
        return np.array([u, v, w])

    # Choose a random axis to spin the molecules, and also
    # consider the principal axes if the molecule is linear
    # Output the axis as a normed vector or "unit"
    def chooseRandomUnitAngularVelocity(self,Nlinear,principalAxes):
        if (Nlinear):
            u = 2 * np.pi * random.random()
            a = principalAxes[0] * np.sin(u) + principalAxes[1] * np.cos(u)
        else:
            x = random.random()
            y = random.random()
            z = random.random()
            a = np.array([x,y,z])
        return a / np.norm(a)
    
    # Calculate angular speed given the unit angular velocity,
    # moment of inertia tensor, and the required amount of
    # rotational energy
    def getAngularSpeed(self,unitOmega,I,RE):
        REunit = np.inner(unitOmega,np.dot(I,unitOmega))
        return np.sqrt(2*RE/REunit)
    
    # Calculate the angular momentum from the (relative)
    # positions and momenta
    def getAngularMomentum(self,q,p):
        L = np.zeros((3))
        for i in range(len(q)):
            L += np.cross(q[i],p[i])
    
        return L
    
    # Assuming the system is rigid, return the linear momenta
    # which correspond to the given input angular velocity
    def getMomentaFromAngularVelocity(self,m,q,omega):
        p = np.zeros(np.shape(q))
        for i in range(len(m)):
            p[i] = m[i]*np.cross(omega,q[i])
    
        return p
    
    # Calculate the rotational energy and angular velocity, 
    # omega, of the system
    def getErotAndOmegaFromL(self,m,q,L):
        I = self.getMomentOfInertia(m,q)
    
        # The singular value decomposition will be able to
        # ignore essentially-zero components of the moment
        # of inertia tensor (e.g., for linear molecules) and
        # allow the inverse to be taken easily
        U, S, VH = np.linalg.svd(I)
        Sinv = np.diag(S)
        for i in range(3):
            if (Sinv[i,i] > 1.0e-12):
                Sinv[i,i] = 1.0e0 / Sinv[i,i]
            else:
                Sinv[i,i] = 0.0e0
    
        omega = np.dot(VH.transpose() @ Sinv @ U.transpose(), L)
        Erot = 0.5e0*np.dot(omega,L)
    
        return Erot, omega
    
    # Calculate the rotational energy and angular velocity, 
    # omega, of the system
    def getErotAndOmegaFromP(self,m,q,p):
        L = self.getAngularMomentum(q,p)
        Erot, omega = self.getErotAndOmegaFromL(m,q,L)
    
        return Erot, omega

    ##########################################################################################################

    # Calculate Gauss-Legendre x-values and weights (x,w(x))
    def gaussLegendre(self,xLEFT,xRIGHT,n,xx=None):
    
        # First, get the roots of the Gauss-Legendre
        # polynomial, if they were not given
        m = int((n+1)/2)
        if (xx == None):
            xx = np.zeros((m))
            df = np.zeros((m))
            for i in range(m):
                j = i + 1
                xx[i] = np.cos(np.pi*(j - 0.25e0)/(n+0.50e0))
    
                # Find the root with the
                # Newton-Raphson formula
                f = 1.0e0
                while (abs(f) > 1.0e-13):
                    f, df[i] = self.legendrePolynomial(xx[i],n)
                    xx[i] = xx[i] - f / df[i]
    
        # Finally, get the points (x,w(x)) at
        # which the polynomial is evaluated
    
        x = np.zeros((n))
        w = np.zeros((n))
    
        xm = 0.5e0 * (xRIGHT + xLEFT)
        xl = 0.5e0 * (xRIGHT - xLEFT)
        for i in range(m):
            x[i]   = xm - xl * xx[i]
            x[(n-1)-i] = xm + xl * xx[i]
            w[i]   = (xl / ((1.0e0 - xx[i]*xx[i]) * df[i] * df[i]) ) / 0.5e0
            w[(n-1)-i] = w[i]
    
        return x, w
    
    # Calculate the legendre polynomial of
    # order n at x, and its derivative
    def legendrePolynomial(self,x,n):
        p2 = x
        pl = 1.5e0 * x * x - 0.5e0
        for k in range(2,n):
            p1 = p2
            p2 = pl
            pl = float((2*k+1) * p2 * x - k * p1) / (k+1)
    
        dpl = n * (x * pl - p2) / (x * x - 1.0e0)
    
        return pl, dpl
    
    ##########################################################################################################

    # Took this from: XXXXXXX
    def beyer_swinehart(self, freq, egrain, emax, count='dos', interp=False):
      """Beyer-Swinehart state count

      Args:
        freq (array): array of frequencies
        egrain (int): energy grain (same unit as freq)
        emax (int): maximum energy (same unit as freq)
        count (str, optional): Count type - dos or sos. Defaults to 'dos'.
        interp (bool, optional): Interpolate to make smoother. Defaults to False.

      Returns:
        array: density or sum of states
      """

      # Initialize values for counting (reduce to integers divided by egrain)
      emax = int(round(emax/egrain))
      freq = abs(np.array(freq)) # make all values positive
      if type(freq) == np.float64: # to deal with only one provided frequency value
        freq = [freq]
      # For high egrain values, should use a multiplier for correcting the initial SOS array
      ini_multiplier = egrain//freq[0]
      if ini_multiplier == 0:
        ini_multiplier = 1
      # Set up frequencies
      freq = [1 if f<egrain else int(round(f/egrain)) for f in freq]
      # Set up DOS or SOS count
      if count == 'dos':
        bs = np.zeros(emax)
        bs[0] = 1
      elif count == 'sos':
        bs = np.ones(emax) * ini_multiplier
#       ns = np.zeros((len(freq),emax))
      # Count the states
      for i in range(0,len(freq)):
        for j in range(freq[i],emax):
          bs[j] += bs[j-freq[i]]
#         ns[i][j] += bs[j-freq[i]]
      # Interpolate if toggled
      if interp:
        bs[bs < 1] = 0
        idx = np.nonzero(bs)
        x = np.arange(len(bs))
        interp = interp1d(x[idx],bs[idx],fill_value='extrapolate')
        bs = interp(x)
      # Divide DOS by egrain to get DOS per 1 energy unit
      if count == 'dos':
        bs[1:] /= egrain
#     return bs, ns
      return bs

    ##########################################################################################################
    
    # Find out the "turning points" or the bond
    # lengths which are accessible with this
    # vibrational and rotational energy
    def getRovibrationalRange(self,reactantIndexes,E,angularMomentum,V0):
    
        # Initialize some information about the diatom
        L2 = angularMomentum**2
        m = self.mol.get_masses()[reactantIndexes]
        mu  = m[0] * m[1] / (m[0] + m[1])
        qBOTH = self.mol.get_positions()
        q = qBOTH[reactantIndexes]
        bondLength = np.sqrt(sum((q[1]-q[0])**2))
    
        # If the bond length at the energy minimum is too
        # large, print this warning
        if (bondLength > 6):
            raise ValueError("In FINLNJ: bondLength > 6 for diatom sampling")
    
        # Position the diatom so that the bond is along
        # the z-axis; place the atoms so that (assuming
        # the two reactants started separate) the two
        # reactants stay separate
        qCM = self.getCenterOfMass(m,q)
        q[0] = qCM
        q[0][2] -= 0.5e0 * bondLength
        q[1] = qCM
        q[1][2] += 0.5e0 * bondLength
    
        # Calculate the energy; only do this when the two
        # reactants are separate
        qBOTH[reactantIndexes] = q
        self.mol.set_positions(qBOTH)
        V = self.mol.get_potential_energy()
    
        # Calculate Veff, the rotational + vibrational
        # energy
        Erot0 = L2 / (2 * mu * (bondLength**2))
        Veff = (V - V0) + Erot0
    
        if (self.debug):
            print("In FINLNJ: V-V0 = ", (V-V0), "E(not V-V0) = ", Erot0)
            print("In FINLNJ: Veff = ", Veff, "   E = ", E)
    
        # If the energy is ALREADY above that required
        # for the system, then print this warning
        if (Veff > E):
#           raise ValueError("In FINLNJ: Veff > E for diatom sampling")

            # temporary addition:
            Nvib = -0.5e0
            return Nvib, bondLength, bondLength
    
        # If there are no problems, go on ahead with
        # integrating the energy
    
        # First, define the boundaries of the integral
        # as the bond lengths which just are just 
        # barely going over the required energy E
    
        # Find the upper bound
        q[1][2] = q[0][2] + bondLength
        while True:
    
            # Increment the bond by values of 0.001 A
            q[1][2] += 0.001e0
    
            # Evaluate the new energy and Veff
            qBOTH[reactantIndexes] = q
            self.mol.set_positions(qBOTH)
            V = self.mol.get_potential_energy()
    
            newBondLength = q[1][2] - q[0][2]
            Veff = (V - V0) + L2 / (2 * mu * (newBondLength**2))
     
            # Eventually, the potential energy should
            # rise enough to get above the threshold
            if (Veff >= E or newBondLength > 50):
                break
    
        rMAX = newBondLength
        if (self.debug): print("In FINLNJ: rMAX(A) = ", rMAX)
    
        # Find the lower bound
        q[1][2] = q[0][2] + bondLength
        while True:
    
            # Decrement the bond by values of 0.001 A
            q[1][2] -= 0.001e0
    
            # Evaluate the new energy and Veff
            qBOTH[reactantIndexes] = q
            self.mol.set_positions(qBOTH)
            V = self.mol.get_potential_energy()
    
            # Eventually, the potential energy should
            # rise enough to get above the threshold
            newBondLength = q[1][2] - q[0][2]
            Veff = (V - V0) + L2 / (2 * mu * (newBondLength**2))
    
            if (Veff >= E):
                break
    
        rMIN = newBondLength
        if (self.debug): print("In FINLNJ: rMIN(A) = ", rMIN)
    
        # Prepare the bond lengths for which to
        # integrate the energy over

#       r, w = GLPAR(rMIN,rMAX,NorderPolynomial)
        r, w = self.gaussLegendre(rMIN,rMAX,self.NorderPolynomial)
    
        # Next, take the integral by evaluating the
        # energy on these points
        Asum = 0.0e0
        for j in range(self.NorderPolynomial):
            newBondLength = r[j]
    
            q[1][2] = q[0][2] + newBondLength
            qBOTH[reactantIndexes] = q
            self.mol.set_positions(qBOTH)
            V = self.mol.get_potential_energy()
    
            Veff = (V - V0) + L2 / (2 * mu * (newBondLength**2))
    
            # If some of the points are above the
            # integral (should only be the endpoints)
            # don't add them into the integral
            if (E > Veff):
                Asum += w[j] * np.sqrt(E-Veff)
            else:
                if (self.debug): print("In FINLNJ: see point with E<Veff (r(A),E(kcal/mol),Veff(kcal/mol)) = ", (newBondLength, E, Veff))
    
        if (self.debug): print("In FINLNJ: Integral = ", Asum)
    
        # Return the molecule to its minimum
        # energy configuration for now
        q[1][2] = q[0][2] + bondLength
        qBOTH[reactantIndexes] = q
        self.mol.set_positions(qBOTH)
    
        # The "expected" vibrational quantum number
        # corresponding to this energy can then be
        # computed to see if it agrees with the
        # value requested
        Nvib = np.sqrt(8.0e0 * mu) * Asum / (2*np.pi* np.sqrt(2*self.rotConstant2energy))
        Nvib = Nvib - 0.5e0
    
        return Nvib, rMIN, rMAX
    
    # Sample a diatomic molecule's acceptable range
    # of bond lengths for the specified quantum numbers
    def getDiatomBondLengthRangeWithEBK(self,reactantIndexes,Nrot0,Nvib0,Evib0,V0):
    
        # Initialize some information about the diatom
        m = self.mol.get_masses()[reactantIndexes]
        mu  = m[0] * m[1] / (m[0] + m[1])
        q = self.mol.get_positions()[reactantIndexes]
        bondLength = np.sqrt(sum((q[1]-q[0])**2))
    
        hnu = Evib0 / (Nvib0 + 0.5e0)
        AM0  = (np.sqrt(2 * self.rotConstant2energy)) * np.sqrt(float(Nrot0*(Nrot0+1)))

        if (self.debug):
            print("AM from EBK:", AM0)
            print("R0: ", bondLength, "I=mu*R0^2: ", mu*(bondLength**2), "Erot: ", (AM0**2)/(2*mu*(bondLength**2)))
    
        # Only accept the range of bond lengths
        # found by the algorithm "FINLNJ" if its
        # expected vibrational quantum number
        # agrees with the requested one
        Ntries = 0
        Erovib = Evib0
        Nvib_error = 1.0e0
        while (abs(Nvib_error) > self.Nvib_error_threshold):
            Nvib, rMIN, rMAX = self.getRovibrationalRange(reactantIndexes,Erovib,AM0,V0)
            Nvib_error = Nvib0 - Nvib
            Erovib = Erovib + Nvib_error * hnu
    
            if (self.debug): print("Nvib0:",Nvib0, "Nvib:",Nvib, "Nvib_error:",Nvib_error)
    
            Ntries += 1
            if (Ntries > 200):
                raise ValueError("In getDiatomBondLengthRangeWithEBK: Ntries for diatom sampling above 200")
    
        # Make sure not to include the turning points themselves
        # in the subsequent distance scans
        rMIN += 0.001e0
        rMAX -= 0.001e0
    
        pTEST = np.sqrt(2 * mu * Erovib * 0.0001e0)
    
        return rMIN, rMAX, AM0, Erovib, pTEST
    
    def chooseQPforDiatom(self,reactantIndexes,Nrot,Nvib,freq):
    
        qBOTH = self.mol.get_positions()
        m = self.mol.get_masses()[reactantIndexes]
        mu  = m[0] * m[1] / (m[0] + m[1])
        q = qBOTH[reactantIndexes]
        qCM = self.getCenterOfMass(m,q)
    
        if (self.debug): print("Nvib:", [Nvib])
    
        # Get the energy at this optimized structure;
        # this MUST be a energy minimum for this program
        # to work smoothly
        self.separateMolecules()
        V0 = self.mol.get_potential_energy()
        if (self.debug): print("V0:",V0)
    
        # Get the "turning points" or the bounds for
        # the bond length
        Evib = freq * (Nvib + 0.5e0)
        rMIN, rMAX, AM, Erovib, pTEST = self.getDiatomBondLengthRangeWithEBK(reactantIndexes,Nrot,Nvib,Evib,V0)

        if (self.debug):
            print("Evib: ", Evib, "Erovib: ", Erovib)
            print("Rmin: ", rMIN, "Rmax: ", rMAX)

        ErotR2 = (AM**2) / (2*mu)
    
        # Try out a lot of different positions for the
        # atoms in space; stop when the energy is
        # nearly correct
        while True:
            u = random.random()
            r = rMIN + (rMAX - rMIN) * u
    
            q[0] = qCM
            q[0][2] -= 0.5e0 * r
            q[1] = qCM
            q[1][2] += 0.5e0 * r
    
            qBOTH[reactantIndexes] = q
            self.mol.set_positions(qBOTH)
            V = self.mol.get_potential_energy()
    
            Vdiff = (V - V0)
            Ediff = Erovib - ((ErotR2 / (r**2)) + Vdiff)
    
            # This case should occur very rarely if at all...
            # make sure to set Rmin and Rmax correctly to avoid this
            # (so that the MC sampling is correct)
            if (Ediff <= 0.0e0):
                if (self.debug): print("initQP diatom iteration.... ACCEPTED for (r,Ediff) = ", (r,Ediff))
                Ediff = 0.0e0
                PR = 0.0e0
                break
    
            else:
                PR = np.sqrt(2.0e0*mu*Ediff)
                Pkinetic = pTEST/PR
    
                u = random.random()
                if (Pkinetic < u):
                    if (self.debug): print("initQP diatom iteration.... REJECTED for (Kvib>0) (r,Ediff,Pkin) = ", (r,Ediff,Pkinetic))
                    continue
    
                if (self.debug): print("initQP diatom iteration.... ACCEPTED for (Kvib>0) (r,Ediff,Pkin) = ", (r,Ediff,Pkinetic))
                break
    
        # Determine whether to spin clockwise or anticlockwise
        u = random.random()
        if (u < 0.5e0):
            PR = -PR
    
        # Now, give the diatom momenta
        pBOTH = self.mol.get_momenta()
        p = pBOTH[reactantIndexes]
        p = np.zeros(p.shape)
    
        # First, give it vibrational momenta
        vrel = PR / mu
        vel1 = vrel * m[1] / (m[0] + m[1])
        vel2 = vel1 - vrel
        p[0][2] = m[0] * vel1
        p[1][2] = m[1] * vel2
    
        # Second, get a random rotation axis
        u = 2*np.pi * random.random()
        L = np.zeros((3))
        L[0] = AM*np.sin(u)
        L[1] = AM*np.cos(u)
        Ixy = mu * r * r
    
        # Third, give it rotational momenta
        q = np.zeros(np.shape(q))
        q[0][2] -= r * m[1] / (m[0] + m[1])
        q[1][2] += r * m[0] / (m[0] + m[1])
        omega = np.zeros((3))
        omega[0] = -L[0] / Ixy
        omega[1] = -L[1] / Ixy
        p += self.getMomentaFromAngularVelocity(m,q,omega)
    
        if (self.debug):
            print("initQP Evib and potEvib: ", Evib, Vdiff)
            print("initQP Erot and omega: ", self.getErotAndOmegaFromP(m,q,p))
    
        # Set these positions and momenta
        qBOTH[reactantIndexes] = q
        self.mol.set_positions(qBOTH)
    
        pBOTH[reactantIndexes] = p
        self.mol.set_momenta(pBOTH)
    
        # Finally, separate the molecules again
        self.separateMolecules()
    
        return
    
    ##########################################################################################################

    # Choose a total angular momentum from a thermal
    # distribution, similar to that in: (Bunker,1973)
    def chooseAngularMomentumFromSymmetricTopThermalDistribution(self,linear,axesMasses,T):
    
        Erot = 0.0e0
        L = np.zeros((3))
    
        if (not linear):
    
            # Look at the difference between the "axesMasses" or the
            # moment of inertia principal components to see which
            # axes are favored to be "spun on" as Lz
            dI12 = axesMasses[0]-axesMasses[1]
            dI23 = axesMasses[1]-axesMasses[2]
            if (dI12 <= dI23):
                zAxis=2
            else:
                zAxis=0
    
            # Define a maximum L (idk why this value in particular)
            LzMAX = np.sqrt(20.0e0*axesMasses[zAxis]*self.kB*T)
    
            # Do rejection sampling to determine Lz
            while True:
                u = random.random()
                L[zAxis] = u * LzMAX
                probL = np.exp(-(L[zAxis]**2)/(2*axesMasses[zAxis]*self.kB*T))
                u = random.random()
                if (u <= probL): break
    
            # Finally, flip a coin to determine the sign of Lz
            u = random.random()
            if (u > 0.5e0): L[zAxis] = -L[zAxis]
    
            # Calculate the Erot contribution from this
            Erot = (L[zAxis]**2)/axesMasses[zAxis]
    
        # Determine sampling for the non-z axes
    
        # Here, the z-axis is the first spatial dimension
        if (linear or (zAxis==0)):
    
            # Use the inverse CDF to determine Lxy
            u = random.random()
            Ixy = np.sqrt(axesMasses[1]*axesMasses[2])
            Lxyz = np.sqrt(L[0]**2 - 2*Ixy*self.kB*T*np.log(1.0e0-u))
            Lxy = np.sqrt(Lxyz**2 - L[0]**2)
    
            # Determine a random phase for the x-y partitioning
            # of the Lxy component of the angular momentum
            u = 2 * np.pi * random.random()
            L[1] = Lxy*np.sin(u)
            L[2] = Lxy*np.cos(u)
    
            # Calculate the Erot contribution from this
            Erot = 0.5e0*(Erot + ((L[1]**2)/axesMasses[1]) + ((L[2]**2)/axesMasses[2]))
    
        # Here, the z-axis is the third spatial dimension
        else:
    
            # Use the inverse CDF to determine Lxy
            u = random.random()
            Ixy = np.sqrt(axesMasses[0]*axesMasses[1])
            Lxyz = np.sqrt(L[2]**2 - 2*Ixy*self.kB*T*np.log(1.0e0-u))
            Lxy = np.sqrt(Lxyz**2 - L[2]**2)
    
            # Determine a random phase for the x-y partitioning
            # of the Lxy component of the angular momentum
            u = 2 * np.pi * random.random()
            L[0] = Lxy*np.sin(u)
            L[1] = Lxy*np.cos(u)
    
            # Calculate the Erot contribution from this
            Erot = 0.5e0*(Erot + ((L[0]**2)/axesMasses[0]) + ((L[1]**2)/axesMasses[1]))
    
        if (self.debug): print("AM from thermal:", np.sqrt(sum(L**2)))
    
        return Erot, L
    
    # Choose vibrational quantum numbers from a thermal
    # distribution
    def chooseVibrationalQuantaFromThermalDistribution(self,freqs,T):
    
        vibNums = []
        for freq in freqs:
            u = random.random()
            n = np.floor(-np.log(u)*T*self.kB/freq)
            vibNums.append(n)
    
        return np.array(vibNums)
    
    # Calculate the vibrational energy contribution of each
    # normal mode, and then compute the amplitude of that
    # mode's harmonic-oscillator-like spatial displacement
    def getVibrationalEnergiesAndAmplitudes(self,freqs,vibNums):
    
        Nmodes = len(freqs)
        Evibs = []
        amplitudes = []
        for i in range(Nmodes):
            Evib = (0.5+vibNums[i])*freqs[i]
            Evibs.append(Evib)
            amplitudes.append(np.sqrt(2*Evib)/freqs[i])
    
        return np.array(Evibs), np.array(amplitudes)
    
    # After choosing an angular momentum, the frequencies,
    # normal modes, and vibrational quanta, choose a set
    # of initial coordinates
    def chooseQPgivenNandLandNormalModes(self,reactantIndexes,vibNums,L0,freqs,nmodes):
    
        massesBOTH = self.mol.get_masses()
        qBOTH = self.mol.get_positions()
        pBOTH = self.mol.get_momenta()
    
        q0 = qBOTH[reactantIndexes]
        masses = massesBOTH[reactantIndexes]
        qCM = self.getCenterOfMass(masses,q0)
    
        if (self.debug): print("reactant indexes:", reactantIndexes)
    
        # Get the energy at this optimized structure;
        # this MUST be a energy minimum for this program
        # to work smoothly
        self.separateMolecules()
        E0 = self.mol.get_potential_energy()
        self.centerMolecule(reactantIndexes)
        if (self.debug): print("E0:",E0)
    
        # Now, get the corresponding vibrational and
        # rotational energies
        Evibs, amplitudes = self.getVibrationalEnergiesAndAmplitudes(freqs,vibNums)
        Evib0 = sum(Evibs)
        Erot0, omega0 = self.getErotAndOmegaFromL(masses,q0-qCM,L0)
    
        Eint0 = Evib0 + Erot0
        if (self.debug): print("Evib0,Erot0,Eint0:",Evib0,Erot0,Eint0)
    
        Nmodes = len(freqs)
        Natoms = len(reactantIndexes)
        NpureKinetic=0
    
        dq = np.zeros((Nmodes))
        dp = np.zeros((Nmodes))
        while True:
    
            # Fill up some normal modes with purely
            # kinetic energy
            for i in range(NpureKinetic):
                dq[i] = 0.0e0
                dp[i] = -freqs[i]*amplitudes[i]
    
            # Fill up the other normal modes with a mix
            # of kinetic and potential energy
            for i in range(NpureKinetic,Nmodes):
                u = 2 * np.pi * random.random()
                dq[i] = amplitudes[i]*np.cos(u) / (8065.5401*1.8836518e-3)
                dp[i] = -freqs[i]*amplitudes[i]*np.sin(u)
    
            # Modify the original optimized structure
            # with the perturbations
            q = np.copy(q0)
            p = np.zeros(np.shape(q))
            for i in range(Nmodes):
                q += nmodes[i] * dq[i]
                p += nmodes[i] * dp[i]
    
            for i in range(Natoms):
                p[i] = p[i] * masses[i]
    
            qCM = self.getCenterOfMass(masses,q)
    
            # Calculate the new (probably different)
            # angular momentum
            L = self.getAngularMomentum(q-qCM,p)
    
            # Comput the "difference" in angular
            # momentum and velocity
            Ldiff = L - L0
            Erot, omega = self.getErotAndOmegaFromL(masses,q-qCM,Ldiff)
    
            # Use this angular momentum difference to
            # adjust linear momenta for an accurate 
            # angular momentum and rotational energy
            pdiff = self.getMomentaFromAngularVelocity(masses,q-qCM,omega)
            p -= pdiff
    
            # Compute the new total potential and kinetic
            # energies of the system
    
            qBOTH[reactantIndexes] = q
            self.mol.set_positions(qBOTH)
            KE = self.getKineticEnergy(masses,p)
    
            self.separateMolecules()
            E = self.mol.get_potential_energy() + KE
            self.centerMolecule(reactantIndexes)
    
            # The internal energy of the molecule of interest
            # will be the difference between the total energy
            # and the reference energy from earlier (when it
            # is at the local minimum)
            Eint = E - E0
    
            # See how close this internal energy is to the
            # required amount
            fitnessOfEint = abs(Eint0-Eint)/Eint0
    
            qCM = self.getCenterOfMass(masses,q)
            L = self.getAngularMomentum(q-qCM,p)
            Erot, omega = self.getErotAndOmegaFromL(masses,q-qCM,L)

            if (self.debug):
                print("KErot: ", Erot, " KEvib: ", KE-Erot)
                print(NpureKinetic,fitnessOfEint,Eint0,Eint)
    
            # First, if all of the internal energy is rotational,
            # then do not attempt any scaling and just exit
            if (Evib0 < 1.0e-4*Eint0): break
    
            # If the internal energy is not that close, then just
            # convert one of the normal modes to being purely
            # kinetic
            if (fitnessOfEint >= 0.1e0 and NpureKinetic < Nmodes):
                NpureKinetic += 1
                continue
    
            # If the internal energy is close to that required,
            # try scaling it
            Nscale = 0
            #while (Nscale < 1000 and fitnessOfEint >= 0.001e0):
            while (Nscale < 50 and fitnessOfEint >= 0.001e0):
                scalingFactor = np.sqrt(Eint0/Eint)
                p = p * scalingFactor
                q = q0 + (q - q0) * scalingFactor
    
                qCM = self.getCenterOfMass(masses,q)
    
                L = self.getAngularMomentum(q-qCM,p)
    
                Ldiff = L - L0
                Erot, omega = self.getErotAndOmegaFromL(masses,q-qCM,Ldiff)
    
                pdiff = self.getMomentaFromAngularVelocity(masses,q-qCM,omega)
                p -= pdiff
    
                qBOTH[reactantIndexes] = q
                self.mol.set_positions(qBOTH)
                KE = self.getKineticEnergy(masses,p)
    
                self.separateMolecules()
                E = self.mol.get_potential_energy() + KE
                self.centerMolecule(reactantIndexes)
    
                Eint = E - E0
           
                fitnessOfEint = abs(Eint0-Eint)/Eint0
    
                qCM = self.getCenterOfMass(masses,q)
                L = self.getAngularMomentum(q-qCM,p)
                Erot, omega = self.getErotAndOmegaFromL(masses,q-qCM,L)

                if (self.debug):
                    print("Erot:",Erot)
                    print(fitnessOfEint,Eint0,Eint)
                    print(q)
    
                if (fitnessOfEint >= 0.1e0 and NpureKinetic < Nmodes):
                    NpureKinetic += 1
                    break
    
                Nscale += 1
    
            if (fitnessOfEint < 0.001e0): break
    
        pBOTH[reactantIndexes] = p
        self.mol.set_momenta(pBOTH)
    
    
    # Make a function that will set up the two molecules' internal positions
    # and momenta including their rotational and vibrational energies
    def sampleRelativeQP(self):

        # If the molecules are not aleady optimized, then they
        # can be done here at the start
        if (self.optimize):
    
            # Move the two groups FAR from one another
            self.separateMolecules()
    
            # Optimize the two molecules with ASE
#           optimizer = QuasiNewton(
#               self.mol,maxstep=0.0100,    # original NWChemEx value = 0.0250,
            optimizer = LBFGS(
                self.mol,    # original NWChemEx value = 0.0250,
                trajectory=self.optimization_file,
            )

            # The convergence threshold is in the default units (eV and Ang)
#           optimizer.run(1.0e-3, 10000)     # original value = 5.0e-3
#           optimizer.run(4.0e-3, 10000) 
            optimizer.run(1.0e-2, 10000) 
#           optimizer.run(1.8e-2, 10000) 
#           optimizer.run(1.0e-2, 10000) 
    
        ###############################################################################
    
        Nreactant = 0
        reactantGroups = [self.atomsInFirstGroup, self.atomsInSecondGroup]
        samplingMethod = [self.samplingMethodA, self.samplingMethodB]
        rotationSampling = [self.rotationalSampleA, self.rotationalSampleB]
        vibrationSampling = [self.vibrationalSampleA, self.vibrationalSampleB]
        for reactantIndexes in reactantGroups:
    
            Nreactant += 1
            if (self.debug):
                print("##############################################################")
                print("Initializing reactant group ", Nreactant)
                print("   with atomic indexes:", reactantIndexes)

            if (len(reactantIndexes) == 0):
                print("No atoms in reactant group ", Nreactant)
                print("Must be unimolecular! Skipping sampling for this group...")
                continue
    
            # You must center the molecule first before calculating
            # its moment of inertia tensor
            self.centerMolecule(reactantIndexes)
            relativeQ0 = self.mol.get_positions()
            masses = self.mol.get_masses()
    
            I = self.getMomentOfInertia(masses[reactantIndexes],relativeQ0[reactantIndexes])
            axesMasses, axesVectors = self.getPrincipalAxes(I)
            axesMasses  = np.array([axesMasses[2], axesMasses[1], axesMasses[0]] )   # Switch axes so that they are in INCREASING order instead
#           axesVectors = np.array([axesVectors[2],axesVectors[1],axesVectors[0]])   # Switch axes so that they are in INCREASING order instead
            Ix,Iy,Iz = np.copy(axesVectors[:,2]), np.copy(axesVectors[:,1]), np.copy(axesVectors[:,0]) 
            axesVectors[:,0], axesVectors[:,1], axesVectors[:,2] = Ix, Iy, Iz        # Switch axes so that they are in INCREASING order instead
    
            if (self.debug):
                print("I:", I)
                print("axesMasses ... axesVectors")
                for i in range(3):
                    print(axesMasses[i]," ... ",axesVectors[i])
    
            # Determine whether the molecules is linear by looking at the smallest principal axis
#           if (axesMasses[0] < axesMasses[1]*1.0e-12):
#           if (axesMasses[0] < axesMasses[1]*1.0e-8):   # As of now, this seems to be a reasonable threshold (haven't tested large linear molecules, though)
#           if (axesMasses[0] < axesMasses[1]*1.0e-6):   # As of now, this seems to be a reasonable threshold (up to the HCCH molecule) (haven't tested large linear molecules, though)
            if (axesMasses[0] < axesMasses[1]*1.0e-5):   # As of now, this seems to be a reasonable threshold (up to the CCC molecule) (haven't tested large linear molecules, though)
                linear = True
            else:
                linear = False
    
            if (self.debug): print("Reactant is linear?", linear)
    
            # For accurate energy (and force) calculations, the
            # two molecules must be first separated
            self.separateMolecules()
    
            # Create the vibration instance for the first group of atoms
            vibFirstGroup = Vibrations(self.mol,indices=reactantIndexes,name=".vib",delta=0.01,nfree=2)
            vibFirstGroup.clean() # Cleaning is necessary otherwise it will just use results from old runs
            vibFirstGroup.run()
            vibFirstGroup.summary()
            vibFirstGroupData = vibFirstGroup.get_vibrations()
            Es, modes = vibFirstGroupData.get_energies_and_modes(all_atoms=False)
    
            if (linear):
                nonzeroEs = np.real(Es[5:]) * self.freq2energy
                nonzeroModes = modes[5:]
            else:
                nonzeroEs = np.real(Es[6:]) * self.freq2energy
                nonzeroModes = modes[6:]

            if (any(nonzeroEs < 0)):
                print("WARNING: one of the 'nonzero' modes has a negative frequency...")
                print("         for initial sampling, will convert to a positive number")
                nonzeroEs = np.abs(nonzeroEs)

            if (any(nonzeroEs <= 0.001*self.freq2energy)):
                print("WARNING: one of the 'nonzero' modes has a close to zero  frequency...")
                print("         for initial sampling, will convert to a small nonzero number (0.001 cm-1)")
                nonzeroEs[nonzeroEs <= 0.001*self.freq2energy] = 0.001 * self.freq2energy
    
            ###############################################################################################
    
            Natoms = len(reactantIndexes)
    
            q = self.mol.get_positions()[reactantIndexes]

            if (self.debug):
                print("Reactant, before selection:")
                for qi in q:
                    print(qi)
    
            # Monoatomics need no sampling
            if (Natoms == 1):
    
                pass
    
            # If this is a diatom and we are doing the
            # semiclassical QM sampling, set the quantum
            # numbers and enter here
            elif (Natoms == 2 and samplingMethod[Nreactant-1] == "semiclassical"):
    
                # Change these later to be changed via input file
                Nrot = rotationSampling[Nreactant-1]
                Nvib = vibrationSampling[Nreactant-1]
    
                if (self.debug):
                    print("Semiclassical QM sampling")
                    print("Nrot: ", Nrot)
                    print("Nvib: ", Nvib)
    
                freq = nonzeroEs[0]
                self.chooseQPforDiatom(reactantIndexes,Nrot,Nvib,freq)
    
            # In all other cases, we will be doing a thermal
            # sampling, so enter the temperatures of interest
            # here
            elif (samplingMethod[Nreactant-1] == "thermal"):
    
                # Change these later to be changed via input file
                Trot = rotationSampling[Nreactant-1]
                Tvib = vibrationSampling[Nreactant-1]
    
                if (self.debug):
                    print("Thermal sampling")
                    print("Trot: ", Trot)
                    print("Tvib: ", Tvib)
    
                # You must center the molecule first before calculating
                # any angular-velocity-related property
                self.centerMolecule(reactantIndexes)
    
                # Choose the angular momentum vector assuming the molecule
                # is oriented in the principal axes coordinates
                self.rotateMolecule(reactantIndexes,axesVectors)
                modes = self.rotateNormalModes(modes,axesVectors)
                nonzeroModes = self.rotateNormalModes(nonzeroModes,axesVectors)
                Erot, L = self.chooseAngularMomentumFromSymmetricTopThermalDistribution(linear,axesMasses,Trot)
                vibNums = self.chooseVibrationalQuantaFromThermalDistribution(nonzeroEs,Tvib)
    
                if (self.debug):
                    print("kBTrot:",self.kB*Trot)
                    print("kBTvib:",self.kB*Tvib)
                    print("nonzeroFreqs:",nonzeroEs)
                    print("nonzeroModes:",nonzeroModes)
                    print("Erot:",Erot)
                    print("Evib:",sum([float((x+0.5)*y) for x,y in zip (vibNums,nonzeroEs)]))
                    print("L:", L)
                    print("Nvibs:",[float(x) for x in vibNums])
                    print("Evibs:",[float((x+0.5)*y) for x,y in zip (vibNums,nonzeroEs)])
    
                self.chooseQPgivenNandLandNormalModes(reactantIndexes,vibNums,L,nonzeroEs,nonzeroModes)
    
                q = self.mol.get_positions()[reactantIndexes]
                p = self.mol.get_momenta()[reactantIndexes]
                ErotTEST, omegaTEST = self.getErotAndOmegaFromP(masses[reactantIndexes],q,p)

                if (self.debug):
                    print("Erot:",ErotTEST)
                    print("omega:", omegaTEST)

            # Enter in the desired Erot and Evib, and this will produce QP
            elif (samplingMethod[Nreactant-1] == "microcanonical"):

                # Change these later to be changed via input file
                Erot = rotationSampling[Nreactant-1] * (units.kcal/units.mol)
                Evib = vibrationSampling[Nreactant-1] * (units.kcal/units.mol)

                if (self.debug):
                    print("Microcanonical sampling")
                    print("Erot: ", Erot)
                    print("Evib: ", Evib)
    
                # You must center the molecule first before calculating
                # any angular-velocity-related property
                self.centerMolecule(reactantIndexes)

                # Choose the angular momentum vector assuming the molecule
                # is oriented in the principal axes coordinates
                self.rotateMolecule(reactantIndexes,axesVectors)
                modes = self.rotateNormalModes(modes,axesVectors)
                nonzeroModes = self.rotateNormalModes(nonzeroModes,axesVectors)
                Erot0, L = self.chooseAngularMomentumFromSymmetricTopThermalDistribution(linear,axesMasses,300.0)
                L = L * np.sqrt(Erot/Erot0)

                Nnonzeromodes = len(nonzeroEs)
                Evibs = []
                Evib0 = Evib * 1.0e0
                ZPEs, amplitudes = self.getVibrationalEnergiesAndAmplitudes(nonzeroEs,[0 for nonzeroE in nonzeroEs])

                # Approximate microcanonical sampling method (NOT the orthant sampling)
                for i in range(1,Nnonzeromodes):
                  Evibs.append(Evib0 * ((1.0e0 - random.random() ** (1.0e0/(Nnonzeromodes-i)) )))
                  Evib0 -= Evibs[-1]
                Evibs.append(Evib0)

                vibNums = []
                for i in range(Nnonzeromodes):
                  vibNums.append((Evibs[i]/nonzeroEs[i]) - 0.50e0)

                if (self.debug):
                    print("nonzeroFreqs:",nonzeroEs)
                    print("nonzeroModes:",nonzeroModes)
                    print("Erot:",Erot)
                    print("Evib:",sum(Evibs))
                    print("L:", L)
                    print("Nvibs:",[float(x) for x in vibNums])
                    print("Evibs:",[float((x+0.5)*y) for x,y in zip (vibNums,nonzeroEs)])
    
                self.chooseQPgivenNandLandNormalModes(reactantIndexes,vibNums,L,nonzeroEs,nonzeroModes)
    
                q = self.mol.get_positions()[reactantIndexes]
                p = self.mol.get_momenta()[reactantIndexes]
                ErotTEST, omegaTEST = self.getErotAndOmegaFromP(masses[reactantIndexes],q,p)

                if (self.debug):
                    print("Erot:",ErotTEST)
                    print("omega:", omegaTEST)

            # NEW!
            # Enter in the desired Erot and Evib, and this will produce QP
            elif (samplingMethod[Nreactant-1] == "microcanonical-wigner"):

                # Change these later to be changed via input file
                Erot = rotationSampling[Nreactant-1] * (units.kcal/units.mol)
                Evib = vibrationSampling[Nreactant-1] * (units.kcal/units.mol)

                if (self.debug):
                    print("Microcanonical sampling with Wigner rejection")
                    print("Erot: ", Erot)
                    print("Evib: ", Evib)
    
                # You must center the molecule first before calculating
                # any angular-velocity-related property
                self.centerMolecule(reactantIndexes)

                # Choose the angular momentum vector assuming the molecule
                # is oriented in the principal axes coordinates
                self.rotateMolecule(reactantIndexes,axesVectors)
                modes = self.rotateNormalModes(modes,axesVectors)
                nonzeroModes = self.rotateNormalModes(nonzeroModes,axesVectors)
                Erot0, L = self.chooseAngularMomentumFromSymmetricTopThermalDistribution(linear,axesMasses,300.0)
                L = L * np.sqrt(Erot/Erot0)

                Nnonzeromodes = len(nonzeroEs)
                ZPEs, amplitudes = self.getVibrationalEnergiesAndAmplitudes(nonzeroEs,[0 for nonzeroE in nonzeroEs])

                Pwigner_max = np.exp(-2*Evib/nonzeroEs[-1])  # The largest frequency determines the maximum probability
                while True:

                  # Approximate microcanonical sampling method (NOT the orthant sampling)
                  Evibs = []
                  Evib0 = Evib * 1.0e0
                  for i in range(1,Nnonzeromodes):
                    Evibs.append(Evib0 * ((1.0e0 - random.random() ** (1.0e0/(Nnonzeromodes-i)) )))
                    Evib0 -= Evibs[-1]
                  Evibs.append(Evib0)

                  vibNums = []
                  for i in range(Nnonzeromodes):
                    vibNums.append((Evibs[i]/nonzeroEs[i]) - 0.50e0)

                  Pwigner = np.exp(-2*sum([x+0.5e0 for x in vibNums]))
                  Prand = random.random()*Pwigner_max

                  if (self.debug):
                    print("Wigner rejection: ", Pwigner, ">=", Prand, "?")

                  if (Pwigner >= Prand): break   # Von Neumann rejection

                if (self.debug):
                    print("nonzeroFreqs:",nonzeroEs)
                    print("nonzeroModes:",nonzeroModes)
                    print("Erot:",Erot)
                    print("Evib:",sum(Evibs))
                    print("L:", L)
                    print("Nvibs:",[float(x) for x in vibNums])
                    print("Evibs:",[float((x+0.5)*y) for x,y in zip (vibNums,nonzeroEs)])
    
                self.chooseQPgivenNandLandNormalModes(reactantIndexes,vibNums,L,nonzeroEs,nonzeroModes)
    
                q = self.mol.get_positions()[reactantIndexes]
                p = self.mol.get_momenta()[reactantIndexes]
                ErotTEST, omegaTEST = self.getErotAndOmegaFromP(masses[reactantIndexes],q,p)

                if (self.debug):
                    print("Erot:",ErotTEST)
                    print("omega:", omegaTEST)

            # NEW!
            # Enter in the desired Erot and Evib, and this will produce QP
            elif (samplingMethod[Nreactant-1] == "microcanonical-quantum"):

                # Change these later to be changed via input file
                Erot = rotationSampling[Nreactant-1] * (units.kcal/units.mol)
                Evib = vibrationSampling[Nreactant-1] * (units.kcal/units.mol)

                if (self.debug):
                    print("Microcanonical sampling with Wigner rejection")
                    print("Erot: ", Erot)
                    print("Evib: ", Evib)
    
                # You must center the molecule first before calculating
                # any angular-velocity-related property
                self.centerMolecule(reactantIndexes)

                # Choose the angular momentum vector assuming the molecule
                # is oriented in the principal axes coordinates
                self.rotateMolecule(reactantIndexes,axesVectors)
                modes = self.rotateNormalModes(modes,axesVectors)
                nonzeroModes = self.rotateNormalModes(nonzeroModes,axesVectors)
                Erot0, L = self.chooseAngularMomentumFromSymmetricTopThermalDistribution(linear,axesMasses,300.0)
                L = L * np.sqrt(Erot/Erot0)

                Nnonzeromodes = len(nonzeroEs)
                ZPEs, amplitudes = self.getVibrationalEnergiesAndAmplitudes(nonzeroEs,[0 for nonzeroE in nonzeroEs])

                reorder_flag = True
                if reorder_flag:
                    i1=1
                    i2=2
                    nonzeroModes[i1],nonzeroModes[i2] = nonzeroModes[i2],nonzeroModes[i1]
                    ZPEs[i1],ZPEs[i2] = ZPEs[i2],ZPEs[i1]
                    amplitudes[i1],amplitudes[i2] = amplitudes[i2],amplitudes[i1]
                    nonzeroEs[i1],nonzeroEs[i2] = nonzeroEs[i2],nonzeroEs[i1]

                tmp_nonzeroEs = list(nonzeroEs)
                tmp_indices = list(range(Nnonzeromodes))
                dEgrain = 0.025 # min(0.01,Evib*1.1*5.0e-3)
                DOSs = []
                next_indices = []
                for i in range(Nnonzeromodes-1):
                    tmp_nonzeroEs.pop()
                    next_index = tmp_indices.pop()
                    next_indices.append(next_index)
                    states = self.beyer_swinehart(tmp_nonzeroEs, dEgrain, Evib*1.1, count='dos', interp=False)
                    DOSs.append(states)

                if (DOSs[0][round((Evib - 0.5e0*sum(nonzeroEs)) // dEgrain)] == 0):
                  raise ValueError("Quantum microcanonical ensemble does not a vibrational state with exactly {:.3f} kcal/mol energy".format(Evib/(units.kcal/units.mol)))

                while True:

                  # Approximate microcanonical sampling method (NOT the orthant sampling)
                  Evibs = []
                  Evib0 = Evib * 1.0e0 - 0.5e0*sum(nonzeroEs)   # Take out the ZPE
                  for next_index,dos in zip(next_indices,DOSs):
                    while True:
                      Evib_chosen = round(random.random() * Evib0 / nonzeroEs[next_index]) * nonzeroEs[next_index]
                      Pquantum = dos[int((Evib0-Evib_chosen) // dEgrain)] / dos[int((Evib0) // dEgrain)]
                      Prand = random.random()
                      if (self.debug): print("MicrocanonicalQuantum rejection: ", Pquantum, ">=", Prand, "? for mode", next_index, "Evib_left:", Evib0-Evib_chosen)
                      if (Pquantum > Prand): break
                    Evibs.insert(0,Evib_chosen)
                    Evib0 -= Evibs[0]
                    if (next_index == 2): break

                  candidateEvib12s = []
                  for n1 in range(1+int(Evib0//nonzeroEs[0])):
                    Evib1 = n1 * nonzeroEs[0]
                    for n2 in range(1+int(Evib0//nonzeroEs[1])):
                      Evib2 = n2 * nonzeroEs[1]
                      if (np.abs(Evib1+Evib2-Evib0) < dEgrain):
                        candidateEvib12s.append([Evib1,Evib2])
                        
                  if (len(candidateEvib12s) == 0):
                    if (self.debug): print("Warning: inconsistency in microcanonical-quantum sampling ... will try to resample...")
                    continue
                  tmpEvibs = random.choice(candidateEvib12s) + Evibs
                  Evibs = [float(x + 0.5e0*y) for x,y in zip(tmpEvibs,nonzeroEs)]

                  vibNums = []
                  for i in range(Nnonzeromodes):
                    vibNums.append((Evibs[i]/nonzeroEs[i]) - 0.50e0)

                  break

                if reorder_flag:
                    i1=1
                    i2=2
                    nonzeroModes[i1],nonzeroModes[i2] = nonzeroModes[i2],nonzeroModes[i1]
                    ZPEs[i1],ZPEs[i2] = ZPEs[i2],ZPEs[i1]
                    amplitudes[i1],amplitudes[i2] = amplitudes[i2],amplitudes[i1]
                    nonzeroEs[i1],nonzeroEs[i2] = nonzeroEs[i2],nonzeroEs[i1]

                    Evibs[i1],Evibs[i2] = Evibs[i2],Evibs[i1]
                    vibNums[i1],vibNums[i2] = vibNums[i2],vibNums[i1]



                if (self.debug):
                    print("nonzeroFreqs:",nonzeroEs)
                    print("nonzeroModes:",nonzeroModes)
                    print("Erot:",Erot)
                    print("Evib:",sum(Evibs))
                    print("L:", L)
                    print("Nvibs:",[float(x) for x in vibNums])
                    print("Evibs:",[float((x+0.5)*y) for x,y in zip (vibNums,nonzeroEs)])
    
                self.chooseQPgivenNandLandNormalModes(reactantIndexes,vibNums,L,nonzeroEs,nonzeroModes)
    
                q = self.mol.get_positions()[reactantIndexes]
                p = self.mol.get_momenta()[reactantIndexes]
                ErotTEST, omegaTEST = self.getErotAndOmegaFromP(masses[reactantIndexes],q,p)

                if (self.debug):
                    print("Erot:",ErotTEST)
                    print("omega:", omegaTEST)

    
            else:
    
                raise ValueError("Incorrect sampling method chosen: ", samplingMethod[Nreactant-1])
    
            q = self.mol.get_positions()[reactantIndexes]
            if (self.debug):
                print("Reactant, after selection:")
                for qi in q:
                    print(qi)
    
            # After choosing relative positions, rotate the
            # molecule randomly in 3D space
            axesVectors = self.chooseRandomSpatialRotation()
            self.centerMolecule(reactantIndexes)
            self.rotateMolecule(reactantIndexes,axesVectors)
            self.separateMolecules()
 
    # After sampling the relative coordinates and momenta of
    # each molecule, the relative position and speed of the
    # two molecules with respect to each other can be set
    # Collision energy in kcal/mol
    # Center-of-mass distance (dCM) and
    # impact parameter (b) in Anstrom
    def sampleAbsoluteQP(self,collisionEnergy,dCM=10.0,b=0.0):

        # Get the masses of each group of atoms
        masses=self.mol.get_masses()
        totalMass = sum(masses)
        massA = sum([masses[i] for i in self.atomsInFirstGroup])
        massB = totalMass - massA
    
        self.centerMolecule(self.atomsInFirstGroup)
        self.centerMolecule(self.atomsInSecondGroup)
    
        # Construct the  vector between the two
        # molecules' centers of mass
        dPARALLEL = np.sqrt(dCM**2 - b**2)
        rPARALLEL = np.array([1.0, 0.0, 0.0])
        rREL = np.array([dPARALLEL, b, 0.0])
    
        # First, prepare the impact parameter with this vector
        # orthogonal to the centers of mass vector
        q = self.mol.get_positions()
        for i in self.atomsInFirstGroup:
            q[i] += rREL
    
        # Second, prepare the center of mass velocities given the
        # collision energy
        reducedmass = massA*massB/(totalMass)
        speedREL = np.sqrt(2*(collisionEnergy*(units.kcal/units.mol))/reducedmass)
        vA = -(massB/totalMass)*(speedREL)*rPARALLEL
        vB =  (massA/totalMass)*(speedREL)*rPARALLEL
    
        v = self.mol.get_velocities()
        p = self.mol.get_momenta()
        for i in range(self.Natoms):
            if i in self.atomsInFirstGroup:
                v[i] += vA
            else:
                v[i] += vB
    
            p[i] = v[i] * masses[i]
    
        # Set the new positions and velocities
        self.mol.set_positions(q)
        self.mol.set_velocities(v)
        self.mol.set_momenta(p)   

