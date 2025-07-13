import numpy as np

from ase.calculators.calculator import Calculator
from ase.units import Ha, Ang, Bohr

import psi4
from psi4.driver.p4util.exceptions import SCFConvergenceError
from psi4.core import OEProp
from psi4.core import variable, variables

class psi4calculator(Calculator):

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        input_path,
        n_threads=1,
        E_to_eV=Ha,
        F_to_eV_Ang=Ha/Bohr,
        use_torch=False,
        *args,
        **kwargs
    ):
        """
        ASE calculator for the psi4 ab initio module

        A calculator takes atomic numbers and atomic positions from an Atoms object and calculates the energy and forces.

        Note
        ----
        ASE uses eV and Angstrom as energy and length unit, respectively. Unless the paramerters `E_to_eV` and `F_to_eV_Ang` are specified, the sGDML model is assumed to use kcal/mol and Angstorm and the appropriate conversion factors are set accordingly.
        Here is how to find them: `ASE units <https://wiki.fysik.dtu.dk/ase/ase/units.html>`_.

        Parameters
        ----------
                input_path : :obj:`str`
                        Path to a psi4 input file
                E_to_eV : float, optional
                        Conversion factor from whatever energy unit is used by the model to eV. By default this parameter is set to convert from kcal/mol.
                F_to_eV_Ang : float, optional
                        Conversion factor from whatever length unit is used by the model to Angstrom. By default, the length unit is not converted (assumed to be in Angstrom)
                use_torch : boolean, optional
                        Use PyTorch to calculate predictions
        """

        super(psi4calculator, self).__init__(*args, **kwargs)

        # Default values for the psi4 input
        self.maxiter = 1000
        self.d_convergence = 1.0e-8
        self.e_convergence = 1.0e-8
        self.referencemethod = 'uhf'
        self.freeze_core = 0
        self.mp2_type = "df"
        self.qc_module = None
        self.df_ints_io = "None"
        self.dft_spherical_points = 302   # The default for DFT jobs
        self.dft_radial_points = 75       # The default for DFT jobs
        self.psi4method = 'b3lyp/6-31g*'  # LevelOfTheory/BasisSet with no spaces
        self.restarts_max = 10            # Max number of allowable SCF restarts
        self.mulliken = 0
        self.charge = 0
        self.multiplicity = 2    # 2s+1

        self.memory = 1500       # In MB
        self.scratchdir = "/tmp"

        # Read in the psi4 input
        with open(input_path, 'r') as f:
            for line in f:
                strippedline=" ".join(line.split())
                entries = strippedline.split(" ")
                if (entries[0] == "memory"): self.memory = int(entries[1])
                if (entries[0] == "scratchdir"): self.scratchdir = str(entries[1])
                if (entries[0] == "referencemethod"): self.referencemethod = str(entries[1])
                if (entries[0] == "freeze_core"): self.freeze_core = int(entries[1])
                if (entries[0] == "df_ints_io"): self.df_ints_io = str(entries[1])
                if (entries[0] == "dft_spherical_points"): self.dft_spherical_points = int(entries[1])
                if (entries[0] == "dft_radial_points"): self.dft_radial_points = int(entries[1])
                if (entries[0] == "psi4method"): self.psi4method = str(entries[1])
                if (entries[0] == "mulliken"): self.mulliken = int(entries[1])
                if (entries[0] == "charge"): self.charge = int(entries[1])
                if (entries[0] == "multiplicity"): self.multiplicity = int(entries[1])
                if (entries[0] == "mp2_type"): self.mp2_type = str(entries[1])
                if (entries[0] == "qc_module"): self.qc_module = str(entries[1])

                if (entries[0] == "maxiter"): self.maxiter = int(entries[1])
                if (entries[0] == "d_convergence"): self.d_convergence = float(entries[1])
                if (entries[0] == "e_convergence"): self.e_convergence = float(entries[1])

        # Set up the options for the psi4 calculation
        # Check keywords in the documentation: https://psicode.org/psi4manual/master/autodoc_glossary_options_c.html
        psi4.set_memory(str(self.memory)+' MB')
        psi4.set_options({'maxiter': self.maxiter})
        psi4.set_options({'d_convergence': self.d_convergence})
        psi4.set_options({'e_convergence': self.e_convergence})
        psi4.set_options({'reference': self.referencemethod})
        psi4.set_options({'freeze_core': self.freeze_core})
        psi4.set_options({'df_ints_io': self.df_ints_io})
        psi4.set_options({'dft_spherical_points': self.dft_spherical_points})
#       psi4.set_options({'DFT_SPHERICAL_POINTS': self.dft_spherical_points})
        psi4.set_options({'dft_radial_points': self.dft_radial_points})
#       psi4.set_options({'DFT_RADIAL_POINTS': self.dft_radial_points})
        psi4.set_options({'df_ints_io': self.df_ints_io})
        if ("mp2" in self.psi4method):
            self.mp2 = True
            self.basisset = self.psi4method.replace("mp2/","")
            self.psi4method = self.psi4method.replace("mp2","scf")
            psi4.set_options({'mp2_type': self.mp2_type})
        else:
            self.mp2 = False

        if (not (self.qc_module is None)):
            psi4.set_options({'qc_module': self.qc_module})

        # Some alternate psi4 SCF convergence options for second-order SCF (SOSCF):
        if (False):
            psi4.set_options({'soscf': True})
            psi4.set_options({'soscf_max_iter': self.maxiter})
            psi4.set_options({'soscf_conv': self.d_convergence})
            psi4.set_options({'soscf_start_convergence': 1.0e-6})

        self.movecs = self.scratchdir+"/md.wfn"
        self.ref_wfn = None

        # Set the number of threads used in this psi4 session
        psi4.set_num_threads(n_threads)

        # Converts energy from the unit used by psi4 to eV.
        self.E_to_eV = E_to_eV

        # Converts length from the unit used in psi4 to Ang.
        self.Ang_to_R = F_to_eV_Ang / E_to_eV

        # Converts force from the unit used by psi4 to eV/Ang.
        self.F_to_eV_Ang = F_to_eV_Ang

    def calculate(self, atoms=None, *args, **kwargs):

        super(psi4calculator, self).calculate(atoms, *args, **kwargs)

        # Convert model units to ASE default units
        r = np.array(atoms.get_positions()) * self.Ang_to_R

        # Load the geometry into the psi4 module
        geometry_string="{0} {1}\n".format(self.charge,self.multiplicity)
        geometry_string+="symmetry c1\n"   # Don't enable symmetry or that messes up the wavefunction step-by-step
        geometry_string+="no_reorient\n"
        geometry_string+="no_com\n"
        for anAtom in atoms:
            geometry_string+="{0} {1} {2} {3}\n".format(anAtom.symbol,*anAtom.position)
        #print(geometry_string)
        psi4.geometry(geometry_string)

        # Try the SCF convergence a few times with different approaches:
        #    (1) Try using the reference wavefunction (from the previous step)
        #    (2) Try the default superposition of atomic densities (SAD)
        tmp_d_convergence = self.d_convergence
        tmp_e_convergence = self.e_convergence
#       self.ref_wfn = None   # Kazuumi SUPER temporary line to test no initial guess wfn
        for i in range(100):
            if (self.ref_wfn is None):
                try:
                    e, self.ref_wfn = psi4.energy(self.psi4method,write_orbitals=self.movecs,return_wfn=True)
                    psi4.set_options({'d_convergence': self.d_convergence})
                    psi4.set_options({'e_convergence': self.e_convergence})
                    break
                except SCFConvergenceError:
                    if (i>=self.restarts_max):
                        raise SCFConvergenceError("SCF convergence failed too many times in a row!")
                    print("SCF convergence ({0}) ... will now increase convergence threshold by a factor of 3".format(i))
                    tmp_d_convergence = tmp_d_convergence * 3
                    tmp_e_convergence = tmp_e_convergence * 3
                    psi4.set_options({'d_convergence': tmp_d_convergence})
                    psi4.set_options({'e_convergence': tmp_e_convergence})
            else:
                try:
                    e, self.ref_wfn = psi4.energy(self.psi4method,restart_file=self.movecs,write_orbitals=self.movecs,return_wfn=True)
                    break
                except SCFConvergenceError:
                    print("SCF convergence (0) ... will now use a SAD guess")
                    self.ref_wfn = None

        # After converging the wavefunction, get the energy gradient
#       psi4.set_options({'basis': 'cc-pVDZ', 'mp2_type': 'conv'}); f = psi4.gradient('mp2')
#       f = psi4.gradient(self.psi4method)
        if (self.mp2):
            psi4.set_options({'basis': self.basisset, 'mp2_type': self.mp2_type, 'freeze_core': self.freeze_core})
            f, mp2_wfn = psi4.gradient("mp2",ref_wfn=self.ref_wfn,return_wfn=True)
            try:
                e = mp2_wfn.get_variable("CURRENT ENERGY")
            except:
                e = mp2_wfn.scalar_variable("CURRENT ENERGY")
        else:
            f = psi4.gradient(self.psi4method,ref_wfn=self.ref_wfn)
        f = f.to_array()

        # If there are 'extra' things, do them here
        # Check variables in the documentation: https://psicode.org/psi4manual/master/glossary_psivariables.html
        if (self.mulliken > 0):
            self.oeprop = OEProp(self.ref_wfn)
            self.oeprop.add("MULLIKEN_CHARGES")
            self.oeprop.compute()

        # Convert model units to ASE default units (eV and Ang)
        e *= self.E_to_eV
        f *= -self.F_to_eV_Ang

        f = f.reshape(-1, 3)
        Natoms = len(atoms)
        print("")
        print("psi4 flag 1")
        print(Natoms)
        print("Energy: ", e)
        #for anAtom in atoms:
        for i in range(Natoms):
            anAtom = atoms[i]
            print("{0:2s}  {1:16.8f} {2:16.8f} {3:16.8f}    {4:18.10f} {5:18.10f} {6:18.10f}".format(anAtom.symbol,*anAtom.position, *f[i]))
        print("")

        self.results = {'energy': e, 'forces': f}

    def get_forces(self, atoms=None, force_consistent=False):
        forces = self.get_property('forces', atoms)
        return forces

