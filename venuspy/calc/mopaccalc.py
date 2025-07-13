import numpy as np
import shutil, os

from ase.calculators.calculator import Calculator
from ase.units import Ha, Ang, Bohr

from ase.calculators.mopac import MOPAC

# For debugging:
import os.path

class mopaccalculator(Calculator):

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        input_path,
        output_path=None,
        E_to_eV=Ha,
        F_to_eV_Ang=Ha/Bohr,
        use_torch=False,
        *args,
        **kwargs
    ):
        """
        ASE calculator for the MOPAC ab initio module

        A calculator takes atomic numbers and atomic positions from an Atoms object and calculates the energy and forces.

        Note
        ----
        ASE uses eV and Angstrom as energy and length unit, respectively. Unless the paramerters `E_to_eV` and `F_to_eV_Ang` are specified, the sGDML model is assumed to use kcal/mol and Angstorm and the appropriate conversion factors are set accordingly.
        Here is how to find them: `ASE units <https://wiki.fysik.dtu.dk/ase/ase/units.html>`_.

        Parameters
        ----------
                input_path : :obj:`str`
                        Path to a NWChem input file
                E_to_eV : float, optional
                        Conversion factor from whatever energy unit is used by the model to eV. By default this parameter is set to convert from kcal/mol.
                F_to_eV_Ang : float, optional
                        Conversion factor from whatever length unit is used by the model to Angstrom. By default, the length unit is not converted (assumed to be in Angstrom)
                use_torch : boolean, optional
                        Use PyTorch to calculate predictions
        """

        super(mopaccalculator, self).__init__(*args, **kwargs)

        # Some suggested default values (for MD)
#       self.maxiter = 5000
#       self.d_convergence = 1.0e-8
#       self.e_convergence = 1.0e-8
#       self.freeze_core = 1

        # Default values for the input
        self.method = 'PM7'      # LevelOfTheory
        self.charge = 0
        self.multiplicity = 2    # 2s+1
        self.save_movecs_interval = None
        self.custominput = None

        # Some input values that are currently not set up (handled differently by QCEngine)
        self.memory = 1500       # In MB
        self.scratchdir = "/tmp"

        # Read in the input
        with open(input_path, 'r') as f:
            for line in f:
                strippedline=" ".join(line.split())
                entries = strippedline.split(" ")
                if (entries[0] == "memory"): self.memory = int(entries[1]) 
                if (entries[0] == "scratchdir"): self.scratchdir = str(entries[1]) 
                elif (entries[0] == "method"): self.method = str(entries[1])
                elif (entries[0] == "mulliken"): pass
                elif (entries[0] == "charge"): self.charge = int(entries[1])
                elif (entries[0] == "multiplicity"): self.multiplicity = int(entries[1])
                elif (entries[0] == "custominput"): self.custominput = " ".join(entries[1:])
                elif (entries[0] == "maxiter"):
                    self.maxiter = int(entries[1])

                else:
                    print("Ignoring keyword line (nothing will be added to the NWChem input file) with entries: ", entries)

        # Get the calculator, the spin, and the charge ready
        self.spinarray = None
        self.chargearray = None
        self.mopaccalc = MOPAC(label='.tmp',task='1SCF GRADIENTS')
        self.mopaccalc.parameters['relscf'] = 0.0001                  # default is 0.0001
#       self.mopaccalc.parameters['relscf'] = 0.00001                 # default is 0.0001

        if (not (self.custominput is None)):
            self.mopaccalc = MOPAC(label='.tmp',task='1SCF GRADIENTS',custominput="AM1 VECTORS 1SCF GRADIENTS GEO-OK C.I.=6 SINGLET"); self.multiplicity = 1  # temporary change

        # Converts energy from the unit used by the ab initio method to eV.
        self.E_to_eV = E_to_eV

        # Converts length from the unit used in the ab initio method to Ang.
        self.Ang_to_R = F_to_eV_Ang / E_to_eV

        # Converts force from the unit used by the ab initio method to eV/Ang.
        self.F_to_eV_Ang = F_to_eV_Ang

        print("Summary of MOPAC ab initio input file")
        print("     Method:", self.method)

    def calculate(self, atoms=None, *args, **kwargs):

        # Set the total spin + charge for the MOPAC calculation
        # by setting initial local spins + charges
        if (self.spinarray is None):
            dum1 = np.array([0 for i in atoms])
            self.spinarray = dum1.copy()
            self.spinarray[0] = self.multiplicity - 1
            self.chargearray = dum1.copy()
            self.chargearray[0] = self.charge

        atoms.set_initial_magnetic_moments(self.spinarray)
        atoms.set_initial_charges(self.chargearray)

        super(mopaccalculator, self).calculate(atoms, *args, **kwargs)

        f = self.mopaccalc.get_forces(atoms=atoms)
        e = self.mopaccalc.get_final_heat_of_formation()

        # Print the NWChem output to stdout
        with open('.tmp.out', 'r') as afile:
            print(afile.read())

        # Make sure that the gradient was correctly read in
        assert len(atoms) == len(f)

        # Convert model units to ASE default units (eV and Ang)
        e *= self.E_to_eV
        f *= self.F_to_eV_Ang

        f = f.reshape(-1, 3)
        Natoms = len(atoms)
        print("")
        print("MOPAC flag 1")
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

