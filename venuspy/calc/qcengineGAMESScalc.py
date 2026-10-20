import numpy as np

from ase.calculators.calculator import Calculator
from ase.units import Ha, Ang, Bohr

# For QCEngine:
import qcengine as qcng
import qcelemental as qcel

class qcengineGAMESScalculator(Calculator):

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
        ASE calculator for the QCEngine/GAMESS ab initio module

        A calculator takes atomic numbers and atomic positions from an Atoms object and calculates the energy and forces.

        Note
        ----
        ASE uses eV and Angstrom as energy and length unit, respectively. Unless the paramerters `E_to_eV` and `F_to_eV_Ang` are specified, the sGDML model is assumed to use kcal/mol and Angstorm and the appropriate conversion factors are set accordingly.
        Here is how to find them: `ASE units <https://wiki.fysik.dtu.dk/ase/ase/units.html>`_.

        Parameters
        ----------
                input_path : :obj:`str`
                        Path to a QCEngine/GAMESS input file
                E_to_eV : float, optional
                        Conversion factor from whatever energy unit is used by the model to eV. By default this parameter is set to convert from kcal/mol.
                F_to_eV_Ang : float, optional
                        Conversion factor from whatever length unit is used by the model to Angstrom. By default, the length unit is not converted (assumed to be in Angstrom)
                use_torch : boolean, optional
                        Use PyTorch to calculate predictions
        """

        super(qcengineGAMESScalculator, self).__init__(*args, **kwargs)

        # Default values for the input
        self.maxiter = 10000
        self.d_convergence = 1.0e-7
        self.e_convergence = 1.0e-8
        self.referencemethod = 'uhf'
        self.freeze_core = 0
        self.df_ints_io = "None"
        self.method = 'b3lyp'              # LevelOfTheory
        self.libxc_functional = None       # LibXC Functional (if method="uselibxc")
        self.basis_set = '6-31g*'          # BasisSet
        self.mulliken = 0
        self.charge = 0
        self.multiplicity = 2    # 2s+1
        self.printlevel = 7    # default is 7

        self.memory = 1500       # In MB
        self.scratchdir = "/tmp"

        # Read in the input
        with open(input_path, 'r') as f:
            for line in f:
                strippedline=" ".join(line.split())
                entries = strippedline.split(" ")
                if (entries[0] == "memory"): self.memory = int(entries[1])
                if (entries[0] == "scratchdir"): self.scratchdir = str(entries[1])
                if (entries[0] == "referencemethod"): self.referencemethod = str(entries[1])
                if (entries[0] == "freeze_core"): self.freeze_core = int(entries[1])
                if (entries[0] == "df_ints_io"): self.df_ints_io = str(entries[1])
#               if (entries[0] == "method"): self.method = str(entries[1])
                if (entries[0] == "method"):
                    subentries = str(entries[1]).split("/")
                    self.method = str(subentries[0])
                    if (len(subentries) > 1):
                      self.basis_set = str(subentries[1])
#               if (entries[0] == "basis_set"): self.basis_set = str(entries[1]) 
                if (entries[0] == "libxc_functional"): self.libxc_functional = str(entries[1])
                if (entries[0] == "mulliken"): self.mulliken = int(entries[1])
                if (entries[0] == "charge"): self.charge = int(entries[1])
                if (entries[0] == "multiplicity"): self.multiplicity = int(entries[1])
                if (entries[0] == "print"): self.printlevel = str(entries[1])

        # By default, we start off with no initial MO guess
        self.ref_wfn = None
        self.Norb = None

        # Prepare a minimally descriptive "model" for QCEngine/GAMESS
        # GAMESS basis set documentation: https://myweb.liu.edu/~nmatsuna/gamess/input/BASIS.html
        # QCEngine/GAMESS methods documentation: https://github.com/MolSSI/QCEngine/blob/f5f6da3751373fa9b57ea484cbf71416ba679743/qcengine/programs/gamess/germinate.py
        self.model = {"method":self.method, "basis":self.basis_set}

        # Prepare the environment and keywords for QCEngine/GAMESS
        self.keywords = {
            "contrl__scftyp" : self.referencemethod,
            "contrl__maxit" : 200,       # For some reason, 200 iterations is the maximum
            "contrl__coord" : "unique",  # Turns out this does nothing in QCEngine
            "scf__diis" : ".true.", 
            "scf__ethrsh" : 1.0,         # DIIS error to start DIIS (deafult = 0.5 hartree)
            "scf__conv" : self.d_convergence,
            "contrl__nprint" : self.printlevel,
        }

        # If LibXC is available, use it
        if (self.method == "uselibxc"):
            if (self.libxc_functional is None):
                raise ValueError("If setting 'uselibxc', need to specify 'libxc_functional' in the input file! ... e.g,\n'libxc_functional b3lyp'")
            self.keywords["libxc__functional"] = self.libxc_functional

       # From the GAMESS documentation:
#NPUNCH =     option for output to the PUNCH file
#       =  0  do not punch out the final orbitals
#       =  1  punch out the occupied orbitals
#       =  2  punch out occupied and virtual orbitals
#             The default is NPUNCH = 2.
#NPREO  = energy and orbital printing options, applied
#         after other output options, for example NPRINT=-5
#         for no orbital output overrules NPREO.  NPREO
#         affects only printing, see NPUNCH just above.
#         Default: 1,9999,2,1 (meaning print all orbitals,
#         but no separate list of orbital energies).
#         To print only HOMO and "LUMO" LCAO coefficients,
#         and all orbital energies, enter:
#              NPREO(1)=0,-1,1,9999
        if True:
            self.keywords["scf__npunch"] = "0"
            self.keywords["scf__npreo(1)"] = "0,-1,1,9999"

        self.MPIconfig = {
#           "use_mpiexec" : True,
#           "mpiexec_command": mpiexec_command,
            "scratch_directory":".",
            "nnodes": 1,
            "ncores": n_threads,
#           "cores_per_rank": 2,
        }

        # Set the number of threads used in this session
        self.n_tasks = n_threads

        # Converts energy from the unit used by the ab initio method to eV.
        self.E_to_eV = E_to_eV

        # Converts length from the unit used in the ab initio method to Ang.
        self.Ang_to_R = F_to_eV_Ang / E_to_eV

        # Converts force from the unit used by the abinitio method to eV/Ang.
        self.F_to_eV_Ang = F_to_eV_Ang

        print("Summary of QCEnginge ab initio input file (for GAMESS)")
        print("      Model:", self.model)
        print("   Keywords:", self.keywords)
        print("      nproc:", self.n_tasks)

    def calculate(self, atoms=None, *args, **kwargs):

        super(qcengineGAMESScalculator, self).calculate(atoms, *args, **kwargs)

        # Convert model units to ASE default units
#       r = np.array(atoms.get_positions()) * self.Ang_to_R

        out = ""
        for anAtom in atoms:
            out += anAtom.symbol + " " + str(anAtom.position[0]) + " " + str(anAtom.position[1]) + " " + str(anAtom.position[2]) + "\n"

        # Fixing CoM and orientation and turning off symmetry to prevent molecular translation
        mol = qcel.models.v2.Molecule.from_data(out, fix_com=True, fix_orientation=True, fix_symmetry="C1")   # The newest version uses "v2" models
#       mol = qcel.models.Molecule.from_data(out, fix_com=True, fix_orientation=True, fix_symmetry="C1")    # Doesn't seem to work?
#       mol = qcel.models.Molecule.from_data(out, fix_com=True, fix_symmetry="C1")

        # Try the SCF convergence a few times with different approaches:
        #    (1) Try using the reference wavefunction (from the previous step)
        #    (2) Try the default superposition of atomic densities (SAD)
#       tmp_d_convergence = self.d_convergence
#       tmp_e_convergence = self.e_convergence
#       for i in range(100):
#           if (self.ref_wfn is None):
#               try:
#                   e, self.ref_wfn = psi4.energy(self.psi4method,write_orbitals=self.movecs,return_wfn=True)
#                   psi4.set_options({'d_convergence': self.d_convergence})
#                   psi4.set_options({'e_convergence': self.e_convergence})
#                   break
#               except SCFConvergenceError:
#                   if (i==10):
#                       raise SCFConvergenceError("SCF convergence failed too many times in a row!")
#                   print("SCF convergence ({0}) ... will now increase convergence threshold by a factor of 3".format(i))
#                   tmp_d_convergence = tmp_d_convergence * 3
#                   tmp_e_convergence = tmp_e_convergence * 3
#                   psi4.set_options({'d_convergence': tmp_d_convergence})
#                   psi4.set_options({'e_convergence': tmp_e_convergence})
#           else:
#               try:
#                   e, self.ref_wfn = psi4.energy(self.psi4method,restart_file=self.movecs,write_orbitals=self.movecs,return_wfn=True)
#                   break
#               except SCFConvergenceError:
#                   print("SCF convergence (0) ... will now use a SAD guess")
#                   self.ref_wfn = None

#       spec = qcel.models.v2.AtomicSpecification(driver=qcel.models.v2.DriverEnum.gradient,model=qcel.models.v2.Model(method=self.method,basis="sto",),keywords=self.keywords)
        spec = qcel.models.v2.AtomicSpecification(driver=qcel.models.v2.DriverEnum.gradient,model=qcel.models.v2.Model(method=self.method,basis=self.basis_set),keywords=self.keywords)
        inp = qcel.models.v2.AtomicInput(molecule=mol,specification=spec)       # The newest version uses "v2" models
#       inp = qcel.models.AtomicInput(molecule=mol, driver="gradient", model=self.model, keywords=self.keywords)
        if (self.ref_wfn is not None):
            inp.extras["VECguess"] =  self.ref_wfn
            inp.extras["VECnorb"]  =  self.Norb

        f_results = qcng.compute(inp, 'gamess', task_config=self.MPIconfig)

        if ("VEC" in f_results.extras) and ("VECnorb" in f_results.extras):
            self.ref_wfn  = f_results.extras["VEC"] 
            self.Norb     = f_results.extras["VECnorb"] 

        try:
          print(f_results.stdout)
        except:
          print(f_results.error.error_message)

        e = float(f_results.extras["qcvars"]["CURRENT ENERGY"])
        f = f_results.return_result

#       f = -np.array(f)
        f = np.array(f)

        # Make sure that the gradient was correctly read in
        assert len(atoms) == len(f)

        # If there are 'extra' things, do them here (e.g., Mulliken analysis)
#       if (self.mulliken > 0):
#           self.oeprop = OEProp(self.ref_wfn)
#           self.oeprop.add("MULLIKEN_CHARGES")
#           self.oeprop.compute()

        # Convert model units to ASE default units (eV and Ang)
        e *= self.E_to_eV
        f *= -self.F_to_eV_Ang

        f = f.reshape(-1, 3)
        Natoms = len(atoms)
        print("")
        print("QCEngine flag 1")
        print(Natoms)
        print("Energy: ", e)
        #for anAtom in atoms:
        for i in range(Natoms):
            anAtom = atoms[i]
            print("{0:2s}  {1:16.8f} {2:16.8f} {3:16.8f}    {4:18.10f} {5:18.10f} {6:18.10f}".format(anAtom.symbol,*anAtom.position, *f[i]))
        print("",flush=True)

        self.results = {'energy': e, 'forces': f}

    def get_forces(self, atoms=None, force_consistent=False):
        forces = self.get_property('forces', atoms)
        return forces

