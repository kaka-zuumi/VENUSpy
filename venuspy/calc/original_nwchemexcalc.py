import numpy as np

from ase.calculators.calculator import Calculator
from ase.units import Ha, Ang, Bohr

# For NWChemEx:
import simde
import pluginplay
from nwchemex import compute_energy, load_modules
from chemist import Atom, Molecule, ChemicalSystem, PointSetD
from chemist.basis_set import AtomicBasisSetD

from friendzone.nwx2qcengine.call_qcengine import call_qcengine

class nwchemexcalculator(Calculator):

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
        ASE calculator for the NWChemEx ab initio module

        A calculator takes atomic numbers and atomic positions from an Atoms object and calculates the energy and forces.

        Note
        ----
        ASE uses eV and Angstrom as energy and length unit, respectively. Unless the paramerters `E_to_eV` and `F_to_eV_Ang` are specified, the sGDML model is assumed to use kcal/mol and Angstorm and the appropriate conversion factors are set accordingly.
        Here is how to find them: `ASE units <https://wiki.fysik.dtu.dk/ase/ase/units.html>`_.

        Parameters
        ----------
                input_path : :obj:`str`
                        Path to a NWChemEx input file
                E_to_eV : float, optional
                        Conversion factor from whatever energy unit is used by the model to eV. By default this parameter is set to convert from kcal/mol.
                F_to_eV_Ang : float, optional
                        Conversion factor from whatever length unit is used by the model to Angstrom. By default, the length unit is not converted (assumed to be in Angstrom)
                use_torch : boolean, optional
                        Use PyTorch to calculate predictions
        """

        super(nwchemexcalculator, self).__init__(*args, **kwargs)

        # Default values for the input
        self.maxiter = 10000
        self.d_convergence = 1.0e-8
        self.e_convergence = 1.0e-8
        self.referencemethod = 'uhf'
        self.freeze_core = 0
        self.df_ints_io = "None"
        self.method = 'b3lyp'      # LevelOfTheory
        self.basis_set = '6-31g*'  # BasisSet
        self.mulliken = 0
        self.charge = 0
        self.multiplicity = 2    # 2s+1

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
                if (entries[0] == "method"): self.method = str(entries[1])
                if (entries[0] == "basis_set"): self.basis_set = str(entries[1])
                if (entries[0] == "mulliken"): self.mulliken = int(entries[1])
                if (entries[0] == "charge"): self.charge = int(entries[1])
                if (entries[0] == "multiplicity"): self.multiplicity = int(entries[1])

        self.movecs = self.scratchdir +"/"+".nwchem.tmp.scf.movecs"
        self.movecsSCF = self.movecs
        self.movecsDFT = self.movecs
        self.ref_wfn = None

        assert (self.method == "SCF") # Only use "NWchem : SCF" for now

        # Prepare a minimally descriptive "model" for NWChemEx
        self.model = {"method":self.method, "basis":self.basis_set}

        # Prepare the environment and keywords for NWChemEx/QCEngine
        self.method = "NWChem : " + self.method
        self.keywords = {
#           "geometry__nocenter": ".true.",               # Required
#           "geometry__noautoz" : ".true.",               # (Basically) required
            "scf; "+self.referencemethod+"; " : "end",    # The reference wavefunction
            "set scf:maxiter"   : self.maxiter,           # Max number of iterations for the main wavefunction SCF
            "set cphf:maxiter"  : self.maxiter,           # Max number of iterations for the reference wavefunction
            "set scf:thresh"    : self.e_convergence,     # Energy convergence threshold
        }

        mpiexec_command = "srun --overlap --mpi=pmix --nodes={nnodes} --ntasks-per-node={ranks_per_node} --ntasks={total_ranks} --cpus-per-task={cores_per_rank}" # For QCEngine, it has to be of this format: (1) nodes, (2) ranks_per_node, (3) total_ranks, (4) cores_per_rank
        self.MPIconfig = {
            "use_mpiexec" : True,
            "mpiexec_command": mpiexec_command,
            "nnodes": 1,
            "ncores": n_threads,
            "cores_per_rank": 2,
        }

        self.mm = pluginplay.ModuleManager()
        load_modules(self.mm)
        self.mm.change_input(self.method, 'basis set', self.basis_set)
        self.mm.change_input(self.method + " Gradient", 'basis set', self.basis_set)
        self.mm.change_input(self.method, 'MPI config', self.MPIconfig)
        self.mm.change_input(self.method + " Gradient", 'MPI config', self.MPIconfig)
#       self.mm.change_input(self.method, 'keywords', self.keywords)
#       self.mm.change_input(self.method + " Gradient", 'keywords', self.keywords)

        # Set the number of threads used in this session
        self.n_tasks = n_threads

        # Converts energy from the unit used by the ab initio method to eV.
        self.E_to_eV = E_to_eV

        # Converts length from the unit used in the ab initio method to Ang.
        self.Ang_to_R = F_to_eV_Ang / E_to_eV

        # Converts force from the unit used by the ab initio method to eV/Ang.
        self.F_to_eV_Ang = F_to_eV_Ang

        print("Summary of NWChemEx ab initio input file (for NWChem)")
        print("      Model:", self.model)
        print("   Keywords:", self.keywords)
        print("      nproc:", self.n_tasks)

    def calculate(self, atoms=None, *args, **kwargs):

        super(nwchemexcalculator, self).calculate(atoms, *args, **kwargs)

        # Convert model units to ASE default units
        r = np.array(atoms.get_positions()) * self.Ang_to_R

#       # Load the geometry into the Chemist objects
        chemist_mol = Molecule()
        for anAtom in atoms:
            chemist_mol.push_back(Atom(anAtom.symbol,anAtom.number,anAtom.mass,*anAtom.position * self.Ang_to_R))
        chemist_mol.set_charge(self.charge)
        chemist_mol.set_multiplicity(self.multiplicity)
        chemist_sys = ChemicalSystem(chemist_mol)

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

        if True:    # Using "run_as"
            if (self.ref_wfn is None):
                self.keywords["scf; vectors "] = " output "+self.movecsSCF+"; end"     # Where to save the MOvecs
                self.keywords["dft; vectors "] = " output "+self.movecsDFT+"; end"     # Where to save the MOvecs

            else:
                self.keywords["scf; vectors "] = " input "+self.movecsSCF+" output "+self.movecsSCF+"; end"     # Where to save the MOvecs
                self.keywords["dft; vectors "] = " input "+self.movecsDFT+" output "+self.movecsDFT+"; end"     # Where to save the MOvecs

            print(self.mm)
            print(self.method, self.basis_set, chemist_sys)
            print(self.model)
            print(self.keywords)

            self.mm.change_input(self.method, 'keywords', self.keywords)
            e = self.mm.run_as(simde.TotalEnergy(), self.method, chemist_sys)
            self.ref_wfn = True

            self.keywords["scf; vectors "] = " input "+self.movecsSCF+" output "+self.movecsSCF+"; end"     # Where to save the MOvecs
            self.keywords["dft; vectors "] = " input "+self.movecsDFT+" output "+self.movecsDFT+"; end"     # Where to save the MOvecs
            self.mm.change_input(self.method + " Gradient", 'keywords', self.keywords)

#           f = self.mm.run_as(simde.provisional.EnergyNuclearGradientD(), self.method + " Gradient", chemist_sys)
            f = self.mm.run_as(simde.EnergyNuclearGradientStdVectorD(), self.method + " Gradient", chemist_sys, PointSetD())
#           f = -np.array(f).reshape(-1,3)
            f = np.array(f).reshape(-1,3)

            if False:
                print(self.mm)
                print(self.method, self.basis_set, chemist_sys)
                print(self.model)
                print(self.keywords)
                print("          simde.TotalEnergy(): ", simde.TotalEnergy())
                print("inputs of simde.TotalEnergy(): ", simde.TotalEnergy().inputs)
                print("Now PREPARING (inputs) of the dummy module:")
                self.mm.change_input('InitQP : samplingMethod1', 'basis set', self.basis_set)
#               self.mm.change_submod('InitQP : samplingMethod1', 'energy submodule', simde.TotalEnergy())
                self.mm.change_submod('InitQP : samplingMethod1', 'energy submodule', 'NWChem : SCF')
                self.mm.change_submod('InitQP : samplingMethod1', 'gradient submodule', 'NWChem : SCF Gradient')
                print("Now STARTING  the dummy module:")
#               qp = self.mm.run_as("", 'InitQP : samplingMethod1', chemist_sys)
                qp = self.mm.run_as(simde.TotalEnergy(), 'InitQP : samplingMethod1', chemist_sys)
                print("Now FINISHING the dummy module:")
                print("QP: ", qp)

        else:   # Skipping "run_as"
            if (self.ref_wfn is None):
                self.keywords["scf; vectors "] = " output "+self.movecsSCF+"; end"     # Where to save the MOvecs
                self.keywords["dft; vectors "] = " output "+self.movecsDFT+"; end"     # Where to save the MOvecs

                e = call_qcengine(simde.TotalEnergy(), chemist_sys, 'nwchem', model=self.model, keywords=self.keywords, srun_ncores=self.n_tasks)
                self.ref_wfn = True

            else:
                self.keywords["scf; vectors "] = " input "+self.movecsSCF+" output "+self.movecsSCF+"; end"     # Where to save the MOvecs
                self.keywords["dft; vectors "] = " input "+self.movecsDFT+" output "+self.movecsDFT+"; end"     # Where to save the MOvecs

                e = call_qcengine(simde.TotalEnergy(), chemist_sys, 'nwchem', model=self.model, keywords=self.keywords, srun_ncores=self.n_tasks)

            self.keywords["scf; vectors "] = " input "+self.movecsSCF+" output "+self.movecsSCF+"; end"     # Where to save the MOvecs
            self.keywords["dft; vectors "] = " input "+self.movecsDFT+" output "+self.movecsDFT+"; end"     # Where to save the MOvecs

            f = call_qcengine(simde.provisional.EnergyNuclearGradientD(), chemist_sys, 'nwchem', model=self.model, keywords=self.keywords, srun_ncores=self.n_tasks)
#           f = -np.array(f)
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

        print("f:", f)

        f = f.reshape(-1, 3)
        Natoms = len(atoms)
        print("")
        print("NWChemEx flag 1")
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

