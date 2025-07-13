import numpy as np
import shutil, os

from ase.calculators.calculator import Calculator
from ase.units import Ha, Ang, Bohr

# For NWChemEx:
import simde
import pluginplay
from nwchemex import compute_energy, load_modules
from chemist import Atom, Molecule, ChemicalSystem, PointSetD
from chemist.basis_set import AtomicBasisSetD
from chemist import PointSetD

from friendzone.nwx2qcengine.call_qcengine import call_qcengine

class nwchemexcalculator(Calculator):

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        input_path,
        output_path=None,
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

        # Some suggested default values (for MD)
#       self.maxiter = 5000
#       self.d_convergence = 1.0e-8
#       self.e_convergence = 1.0e-8
#       self.freeze_core = 1

        # Default values for the input
        self.referencemethod = 'uhf'
        self.method = 'b3lyp'      # LevelOfTheory
        self.basis_set = '6-31g*'  # BasisSet
        self.charge = 0
        self.multiplicity = 2    # 2s+1
        self.save_movecs_interval = None

        # Some input values that are currently not set up (handled differently by QCEngine)
        self.memory = 1500       # In MB
        self.scratchdir = "/tmp"

        self.keywords = {
#           "geometry__nocenter": ".true.",               # Required
#           "geometry__noautoz" : ".true.",               # (Basically) required
        }

        # Read in the input
        with open(input_path, 'r') as f:
            for line in f:
                strippedline=" ".join(line.split())
                entries = strippedline.split(" ")
#               if (entries[0] == "memory"): self.memory = int(entries[1])                 # Memory is handled by QCEngine
#               if (entries[0] == "scratchdir"): self.scratchdir = str(entries[1])         # Scratch directories is handled by QCEngine
                if (entries[0] == "referencemethod"):
                    self.referencemethod = str(entries[1])
                    self.keywords["scf; "+self.referencemethod+"; "] = "end"
                    if (self.referencemethod == "rohf" or self.referencemethod == "rodft"):
                        self.keywords["dft; cgmin; "] = "end"
                        self.keywords["dft; rodft; "] = "end"
                elif (entries[0] == "freeze_core"):
                    self.freeze_core = int(entries[1])
                    if (self.freeze_core == 1):
                        self.keywords["mp2; freeze atomic; "] = "end"    # Usually the "core" electrons are assumed to be the ATOMIC core orbitals
                elif (entries[0] == "method"): self.method = str(entries[1])
                elif (entries[0] == "basis_set"):
#                   self.basis_set = str(entries[1])
                    basis_string_list = entries
                    basis_string_list.pop(0)
                    self.basis_set = " ".join(basis_string_list)
#                   self.basis_set = basis_string_list
                elif (entries[0] == "mulliken"):
                    self.mulliken = int(entries[1])
                    if (self.mulliken == 1):
                        self.keywords["dft; mulliken; "] = "end"
                        self.keywords["mp2; mulliken; "] = "end"
                elif (entries[0] == "charge"): self.charge = int(entries[1])
                elif (entries[0] == "multiplicity"): self.multiplicity = int(entries[1])
                elif (entries[0] == "diis"):
                    self.diis = int(entries[1])
                    if (self.diis == 1):
                        self.keywords["scf; diis; "] = "end"
                    else:
                        self.keywords["dft; nodiis; "] = "end"
                elif (entries[0] == "diisbas"):
                    self.diisbas = int(entries[1])
                    self.keywords["set scf:diisbas"] = self.diisbas
#                   self.keywords["dft; convergence diis nfock "+str(self.diisbas)+"; "] = "end"
                    self.keywords["dft; convergence diis "+str(self.diisbas)+"; "] = "end"
                elif (entries[0] == "maxiter"):
                    self.maxiter = int(entries[1])
                    self.keywords["set scf:maxiter"] = self.maxiter
                    self.keywords["set dft:iterations"] = self.maxiter
                elif (entries[0] == "guess_tolerance"): self.keywords["set scf:tolguess"] = float(entries[1])
                elif (entries[0] == "dft_grid"): self.keywords["dft; grid "+str(entries[1])+"; "] = "end"
                elif (entries[0] == "scf_level"):
                    level_string_list = entries
                    level_string_list.pop(0)
                    self.keywords["scf; "+" ".join(level_string_list)+"; "] = "end"
                elif (entries[0] == "dft_convergence_other_options"):
                    convergence_string_list = entries
                    convergence_string_list.pop(0)
                    self.keywords["dft; convergence "+" ".join(convergence_string_list)+"; "] = "end"
                elif (entries[0] == "scf_tol2e"):
                    self.keywords["set scf:tol2e"] = float(entries[1])
                elif (entries[0] == "dft_tolerances_tight"):
                    if (int(entries[1]) == 1):
                        self.keywords["dft; tolerances tight; "] = "end"
                elif (entries[0] == "dft_tolerances_tol_rho"):
                    self.keywords["dft; tolerances tol_rho; "+str(entries[1])+"; "] = "end"
                elif (entries[0] == "dft_tolerances_accCoul"):
                    self.keywords["dft; tolerances accCoul; "+str(entries[1])+"; "] = "end"
                elif (entries[0] == "dft_tolerances_radius"):
                    self.keywords["dft; tolerances radius; "+str(entries[1])+"; "] = "end"
                elif (entries[0] == "dft_e_convergence"):
                    self.keywords["dft; convergence energy "+str(entries[1])+"; "] = "end"
                elif (entries[0] == "dft_d_convergence"):
                    self.keywords["dft; convergence density "+str(entries[1])+"; "] = "end"
                elif (entries[0] == "dft_g_convergence"):
                    self.keywords["dft; convergence gradient "+str(entries[1])+"; "] = "end"
                elif (entries[0] == "scf_convergence"):
                    self.keywords["scf; thresh "+str(entries[1])+"; "] = "end"

                elif (entries[0] == "scf_direct"):
                    if (int(entries[1]) == 1):  self.keywords["scf; direct; "] = "end"
                elif (entries[0] == "dft_direct"):
                    if (int(entries[1]) == 1):  self.keywords["dft; direct; "] = "end"
                elif (entries[0] == "mp2_tight_precision"):
                    if (int(entries[1]) == 1):
                        self.keywords["mp2; tight; "] = "end"
                elif (entries[0] == "dft_disp_vdw"):
                    self.keywords["dft; dft disp vdw "+str(entries[1])+"; "] = "end"

                elif (entries[0] == "save_movecs_interval"):
                    self.save_movecs_interval = int(entries[1])
                    self.save_movecs_count = -9999999  # Do not do any saving yet

                else:
                    print("Ignoring keyword line (nothing will be added to the NWChem input file) with entries: ", entries)

        # Some inputs for details on saving intermediate results
        if (output_path):
            self.outputDIR = os.path.abspath(output_path)
            print("In the NWChemEx calculator, looking at this output directory: " + self.outputDIR)
            self.movecsSCF_restart = self.outputDIR+"/"+".nwchem.tmp.scf.movecs.restart"
            self.movecsDFT_restart = self.outputDIR+"/"+".nwchem.tmp.dft.movecs.restart"
            self.movecsSCF_restart_prev = self.outputDIR+"/"+".nwchem.tmp.scf.movecs.restart.prev"
            self.movecsDFT_restart_prev = self.outputDIR+"/"+".nwchem.tmp.dft.movecs.restart.prev"


        # Recently added ... I think this is necessary for ODFT
        if (self.referencemethod == "uhf" or self.referencemethod == "odft"):
            self.keywords["dft; odft; "] = "end"

        # The name of the intermediate file that holds the wavefunction
        self.movecs = self.scratchdir +"/"+".nwchem.tmp.scf.movecs"
        self.movecsSCF = self.scratchdir +"/"+".nwchem.tmp.scf.movecs"
        self.movecsDFT = self.scratchdir +"/"+".nwchem.tmp.dft.movecs"

        # If the user supplied the wavefunction file, use that for the first step
        if (os.path.isfile(self.movecsSCF)):
            print("Reading wavefunction from previous calculation stored in: " + self.movecsSCF + " ... ")
            self.ref_wfn = True
        else:
            self.ref_wfn = None

        # Prepare a minimally descriptive "model" for NWChemEx
        self.model = {"method":self.method, "basis":self.basis_set}

        # Prepare the environment and keywords for NWChemEx/QCEngine
        self.method = "NWChem : " + self.method

        mpiexec_command = "srun --overlap --mpi=pmix --nodes={nnodes} --ntasks-per-node={ranks_per_node} --ntasks={total_ranks} --cpus-per-task={cores_per_rank}" # For QCEngine, it has to be of this format: (1) nodes, (2) ranks_per_node, (3) total_ranks, (4) cores_per_rank
        self.MPIconfig = {
            "use_mpiexec" : True,
            "mpiexec_command": mpiexec_command,
            "nnodes": 1,
            "ncores": n_threads,
            "cores_per_rank": 1,
        }

        self.mm = pluginplay.ModuleManager()
        load_modules(self.mm)
        self.mm.change_input(self.method, 'basis set', self.basis_set)
        self.mm.change_input(self.method + " Gradient", 'basis set', self.basis_set)
        self.mm.change_input(self.method + " EnergyAndGradient", 'basis set', self.basis_set)
        self.mm.change_input(self.method, 'MPI config', self.MPIconfig)
        self.mm.change_input(self.method + " Gradient", 'MPI config', self.MPIconfig)
        self.mm.change_input(self.method + " EnergyAndGradient", 'MPI config', self.MPIconfig)

        # THESE THREE LINES WERE TEMPORARILY ADDED
        self.ref_wfn = True
        self.keywords["scf; vectors "] = " input "+self.movecsSCF+" output "+self.movecsSCF+"; end"     # Where to save the MOvecs
        self.keywords["dft; vectors "] = " input "+self.movecsDFT+" output "+self.movecsDFT+"; end"     # Where to save the MOvecs
        self.mm.change_input(self.method + " EnergyAndGradient", 'keywords', self.keywords)

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

            print(self.method, self.basis_set, chemist_sys)
            print(self.model)
            print(self.keywords)

#           self.mm.change_input(self.method + " EnergyAndGradient", 'keywords', self.keywords)   # TEMPORARILY COMMENTED OUT
            f = self.mm.run_as(simde.EnergyNuclearGradientStdVectorD(), self.method + " EnergyAndGradient", chemist_sys, PointSetD())
            e = f.pop(-1)

#           self.mm.change_input(self.method, 'keywords', self.keywords)
#           e = self.mm.run_as(simde.TotalEnergy(), self.method, chemist_sys)
            self.ref_wfn = True

#           self.keywords["scf; vectors "] = " input "+self.movecsSCF+" output "+self.movecsSCF+"; end"     # Where to save the MOvecs
#           self.keywords["dft; vectors "] = " input "+self.movecsDFT+" output "+self.movecsDFT+"; end"     # Where to save the MOvecs
#           self.mm.change_input(self.method + " Gradient", 'keywords', self.keywords)

#           f = self.mm.run_as(simde.EnergyNuclearGradientStdVectorD(), self.method + " Gradient", chemist_sys, PointSetD())

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

#           f = call_qcengine(simde.provisional.EnergyNuclearGradientD(), chemist_sys, 'nwchem', model=self.model, keywords=self.keywords, srun_ncores=self.n_tasks)
            f = call_qcengine(simde.EnergyNuclearGradientStdVectorD(), chemist_sys, 'nwchem', model=self.model, keywords=self.keywords, srun_ncores=self.n_tasks)
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

        # Save the wavefunction every now and then if required
        if (not (self.save_movecs_interval is None) and self.ref_wfn):
            print("Backing up a restart...")
            self.save_movecs_count += 1
            if (self.save_movecs_count >= self.save_movecs_interval):
                self.save_movecs_count = 0

                coordsfile_restart = os.path.join(self.outputDIR, ".nwchemex.restart.Q.xyz")
                momentafile_restart = os.path.join(self.outputDIR, ".nwchemex.restart.P.xyz")
                coordsfile_restart_prev = os.path.join(self.outputDIR, ".nwchemex.restart.prev.Q.xyz")
                momentafile_restart_prev = os.path.join(self.outputDIR, ".nwchemex.restart.prev.P.xyz")

                # Back up the restart (in case there's a job kill when writing the restart file)
                if (os.path.isfile(coordsfile_restart)):
                    shutil.copy(coordsfile_restart,coordsfile_restart_prev)
                    shutil.copy(momentafile_restart,momentafile_restart_prev)
                    if (os.path.isfile(self.movecsSCF_restart)):
                        shutil.copy(self.movecsSCF_restart,self.movecsSCF_restart_prev)
                    if (os.path.isfile(self.movecsDFT_restart)):
                        shutil.copy(self.movecsDFT_restart,self.movecsDFT_restart_prev)

                if (os.path.isfile(self.movecsSCF)):
                    shutil.copy(self.movecsSCF, self.movecsSCF_restart)
                if (os.path.isfile(self.movecsDFT)):
                    shutil.copy(self.movecsDFT, self.movecsDFT_restart)

                with open(coordsfile_restart,"w") as file:
                    file.write(str(Natoms)+"\n")
                    file.write(str(e)+"\n")
                    for i in range(Natoms):
                        anAtom = atoms[i]
                        file.write("{0:2s}  {1:16.8f} {2:16.8f} {3:16.8f}\n".format(anAtom.symbol,*anAtom.position))

                molmomenta = atoms.get_momenta()
                with open(momentafile_restart,"w") as file:
                    file.write(str(Natoms)+"\n")
                    file.write(str(e)+"\n")
                    for i in range(Natoms):
                        anAtom = atoms[i]
                        file.write("{0:2s}  {1:16.8f} {2:16.8f} {3:16.8f}\n".format(anAtom.symbol,*molmomenta[i]))



    def get_forces(self, atoms=None, force_consistent=False):
        forces = self.get_property('forces', atoms)
        return forces

