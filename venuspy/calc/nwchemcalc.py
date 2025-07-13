import numpy as np
import shutil, os

from ase.calculators.calculator import Calculator
from ase.units import Ha, Ang, Bohr

from ase.calculators.nwchem import NWChem

# For debugging:
import os.path

class nwchemcalculator(Calculator):

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
        ASE calculator for the NWChem ab initio module

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

        super(nwchemcalculator, self).__init__(*args, **kwargs)

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

        scf = {"print":"low"}
        dft = {"print":"low"}
        dft_convergence = {}
        dft_tolerances = {}
        mp2 = {}
        basispar=""    #basispar="spherical"

        # Read in the input
        with open(input_path, 'r') as f:
            for line in f:
                strippedline=" ".join(line.split())
                entries = strippedline.split(" ")
                if (entries[0] == "memory"): self.memory = int(entries[1]) 
                if (entries[0] == "scratchdir"): self.scratchdir = str(entries[1]) 
                if (entries[0] == "referencemethod"):
                    self.referencemethod = str(entries[1])
                elif (entries[0] == "freeze_core"):
                    mp2["freeze"] = "atomic"
                elif (entries[0] == "method"): self.method = str(entries[1])
                elif (entries[0] == "basis_set"):
                    basis_string_list = entries
                    basis_string_list.pop(0)
                    self.basis_set = " ".join(basis_string_list)
                elif (entries[0] == "mulliken"):
                    dft["mulliken"] = None
                    mp2["mulliken"] = None
                elif (entries[0] == "charge"): self.charge = int(entries[1])
                elif (entries[0] == "multiplicity"): self.multiplicity = int(entries[1])
                elif (entries[0] == "diis"):
                    scf["diis"] = None
                    dft["diis"] = None
                elif (entries[0] == "diisbas"):
                    self.diisbas = int(entries[1])
                    scf["diisbas"] = self.diisbas
                    dft_convergence["diis"] = self.diisbas
                elif (entries[0] == "maxiter"):
                    self.maxiter = int(entries[1])
                elif (entries[0] == "guess_tolerance"):
                    scf["tolguess"] = float(entries[1])
                elif (entries[0] == "dft_grid"):
                    dft["grid"] = str(entries[1])
#               elif (entries[0] == "scf_level"):
#                   level_string_list = entries
#                   level_string_list.pop(0)
#                   self.keywords["scf; "+" ".join(level_string_list)+"; "] = "end"
#               elif (entries[0] == "dft_convergence_other_options"):
#                   convergence_string_list = entries
#                   convergence_string_list.pop(0)
#                   self.keywords["dft; convergence "+" ".join(convergence_string_list)+"; "] = "end"
                elif (entries[0] == "scf_tol2e"):
                    scf["tol2e"] = float(entries[1])
                elif (entries[0] == "dft_tolerances_tight"):
                    if (int(entries[1]) == 1):
                        dft_tolerances["tight"] = None
                elif (entries[0] == "dft_tolerances_tol_rho"):
                    dft_tolerances["tol_rho"] = float(entries[1])
                elif (entries[0] == "dft_tolerances_accCoul"):
                    dft_tolerances["accCoul"] = float(entries[1])
                elif (entries[0] == "dft_tolerances_radius"):
                    dft_tolerances["radius"] = float(entries[1])
                elif (entries[0] == "dft_e_convergence"):
                    dft_convergence["energy"] = float(entries[1])
                elif (entries[0] == "dft_d_convergence"):
                    dft_convergence["density"] = float(entries[1])
                elif (entries[0] == "dft_g_convergence"):
                    dft_convergence["gradient"] = float(entries[1])
                elif (entries[0] == "scf_convergence"):
                    scf["thresh"] = float(entries[1])

                elif (entries[0] == "scf_direct"):
                    if (int(entries[1]) == 1):
                        scf["direct"] = None
                elif (entries[0] == "dft_direct"):
                    if (int(entries[1]) == 1):
                        dft["direct"] = None
                elif (entries[0] == "mp2_tight_precision"):
                    if (int(entries[1]) == 1):
                        mp2["tight"] = None
                elif (entries[0] == "dft_disp_vdw"):
                    dft["disp"] = {"vdw":str(entries[1])}

                elif (entries[0] == "save_movecs_interval"):
                    self.save_movecs_interval = int(entries[1])
                    self.save_movecs_count = -9999999  # Do not do any saving yet

                else:
                    print("Ignoring keyword line (nothing will be added to the NWChem input file) with entries: ", entries)

        # Some inputs for details on saving intermediate results
        if (output_path):
            self.outputDIR = os.path.abspath(output_path)
            print("In the NWChem calculator, looking at this output directory: " + self.outputDIR)
            self.movecsSCF_restart = self.outputDIR+"/"+".nwchem.tmp.scf.movecs.restart"
            self.movecsDFT_restart = self.outputDIR+"/"+".nwchem.tmp.dft.movecs.restart"
            self.movecsSCF_restart_prev = self.outputDIR+"/"+".nwchem.tmp.scf.movecs.restart.prev"
            self.movecsDFT_restart_prev = self.outputDIR+"/"+".nwchem.tmp.dft.movecs.restart.prev"

        # The name of the intermediate file that holds the wavefunction
        self.movecs = self.scratchdir +"/"+".nwchem.tmp.scf.movecs"
        self.movecsSCF = self.scratchdir +"/"+".nwchem.tmp.scf.movecs"
        self.movecsDFT = self.scratchdir +"/"+".nwchem.tmp.dft.movecs"

        if (self.method in ["dft","odft","udft","rodft"]):
          dft["xc"] = self.method
          dft["mult"] = self.multiplicity
          dft["iterations"] = self.maxiter
          dft["convergence"] = dft_convergence
          dft["vectors"] = dict(input=self.movecsDFT,output=self.movecsDFT)

#         scf["uhf"] = None

          if (self.referencemethod in ["odft","udft"]): dft["odft"]=None
          if (self.referencemethod in ["rodft"]): dft["rodft"]=None

          calc = NWChem(label="nwchem",memory=str(self.memory)+' mb',charge=self.charge,
                        dft=dft,basis=self.basis_set,basispar=basispar,theory="dft")

        elif (self.method in ["uhf","hf","scf","rohf","rhf","mp2","ump2"]):

          scf["nopen"] = self.multiplicity-1
          scf["maxiter"] = self.maxiter
          scf["vectors"] = dict(input=self.movecsSCF,output=self.movecsSCF)

          if (self.referencemethod in ["uhf","ump2"]): scf["uhf"]=None
          if (self.referencemethod in ["rohf"]): scf["rohf"]=None

          if ("mp2" in self.method):
            calc = NWChem(label="nwchem",memory=str(self.memory)+' mb',charge=self.charge,
                          scf=scf,mp2=mp2,basis=self.basis_set,basispar=basispar,theory="mp2")

          else:
            calc = NWChem(label="nwchem",memory=str(self.memory)+' mb',charge=self.charge,
                          scf=scf,basis=self.basis_set,basispar=basispar,theory="scf")

        else:
          raise ValueError("Wrong reference wavefunction chosen for NWChem: "+self.referencemethod)

        self.nwchemcalc = calc

        # If the user supplied the wavefunction file, use that for the first step
        if (os.path.isfile(self.movecsSCF)):
            print("Reading wavefunction from previous calculation stored in: " + self.movecsSCF + " ... ")
            self.ref_wfn = True
        else:
            self.ref_wfn = None

        # Set the number of threads used in this session
        self.n_tasks = n_threads

        # Converts energy from the unit used by the ab initio method to eV.
        self.E_to_eV = E_to_eV

        # Converts length from the unit used in the ab initio method to Ang.
        self.Ang_to_R = F_to_eV_Ang / E_to_eV

        # Converts force from the unit used by the ab initio method to eV/Ang.
        self.F_to_eV_Ang = F_to_eV_Ang

        print("Summary of NWChem ab initio input file")
        print("     Method:", self.method)
        print("  Basis set:", self.basis_set)
        print("      nproc:", self.n_tasks)

    def calculate(self, atoms=None, *args, **kwargs):

        super(nwchemcalculator, self).calculate(atoms, *args, **kwargs)

        f = self.nwchemcalc.get_forces(atoms=atoms)
        e = self.nwchemcalc.results['energy']
#       print(dir(self.nwchemcalc))
#       print(self.nwchemcalc.results)
#       print(dir(self.nwchemcalc.results))

        # Print the NWChem output to stdout
        with open('nwchem.nwo', 'r') as afile:
            print(afile.read())

        # Make sure that the gradient was correctly read in
        assert len(atoms) == len(f)

        # Convert model units to ASE default units (eV and Ang)
        e *= self.E_to_eV
        f *= self.F_to_eV_Ang

        f = f.reshape(-1, 3)
        Natoms = len(atoms)
        print("")
        print("NWChem flag 1")
        print(Natoms)
        print("Energy: ", e)
        #for anAtom in atoms:
        for i in range(Natoms):
            anAtom = atoms[i]
            print("{0:2s}  {1:16.8f} {2:16.8f} {3:16.8f}    {4:18.10f} {5:18.10f} {6:18.10f}".format(anAtom.symbol,*anAtom.position, *f[i]))
        print("")

        self.results = {'energy': e, 'forces': f}

        # Save the wavefunction every now and then if required
#       if (not (self.save_movecs_interval is None) and self.ref_wfn):
        if (False):
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

