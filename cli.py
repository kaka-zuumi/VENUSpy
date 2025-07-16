from ase.io import read
from ase.md.verlet import VelocityVerlet
from ase import units

from ase.vibrations import Vibrations
import random

import argparse
import numpy as np
import os

import venuspy

# For initial sampling:
from venuspy.initialSampling import initialSampling

# For rewindrestarts:
from ase.io.extxyz import read_xyz

# For rewinds with sGDML:
from venuspy.md import regularMD, pingpongMD, smoothedMD, r2threshold



# Try importing BAGEL
try:
  from venuspy.calc.bagelcalc import bagelcalculator
except ImportError:
  print("WARNING: BAGEL has not been loaded ... it will not be available for initial sampling")

# Try importing psi4
try:
  from venuspy.calc.psi4calc import psi4calculator
except ImportError:
  print("WARNING: psi4 has not been loaded ... it will not be available for initial sampling")

# Try importing xtb (fast, semi-empirical DFT):
try:
# from tblite.ase import TBLite
  from venuspy.calc.xtbcalc import TBLite
except ImportError:
  print("WARNING: xtb has not been loaded ... it will not be available for initial sampling")

# Try import NWChem (not NWChemEx)
#from nwchemcalc import nwchemcalculator
try:
  from venuspy.calc.nwchemcalc import nwchemcalculator
except ImportError:
  print("WARNING: NWChem has not been loaded ... it will not be available for initial sampling")

# Try importing NWChemEx
try:
  from venuspy.calc.nwchemexcalc import nwchemexcalculator
except ImportError:
  print("WARNING: NWChemEx has not been loaded ... it will not be available for initial sampling")

# Try importing QCEngine/GAMESS
try:
  from venuspy.calc.qcengineGAMESScalc import qcengineGAMESScalculator
except ImportError:
  print("WARNING: QCEngine/GAMESS has not been loaded ... it will not be available for initial sampling")

# Try importing sGDML
try:
  import sgdml
  from sgdml.train import GDMLTrain
  from sgdml.predict import GDMLPredict
  from sgdml.intf.ase_calc import SGDMLCalculator

  from venuspy.calc.minisgdmlcalc import miniSGDMLCalculator

except ImportError:
  print("WARNING: sGDML has not been loaded ... it will not be available for initial sampling")

# Try importing chempotpy
try:
  from venuspy.calc.chempotpycalc import chempotpyCalculator
except ImportError:
  print("WARNING: chempotpy has not been loaded ... it will not be available for initial sampling")

# Try importing Schnet
try:
  import schnetpack as spk
  import torch
except ImportError:
  print("WARNING: Schnet has not been loaded ... it will not be available for initial sampling")

# Try importing Physnet
try:
  from venuspy.calc.physnetcalc import PNCalculator
except ImportError:
  print("WARNING: Physnet has not been loaded ... it will not be available for initial sampling")



###################################################

# Define global constants up here in the correct
# units so that internally, everything uses:
#    Energy: eV
#  Distance: Angstrom
#      Mass: Dalton

# global r2threshold

###################################################

parser = argparse.ArgumentParser(description="Do a single MD trajectory using a initial geometry (and momenta) and a sGDML model",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("initialGeometryFile", type=str, help="XYZ file with initial geometry; if initial conditions are sampled in the script, then this argument is required but is just an example XYZ")
parser.add_argument("PESinputFile", type=str, help="PES input file (may be a psi4 input file or a sGDML .npz model)")
parser.add_argument("outputDir", type=str, help="Directory to output stuff in")
parser.add_argument("--isotopeMassesFile", type=str, help="Change masses of specific atoms e.g. like isotopic substitution", default=None)
parser.add_argument("--rewindrestartFile", type=str, help="EXYZ file of a previous trajectory to rewind-restart from",default=None)
parser.add_argument("--rewindrestartEnergyThreshold", type=float, help="The energy threshold for allowable energy jumps to rewind-restart from",default=0.50)
parser.add_argument("--rewindrestartNskip", type=int, help="The number of frames at the end of the trajectory to skip possible restarts from",default=0)
parser.add_argument("--initialMomentaFile", type=str, help="XYZ file with initial momenta")
parser.add_argument("--atomsInFirstGroup", type=str, help="String with atoms which are in first group of atoms, separated by spaces")
parser.add_argument("--collisionEnergy", type=float, help="Collision energy in kcal/mol")
parser.add_argument("--impactParameter", type=float, help="Impact parameter in Angstrom")
parser.add_argument("--centerOfMassDistance", type=float, help="Distance between the two molecules' centers of mass in Angstrom")
#parser.add_argument('--optimize', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--optimize', type=bool, default=True)
parser.add_argument("--production", type=int, help="Supply number of steps for a production run")
parser.add_argument("--interval", type=int, help="How often to print out the energy")
parser.add_argument("--time_step", type=float, help="The time step in fs for a production run")
parser.add_argument("--MDtype", type=str, help="The type of molecular dynamics to do: 'regular', 'pingpong', or 'smoothed' with sGDML", default="regular")
parser.add_argument("--Nrewindsteps", type=int, help="The number of steps to use MLMD for when smoothing",default=4)
parser.add_argument("--dEmax", type=float, help="The maximum change in total energy allowed between consecutive steps",default=1.0)
parser.add_argument("--Edriftmax", type=float, help="The maximum change in total energy allowed from start to end",default=1.5)
parser.add_argument("--E0", type=float, help="The initial total energy to track the drift in total energy (e.g. if the trajectory is being restarted)",default=None)
parser.add_argument("--n_threads", type=int, help="The number of threads to ask psi4 to use")

parser.add_argument("--INITQPa", type=str, help="Initial sampling method for atoms in first group ('semiclassical', 'microcanonical', 'thermal', or None)", default="")
parser.add_argument("--NVIBa", type=int, help="Vibrational quantum number of atoms in first group (supply if using the 'semiclassical' initial sampling)")
parser.add_argument("--NROTa", type=int, help="Rotational quantum number of atoms in first group (supply if using the 'semiclassical' initial sampling)")
parser.add_argument("--EVIBa", type=float, help="Vibrational energy of atoms in first group (supply if using the 'microcanonical' initial sampling)")
parser.add_argument("--EROTa", type=float, help="Rotational energy of atoms in first group (supply if using the 'microcanonical' initial sampling)")
parser.add_argument("--TVIBa", type=float, help="Vibrational temperature of atoms in first group (supply if using the 'thermal' initial sampling)")
parser.add_argument("--TROTa", type=float, help="Rotational temperature of atoms in first group (supply if using the 'thermal' initial sampling)")

parser.add_argument("--INITQPb", type=str, help="Initial sampling method for atoms in second group ('semiclassical', 'microcanonical', 'thermal', or None)", default="")
parser.add_argument("--NVIBb", type=int, help="Vibrational quantum number of atoms in second group (supply if using the 'semiclassical' initial sampling)")
parser.add_argument("--NROTb", type=int, help="Rotational quantum number of atoms in second group (supply if using the 'semiclassical' initial sampling)")
parser.add_argument("--EVIBb", type=float, help="Vibrational energy of atoms in second group (supply if using the 'microcanonical' initial sampling)")
parser.add_argument("--EROTb", type=float, help="Rotational energy of atoms in second group (supply if using the 'microcanonical' initial sampling)")
parser.add_argument("--TVIBb", type=float, help="Vibrational temperature of atoms in second group (supply if using the 'thermal' initial sampling)")
parser.add_argument("--TROTb", type=float, help="Rotational temperature of atoms in second group (supply if using the 'thermal' initial sampling)")
args = vars(parser.parse_args())

########################################################################################

# A function to print the potential, kinetic and total energy
def printenergy(a,step,energyflag):
    if ("ERROR" in energyflag):
      epot = 0.0
      ekin = a.get_kinetic_energy() / (units.kcal/units.mol)
      print('@Epot = %.3f  Ekin = %.3f (T=%3.0fK)  '
            'Etot = %.3f  kcal/mol Step = %8d   %s' % (epot, ekin, ekin / (len(a) * 1.5 * 8.617281e-5), epot + ekin, step, energyflag))
    else:
      epot = a.get_potential_energy() / (units.kcal/units.mol)
      ekin = a.get_kinetic_energy() / (units.kcal/units.mol)
      print('@Epot = %.3f  Ekin = %.3f (T=%3.0fK)  '
            'Etot = %.3f  kcal/mol Step = %8d   %s' % (epot, ekin, ekin / (len(a) * 1.5 * 8.617281e-5), epot + ekin, step, energyflag))

########################################################################################

# Get the various arguments
Qfile = args["initialGeometryFile"]
input_path = args["PESinputFile"]
output_path = args["outputDir"]

rrQPfile = args["rewindrestartFile"]
rrQPdEmax = args["rewindrestartEnergyThreshold"]
rrNskip = args["rewindrestartNskip"]

Pfile = args["initialMomentaFile"]
atomsInFirstGroup = args["atomsInFirstGroup"]
ce = args["collisionEnergy"]
b = args["impactParameter"]
dCM = args["centerOfMassDistance"]

isotopeMassesFile = args["isotopeMassesFile"]
Nsteps = args["production"]
Nprint = args["interval"]
dt = args["time_step"]

MDtype = args["MDtype"]
dEmax = args["dEmax"]
Edriftmax = args["Edriftmax"]
E0= args["E0"]
Nrewindsteps = args["Nrewindsteps"]

optimize_flag = args["optimize"]

if ((Nsteps is None) or (Nprint is None) or (dt is None)):
  raise ValueError("For MD, need to specify these three: --production --interval --time_step")

n_threads = args["n_threads"]
if (n_threads is None): n_threads = 1

samplingMethod      = [ args["INITQPa"], args["INITQPb"] ]
vibrationSampling = []
rotationSampling = []
if (samplingMethod[0] == "semiclassical"):
  vibrationSampling.append(args["NVIBa"])
  rotationSampling.append(args["NROTa"])
elif ("microcanonical" in samplingMethod[0]):
  vibrationSampling.append(args["EVIBa"])
  rotationSampling.append(args["EROTa"])
else:
  vibrationSampling.append(args["TVIBa"])
  rotationSampling.append(args["TROTa"])
if (samplingMethod[1] == "semiclassical"):
  vibrationSampling.append(args["NVIBb"])
  rotationSampling.append(args["NROTb"])
elif ("microcanonical" in samplingMethod[1]):
  vibrationSampling.append(args["EVIBb"])
  rotationSampling.append(args["EROTb"])
else:
  vibrationSampling.append(args["TVIBb"])
  rotationSampling.append(args["TROTb"])

# Note:
# Right now, all arguments are mandatory (even though this does
# not raise a warning) except for isotopeMassesFile

# Adjust the maximum interatomic distance allowed
# for the simulation
if ((b is not None) and (dCM is not None) and (1.2*(b**2 + dCM**2) > r2threshold)):
  r2threshold = 1.2*(b**2 + dCM**2)

########################################################################################

# Look at the input file name to guess its identity
try_bagel = False
try_psi4 = False
try_xtb = False
try_mopac = False
try_chempotpy = False
try_nwchem = False
try_nwchemex = False
try_physnet = False
try_schnet = False
try_sgdml = False
try_qcenginegamess = False
if (input_path.endswith(('.npz',))):

  print("Input file '"+input_path+"' looks like a sGDML file so will attempt to read it in as such...")
  try:
    calc = SGDMLCalculator(input_path)
    try_sgdml = True
#   try_psi4 = False
  except:
    print("   Could not load file '"+input_path+"' as a sGDML model!")
    try_psi4 = True

elif (input_path.endswith(('.physnet.config',))):
  print("Input file '"+input_path+"' looks like a PhysNet config file so will attempt to read it in as such...")

# temporary:
# try:
  if True:
    tmp_mol = read(Qfile)
    calc = PNCalculator(input_path,tmp_mol)
    try_physnet = True

# except:
#   print("   Could not load file '"+input_path+"' as a PhysNet model!")
#   try_physnet = True

elif (input_path.endswith(('.bagel',))):
  try_bagel = True

elif (input_path.endswith(('.psi4',))):
  try_psi4 = True

elif (input_path.endswith(('.nwchem',))):
  try_nwchem = True

elif (input_path.endswith(('.chempotpy',))):
  try_chempotpy = True

elif (input_path.endswith(('.xtb',))):
  try_xtb = True

elif (input_path.endswith(('.mopac',))):
  try_mopac = True

elif (input_path.endswith(('.gamess.qcengine',))):
  try_qcenginegamess = True

elif (input_path.endswith(('.nwchemex',))):
  try_nwchemex = True

else:
  try_schnet = True




if (try_schnet):

  # Initialize the ML ase interface

  # To accomodate for the older versions of numpy used in Schnet==1.0
  np.int = np.int32
  np.float = np.float64
  np.bool = np.bool_

  calc = spk.interfaces.SpkCalculator(
      input_path,
      device="cpu",
      energy="energy",    # Name of energies
      forces="forces",    # Name of forces
      energy_units="kcal/mol",
      forces_units="kcal/mol/A",
      neighbor_list=spk.transform.ASENeighborList(5.0),
  )


if (try_bagel):
  print("Reading input file '"+input_path+"' as a BAGEL input file...")
  calc = bagelcalculator(custominputfile=input_path,command='mpirun /mnt/lustre/koa/koastore/rsun_group/camels/BAGEL_1_AV/BAGEL/bin/BAGEL bagel0.json > bagel0.out')

# calc.restarts_max = 2            # Max number of allowable SCF restarts

  # To conform to VENUS, we are going to keep the units
  # in kcal/mol and Angstroms (which the model was
  # originally trained on)
  calc.E_to_eV = units.Ha
  calc.Ang_to_R = units.Ang
  calc.F_to_eV_Ang = (units.Ha / units.Bohr)

if (try_psi4):
  print("Reading input file '"+input_path+"' as a psi4 input file...")
  calc = psi4calculator(input_path,n_threads=n_threads)

  calc.restarts_max = 2            # Max number of allowable SCF restarts

  # To conform to VENUS, we are going to keep the units
  # in kcal/mol and Angstroms (which the model was
  # originally trained on)
  calc.E_to_eV = units.Ha
  calc.Ang_to_R = units.Ang
  calc.F_to_eV_Ang = (units.Ha / units.Bohr)

if (try_xtb):
  print("Reading input file '"+input_path+"' as a xtb input file...")
  calc = TBLite(input_path)

if (try_qcenginegamess):
  print("Reading input file '"+input_path+"' as a QCEngine/GAMESS input file...")
  calc = qcengineGAMESScalculator(input_path,n_threads=n_threads)

  # To conform to VENUS, we are going to keep the units
  # in kcal/mol and Angstroms (which the model was
  # originally trained on)
  calc.E_to_eV = units.Ha
  calc.Ang_to_R = (units.Ang / units.Bohr)
  calc.F_to_eV_Ang = (units.Ha / units.Bohr)

if (try_nwchemex):
  print("Reading input file '"+input_path+"' as a NWChemEx input file...")
  calc = nwchemexcalculator(input_path,output_path=output_path,n_threads=n_threads)

  # To conform to VENUS, we are going to keep the units
  # in kcal/mol and Angstroms (which the model was
  # originally trained on)
  calc.E_to_eV = units.Ha
  calc.Ang_to_R = (units.Ang / units.Bohr)
  calc.F_to_eV_Ang = (units.Ha / units.Bohr)

if (try_nwchem):
  print("Reading input file '"+input_path+"' as a NWChem (not NWChemEx) input file...")
  calc = nwchemcalculator(input_path,output_path=output_path,n_threads=n_threads)

  # To conform to VENUS, we are going to keep the units
  # in kcal/mol and Angstroms (which the model was
  # originally trained on)
  calc.E_to_eV = 1.0 # units.Ha
  calc.Ang_to_R = 1.0 # (units.Ang / units.Bohr)
  calc.F_to_eV_Ang = 1.0 # (units.Ha / units.Bohr)

if (try_chempotpy):

  print("Reading input file '"+input_path+"' as a chempotpy input file...")

  # Read the input file
  with open(input_path, 'r') as f:

    totallines = []
    totalentries = []
    for line in f:
      strippedline=" ".join(line.split())
      entries = strippedline.split(" ")
      totallines.append(line)
      totalentries = totalentries + entries

    calc = chempotpyCalculator(totallines[0])
    strippedline=" ".join(line.split()[2:])
    print("Single state: {:s} - Chempotpy input: {:s}".format(totalentries[0],strippedline))

    # Note, different chempotpy PES have different maximum allowed distances
    # For O3 surfaces, 100 A is okay (= 2 * 50 A)
    initialSampling.dSEPARATED = 50.0

    # For now, just use a different distance threshold too
    r2threshold = (dCM + 5.0) * (dCM + 5.0)


if (try_mopac):
  print("Reading input file '"+input_path+"' as a MOPAC input file...")
  calc = mopaccalculator(input_path,output_path=output_path)

  # To conform to VENUS, we are going to keep the units
  # in kcal/mol and Angstroms (which the model was
  # originally trained on)
  calc.E_to_eV = 1.0 # units.Ha
  calc.Ang_to_R = 1.0 # (units.Ang / units.Bohr)
  calc.F_to_eV_Ang = 1.0 # (units.Ha / units.Bohr)


# Read in the geometry; set it in the "calculator"
mol = read(Qfile)
mol.set_calculator(calc)
mol.calc = calc
mol._calc = calc

# Get the output trajectory file ready
trajfile = os.path.join(output_path, "production.traj")

# If the masses are given, update the masses
# Note: this must be done BEFORE setting the momenta
if not (isotopeMassesFile is None):
  massFile = open(isotopeMassesFile,"r")
  newMasses = massFile.readlines()
  massFile.close()
  if (len(newMasses) != len(mol)): #.masses)):
    raise ValueError("Number of masses provided in the isotope mass file does not match the input XYZ")
  mol.set_masses([float(i) for i in newMasses])

# If a rewind-restart file is given, read in the coordinates (Q),
# momenta (P), and energies, for a possible restart
if not (rrQPfile is None):

  print("REWIND-RESTARTING using this file: " + rrQPfile)

  # Look at the last 1500 frames to restart from (but skip the last rrNskip)
  rrQPfilehandle = open(rrQPfile)
  if (rrNskip <= 0):
    rrmols = list(read_xyz(rrQPfilehandle,slice(-1500,None))) 
    rrNskip = 0
  else:
    rrmols = list(read_xyz(rrQPfilehandle,slice(-(1500+rrNskip),-rrNskip))) 

  rrQPflag = True
  Nconsecutivegoodframes = 0
  rrmols.reverse()   # Go from the most-recent to least-recent
  for i,rrmol in enumerate(rrmols):
    mol.set_positions(rrmol.get_positions())
    mol.set_velocities(rrmol.get_velocities())
    mol.set_momenta(rrmol.get_momenta())

    # Compare the original and restarted potential energies
    E_original = rrmol.calc.results['energy']
    try:
      Nconsecutivegoodframes += 1
      E_restart = mol.get_potential_energy()
      dE = abs(E_original - E_restart) / (units.kcal/units.mol)
      print("REWIND-RESTARTING sees that the %d-th to last frame (1-th to last = last frame) has original energy %.3f and restart energy %.3f kcal/mol"
             % (rrNskip+i+1, E_original / (units.kcal/units.mol), E_restart / (units.kcal/units.mol)))
    except:
      Nconsecutivegoodframes = 0
      dE = rrQPdEmax * 2
      print("REWIND-RESTARTING sees that the %d-th to last frame (1-th to last = last frame) has original energy %.3f and restart energy NaN kcal/mol"
             % (rrNskip+i+1, E_original / (units.kcal/units.mol)))

    # If they agree within the threshold, then go ahead with the restart
    if (dE < rrQPdEmax):

      # For a smoothed MD simulation, we need a few frames at the start to buffer over
      if (Nconsecutivegoodframes < 10):
        if (MDtype == "smoothed"): continue

      rrQPflag = False
      print("REWIND-RESTARTING accepts using the %d-th to last frame! (1-th to last = last frame)" % (rrNskip+i+1,))
      break

  if (rrQPflag):
    raise ValueError("REWIND-RESTARTING failed ... no acceptable frames to restart from")

# If a momenta file is given, read in the momenta
# Read it in as a geometry, and then set it into the molecule
elif not (Pfile is None):
  frame = read(Pfile)
  p = [atom.position for atom in frame]
  mol.set_momenta(p)

  masses=mol.get_masses()
  v = [p[i]/masses[i] for i in range(len(p))]
  mol.set_velocities(v)

# If there is no momenta file, then do initial sampling
else:

    Natoms = len(mol)

    # If no atoms are specified to be in the first group,
    # assume that this is a unimolecular sampling
    if (atomsInFirstGroup is None):
      atomsInFirstGroup = range(Natoms)

    atomsInFirstGroup = [int(i)-1 for i in atomsInFirstGroup.split()]

    atomsInSecondGroup = []
    for i in range(Natoms):
      if (i not in atomsInFirstGroup): atomsInSecondGroup.append(i)

    if ((len(atomsInFirstGroup) > 0) and (len(atomsInSecondGroup) > 0)):
      bimolecular_flag = True
      if ((ce is None) or (b is None) or (dCM is None)):
        raise ValueError("Lacking an argument for bimolecular sampling (collision energy, impact parameter, of center of mass distance)")
    else:
      bimolecular_flag = False

    print("")
    print("GEOMETRY INPUT")
    print("  Input geometry file: ", Qfile)
    print("Atoms in  first group: ", atomsInFirstGroup)
    print("Atoms in second group: ", atomsInSecondGroup)
    print("SAMPLING INPUT")
    print("  Input momenta file: ", Pfile)
    if (Pfile is None):
      print("          Optimize molecules? ", optimize_flag)
      print("     Group A sampling method: ", samplingMethod[0])
      print("     Group A       vibration: ", vibrationSampling[0])
      print("     Group A        rotation: ", rotationSampling[0])
      print("     Group B sampling method: ", samplingMethod[1])
      print("     Group B       vibration: ", vibrationSampling[1])
      print("     Group B        rotation: ", rotationSampling[1])
      print("        Impact parameter (A): ", b)
      print("      Initial separation (A): ", dCM)
      print("  Collsion energy (kcal/mol): ", ce)
    else:
      print("  Input momenta file: ", Pfile)

    print("##############################################################")
    print("")

    # Sample the internal positions and momenta of each of
    # the two molecules
    sampler = initialSampling(mol,atomsInFirstGroup,optimize=optimize_flag,
                      optimization_file=os.path.join(output_path, "optimization.traj"),
                      samplingMethodA=samplingMethod[0],vibrationalSampleA=vibrationSampling[0],rotationalSampleA=rotationSampling[0],
                      samplingMethodB=samplingMethod[1],vibrationalSampleB=vibrationSampling[1],rotationalSampleB=rotationSampling[1])

    print("Sampling internal degrees of freedom...")
    sampler.sampleRelativeQP()

    if (bimolecular_flag):
        print("Sampling relative degrees of freedom...")
        sampler.sampleAbsoluteQP(ce,dCM=dCM,b=b)

    else:
        sampler.centerMolecule(range(Natoms))

########################################################################################

if (MDtype == "regular"):

  MDgen = regularMD(mol,trajfile,dt,Nprint=Nprint)

elif (MDtype == "pingpong"):

  MDgen = pingpongMD(mol,trajfile,dt,Nprint=Nprint)

elif (MDtype == "smoothed"):

# MDgen = smoothedMD(mol,trajfile,dt,Nprint=Nprint,Nrewindsteps=Nrewindsteps,dEmax=dEmax,Edriftmax=Edriftmax)
  MDgen = smoothedMD(mol,trajfile,dt,Nprint=Nprint,Nrewindsteps=Nrewindsteps,dEmax=dEmax,Edriftmax=Edriftmax,E0=E0,n_threads=n_threads)

else:
  raise ValueError("MD type '"+MDtype+"' is not valid! Exiting...")

MDgen.production(Nsteps)



