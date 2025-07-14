from ase.md.verlet import VelocityVerlet
from ase import units

import numpy as np
import os

# For rewinds with sGDML:
from ase.build import minimize_rotation_and_translation
from ase.io.trajectory import Trajectory


# Try importing sGDML
try:
  import sgdml
  from sgdml.train import GDMLTrain
  from sgdml.predict import GDMLPredict
  from sgdml.intf.ase_calc import SGDMLCalculator

  from .calc.minisgdmlcalc import miniSGDMLCalculator

except ImportError:
  print("WARNING: sGDML has not been loaded ... cannot do 'smoothedMD'")

###################################################

# Define global constants up here in the correct
# units so that internally, everything uses:
#    Energy: eV
#  Distance: Angstrom
#      Mass: Dalton

global r2threshold

r2threshold = 24.0*24.0

###################################################


########################################################################################

# A function to print the potential, kinetic and total energy
def printenergy(a,step,energyflag=""):
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


# A function to see if any interatomic distance is > 20 
def checkGeneralReactionProgress(a):
    Natoms = len(a)
    stop_flag = False
    for i1 in range(1,Natoms):
      for i2 in range(0,Natoms-1):

        # Check the squared interatomic distances if any
        # are greater than r2threshold (default is 20^2 = 400)
        r = sum([(a.positions[i1][i] -
                  a.positions[i2][i])**2 for i in range(3)])
        if (r > r2threshold):
          stop_flag = True
          break

    return stop_flag

########################################################################################

class regularMD():

  def __init__(self,mol,trajfile,dt,Nprint=1):

    self.mol = mol

    # The timestep
    self.dt = dt

    # How often to print
    self.Nprint = Nprint

    # Get the output trajectory file ready
    self.trajfile = trajfile

    # Run MD with constant energy using the velocity verlet algorithm
    self.dyn = VelocityVerlet(self.mol, self.dt * units.fs )

########################################################################################

  def production(self,Nsteps):

    # Now run the dynamics
    printenergy(self.mol,0)
    traj = Trajectory(self.trajfile, mode="a", atoms=self.mol)
    traj.write()

    for i in range(Nsteps):
      self.dyn.run(self.Nprint)
      printenergy(self.mol,i+1)
      traj.write()

      stop_flag = checkGeneralReactionProgress(self.mol)
      if (stop_flag): break


########################################################################################

class pingpongMD():

  def __init__(self,mol,trajfile,dt,Nprint=1,dEmax=1.0,Edriftmax=1.5,E0=None):

    self.mol = mol

    # The timestep
    self.dt = dt

    # How often to print
    self.Nprint = Nprint

    # The maximum change in total energy allowed for consecutive steps
    self.dEmax = dEmax

    # The maximum change in total energy allowed from start to end
    self.Edriftmax = Edriftmax

    # Set the initial total energy to follow the energy drift
    if (E0 is not None):
      self.E0 = E0
    else:
      self.E0 = 0.0

    # A flag to keep track of when to "pong" (go in reverse)
    self.reversal_countdown = 0

    # Get the output trajectory file ready
    self.trajfile = trajfile

    # Run MD with constant energy using the velocity verlet algorithm
    self.dyn = VelocityVerlet(self.mol, self.dt * units.fs )

########################################################################################

  def production(self,Nsteps):

    # Now run the dynamics
    printenergy(self.mol,0)
    traj = Trajectory(self.trajfile, mode="a", atoms=self.mol)
    traj.write()

    for Nstep in range(Nsteps):
      self.dyn.run(self.Nprint)
      printenergy(self.mol,(Nstep+1)*self.Nprint)
      traj.write()
    
      epot = self.mol.get_potential_energy()
      ekin = self.mol.get_kinetic_energy()
      etot = (epot + ekin) / (units.kcal/units.mol)
    
      # Set some variables up for initialization and to keep track
      # of jumps/drifts in energy
      if (Nstep > 0):
        dE = np.abs(etot - prev_etot)
        Edrift = np.abs(etot - E0)
      else:
        E0 = etot
        dE = 0.0
        Edrift = 0.0

      prev_etot = etot
    
      # Keep track of energy jumps and reverse if its too bad
      if ((dE > self.dEmax) or (Edrift > self.Edriftmax)):
        if (self.reversal_countdown == 0):
          self.reversal_countdown = -11

      # After the reversal flag is set (counter is negative),
      # let the simulation go for 10 steps before reversing
      if (self.reversal_countdown < 0):
        self.reversal_countdown += 1
        if (self.reversal_countdown == 0): break

      stop_flag = checkGeneralReactionProgress(self.mol)
      if (stop_flag): break

    # Run MD with constant energy using the velocity verlet algorithm in REVERSE
    self.reversedyn = VelocityVerlet(self.mol, -self.dt * units.fs )

    printenergy(self.mol,Nsteps)
    traj.write()

    for Nstep in range(Nsteps,0,-1):
      self.reversedyn.run(self.Nprint)
      printenergy(self.mol,(Nstep-1)*self.Nprint)
      traj.write()

      stop_flag = checkGeneralReactionProgress(self.mol)
      if (stop_flag): break


########################################################################################

class smoothedMD():

  def __init__(self,mol,trajfile,dt,Nprint=1,Nrewindsteps=4,dEmax=1.0,Edriftmax=1.5,E0=None,n_threads=None):

    self.mol = mol
    self.molsgdml = mol.copy()

    # The timestep
    self.dt = dt

    # How often to print
    self.Nprint = Nprint

    # How many steps to use MLMD at a time
    self.Nrewindsteps = Nrewindsteps

    # The maximum change in total energy allowed for consecutive steps
    self.dEmax = dEmax

    # The maximum change in total energy allowed from start to end
    self.Edriftmax = Edriftmax

    # Set the initial total energy to follow the energy drift
    if (E0 is not None):
      self.E0 = E0
    else:
      self.E0 = 0.0

    # Number of threads
    self.n_threads = n_threads

    # Get the output trajectory file ready
    self.trajfile = trajfile

    # Run MD with constant energy using the velocity verlet algorithm
    self.dyn = VelocityVerlet(self.mol, self.dt * units.fs )

    # When needed, run MLMD with constant energy using the velocity verlet algorithm
    self.dynsgdml = VelocityVerlet(self.molsgdml, self.dt * units.fs)

########################################################################################

  def production(self,Nsteps):

    # Get some stuff initialized
    gdml_trajectory = []
    trajectoryTrainingSet = []
    temporaryTrainingSet = []
    gdml_train = GDMLTrain()

    currentstep = 0
    Nrewind = 0
    model = None
    E0 = self.E0

    # Now run the dynamics
    printenergy(self.mol,currentstep,"AB INITIO")
    traj = Trajectory(self.trajfile, mode="a", atoms=self.mol)
    traj.write()
    
    for Nstep in range(Nsteps):
    
      # Try to use the ab initio method
      energyflag = "AB INITIO"
      newframeflag = True
      try:
        self.dyn.run(self.Nprint)
        currentstep += self.Nprint
    
        epot = self.mol.get_potential_energy()
        ekin = self.mol.get_kinetic_energy()
        etot = (epot + ekin) / (units.kcal/units.mol)
    
        # Set some variables up for initialization and to keep track
        # of jumps/drifts in energy
        if (Nstep > 0):
          dE = np.abs(etot - trajectoryTrainingSet[-1]['etot'])
          Edrift = np.abs(etot - E0)
        else:
          E0 = etot
          dE = 0.0
          Edrift = 0.0
    
        # Keep track of energy jumps and rewind accordingly
        if ((dE > self.dEmax) or (Edrift > self.Edriftmax)):
          energyflag += " ENERGY JUMP/DRIFT (%.1f/%.1f)" % (dE,Edrift)
          Nrewind = self.Nrewindsteps
    
          # Test whether or not this new frame is geometrically "new"
          # compared to the training set
          if (len(temporaryTrainingSet) > 0):
            rmsds = []
            tmpmol = self.mol.copy()
            for oldframe in temporaryTrainingSet:
              oldmol = oldframe["mol"]
              minimize_rotation_and_translation(oldmol,tmpmol)
              rmsds.append(np.std(oldmol.get_positions() - tmpmol.get_positions()))
    
            # Usually for small molecules, consecutive frames differ by ~ 0.01 A
            if (np.min(rmsds) < 0.01):
    
              # If it's not new, then don't add it to the training set
              # and do a little bit more MLMD
              newframeflag = False
              Nrewind = 2
    
            # If it's not new, then we will be resetting the MLMD
            else:
              energyflag += " REWIND (%d)" % (1+len(gdml_trajectory))
              gdml_trajectory = []
    
          # If it's not new, then we will be resetting the MLMD
          else:
            energyflag += " REWIND (%d)" % (1+len(gdml_trajectory))
            gdml_trajectory = []
    
        else:
    
          # If the ab initio is successful from a rewind, then throw out
          # the old model to prepare for a new one later
          if (Nrewind > 0):
            model = None
    
          # A successful a.i. calculation means we can finally go "forward"
          # in time, including catching up on the ML frames we did earlier
          for oldmol in gdml_trajectory:
            traj.write(atoms=oldmol,calc=oldmol._calc,energy=oldmol._calc.results["energy"])
          traj.write()
          gdml_trajectory = []
    
          Nrewind = 0
          trajectoryTrainingSet.append({"mol":self.mol.copy(),"etot":etot,"energy":self.mol.calc.results['energy'],"forces":self.mol.calc.results['forces'],"step":currentstep})
          Ntrain = len(trajectoryTrainingSet)
    
#         # Temporary fix to ensure we have at least 10 GOOD frames
#         # and then use the ab initio less often afterwards
#         if (Ntrain > 9):
#           self.mol.calc.restarts_max = 1
#           self.mol._calc.restarts_max = 1
    
          # Empty the trajectory training set out if it gets too large
          if (Ntrain > 15):
            del trajectoryTrainingSet[0]
    
      # But prepare for failure (e.g. in the SCF convergence)
      except:
        energyflag += " ERROR"
        Nrewind = self.Nrewindsteps
        newframeflag = False

#       currentstep -= self.Nprint  # The last step was NOT successful
        if (model is None):
          energyflag += " REWIND (%d)" % (1)
          currentstep += 1  # or + Nprint dependning on how many successful steps there were before the error
    
      printenergy(self.mol,currentstep,energyflag)
    
      # Rewind a step or two and do MLMD instead
      if (Nrewind > 0):
    
        # If we are using MLMD in response to an energy jump, and
        # happen upon a new frame, add that new a.i. calculation
        # to the training set
        if (newframeflag):
          model = None
    
          # The a.i. calculation led to an energy jump so we won't
          # use it in the trajectory, but add it to the training set
          temporaryTrainingSet.append({"mol":self.mol.copy(),"etot":etot,"energy":self.mol.calc.results['energy'],"forces":self.mol.calc.results['forces'],"step":currentstep})
    
          # Empty the temporary training set out if it gets too large
          Ntrain = len(temporaryTrainingSet)
          if (Ntrain > 15):
            del temporaryTrainingSet[0]
    
        # (Re)make the model with the training set
        if (model is None):
    
          # Go back to the last successful a.i. calculation and
          # smooth it with ML (in case the problem was there)
          lastmol = trajectoryTrainingSet[-1]["mol"]
          q = lastmol.get_positions()
          self.molsgdml.set_positions(q)
          p = lastmol.get_momenta()
          self.molsgdml.set_momenta(p)
          v = lastmol.get_velocities()
          self.molsgdml.set_velocities(v)
          currentstep = trajectoryTrainingSet[-1]["step"]
    
          # Because we are going back at least one step, we can
          # go forward at least once more to catch up (may be even
          # more if the last successful a.i. calculation was a
          # number of steps back)
          Nrewind += 1
    
          # Set up the training set and model
          c1 = 1.0 / (units.kcal/units.mol)
    
          movingTrainingSet = trajectoryTrainingSet + temporaryTrainingSet
          Ntrain = len(movingTrainingSet)
    
          E = np.array([ x['energy']*c1 for x in movingTrainingSet])
          F = np.array([ x['forces']*c1 for x in movingTrainingSet])
          R = np.array([x['mol'].get_positions() for x in movingTrainingSet])
          Z = np.array([x['mol'].get_atomic_numbers() for x in movingTrainingSet])
          z = Z[0]
    
          dataset = {'type': 'd', 'code_version': sgdml.__version__, 'name': np.str_('movingTrainingSet'), 'theory': np.str_('unknown'), 'R': R, 'z': z, 'F': F, 'E': E,}
          dataset['F_min'], dataset['F_max'] = np.min(F.ravel()), np.max(F.ravel())
          dataset['E_min'], dataset['E_max'] = np.min(E.ravel()), np.max(E.ravel())
          dataset['r_unit'] = "Ang"; dataset['e_unit'] = "kcal/mol"
    
          # Do a small scan of sigma values to fit the model over
          sigs = [np.random.rand() * 7.5 for i in range(4)]
          sigs.append(5.0)
          sigs.reverse()
    
          depots = []
          models = []
          for sig in sigs:
            task = gdml_train.create_task(dataset,Ntrain,dataset,0,sig=sig,use_E_cstr=True)
#           task = gdml_train.create_task(dataset,Ntrain,dataset,0,sig=sig)
            model = gdml_train.train(task)
    
            calcsgdml = miniSGDMLCalculator(model,n_threads=self.n_threads)
            self.molsgdml.set_calculator(calcsgdml)
            self.molsgdml.calc = calcsgdml
            self.molsgdml._calc = calcsgdml
    
            self.molsgdml.calc.results = {}
            self.molsgdml._calc.results = {}
    
            models.append(model)
            depots.append(np.abs(self.molsgdml.get_potential_energy() - trajectoryTrainingSet[-1]["energy"]) / (units.kcal/units.mol))
    
            print("   Retrained the sGDML model with Ntrain: %d  and sigma: %.3f   ... first point error: %.2f kcal/mol" % (Ntrain,sig,depots[-1]))
    
            if (depots[-1] < 0.1): break
    
          # Use the model that best fits the first point
          model = models[np.argmin(depots)]
    
          calcsgdml = miniSGDMLCalculator(model)
          self.molsgdml.set_calculator(calcsgdml)
          self.molsgdml.calc = calcsgdml
          self.molsgdml._calc = calcsgdml
    
          self.molsgdml.calc.results = {}
          self.molsgdml._calc.results = {}
    
          # Check to see how the ML agrees on the first point
          printenergy(self.molsgdml,currentstep,"sGDML")
    
        else:
          self.molsgdml.calc.results = {}
          self.molsgdml._calc.results = {}

          # Note: is there a more elegant way to handle this case?
          if ("JUMP" in energyflag):
            currentstep -= self.Nprint
    
        # Do the MLMD
        for Nrewindstep in range(Nrewind):
          self.dynsgdml.run(self.Nprint)
          currentstep += self.Nprint
          printenergy(self.molsgdml,currentstep,"sGDML")
    
          gdml_trajectory.append(self.molsgdml.copy())
          gdml_trajectory[-1]._calc = miniSGDMLCalculator(model)
          gdml_trajectory[-1]._calc.results = {"energy":self.molsgdml._calc.results["energy"][0]}
          gdml_trajectory[-1].calc = gdml_trajectory[-1]._calc
    
    
        q = self.molsgdml.get_positions()
        self.mol.set_positions(q)
        p = self.molsgdml.get_momenta()
        self.mol.set_momenta(p)
        v = self.molsgdml.get_velocities()
        self.mol.set_velocities(v)
    
        self.mol.calc.results = {}
        self.mol._calc.results = {}
    
    
      stop_flag = checkGeneralReactionProgress(self.mol)
      if (stop_flag): break

