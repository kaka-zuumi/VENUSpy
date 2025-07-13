import logging

import numpy as np

import chempotpy

from ase.calculators.calculator import Calculator
from ase.units import eV, kcal, mol

class chempotpyCalculator(Calculator):

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        model,
        E_to_eV=eV,       # kcal / mol,
        F_to_eV_Ang=eV,   # kcal / mol,
        n_threads=None,
        *args,
        **kwargs
    ):
        """
        ASE calculator for a chempotpy potential.

        A calculator takes atomic numbers and atomic positions from an Atoms object and calculates the energy and forces.

        Note
        ----
        ASE uses eV and Angstrom as energy and length unit, respectively. Unless the paramerters `E_to_eV` and `F_to_eV_Ang` are specified, the sGDML model is assumed to use kcal/mol and Angstorm and the appropriate conversion factors are set accordingly.
        Here is how to find them: `ASE units <https://wiki.fysik.dtu.dk/ase/ase/units.html>`_.

        Parameters
        ----------
                model : :obj:`chempotpy model`
                        A chempotpy model file
                E_to_eV : float, optional
                        Conversion factor from whatever energy unit is used by the model to eV. By default this parameter is set to convert from kcal/mol.
                F_to_eV_Ang : float, optional
                        Conversion factor from whatever length unit is used by the model to Angstrom. By default, the length unit is not converted (assumed to be in Angstrom)
        """

        super(chempotpyCalculator, self).__init__(*args, **kwargs)

        self.log = logging.getLogger(__name__)

        modelargs = model.split()

        # Figure out whether the "model" input is a file or a one-liner description
        if (len(modelargs) == 1):   # This is a file
            with open(modelargs[0]) as f:
                modelargs = f.readlines()[0]

        self.system = modelargs[2]
        self.modelname = modelargs[3]

        # Figure out whether this is multistate or not
        if (len(modelargs) > 4):
            self.multistate = True
            self.state = int(modelargs[4])
        else:
            self.multistate = False

        self.log.warning(
            'Please remember to specify the proper conversion factors, if your model does not use \'kcal/mol\' and \'Ang\' as units.'
        )

        # From the chempotpy documentation:
#   The input to ChemPotPy is ALWAYS Cartesian coordinates in angstroms.
#   The output units are ALWAYS potential energies in eV, gradients in 
#   eV/angstrom, and nonadiabatic coupling vectors in 1/angstrom. 

        # Converts energy from the unit used by the sGDML model to eV.
        self.E_to_eV = E_to_eV

        # Converts length from eV to unit used in sGDML model.
        self.Ang_to_R = F_to_eV_Ang / E_to_eV

        # Converts force from the unit used by the sGDML model to eV/Ang.
        self.F_to_eV_Ang = F_to_eV_Ang

    def calculate(self, atoms=None, *args, **kwargs):

        super(chempotpyCalculator, self).calculate(atoms, *args, **kwargs)

        # convert model units to ASE default units
        r = atoms.get_positions() * self.Ang_to_R
        geom = []; i = 0
        for atom in atoms:
            geom.append([atom.symbol, r[i][0], r[i][1], r[i][2]]); i+=1

        e, f, nac = chempotpy.pgd(self.system,self.modelname,geom)

        # convert model units to ASE default units (eV and Ang)
        if (self.multistate):
            e = e[self.state] * self.E_to_eV
            f = -f[self.state] * self.F_to_eV_Ang
        else:
            e = e * self.E_to_eV
            f = -f * self.F_to_eV_Ang

        self.results = {'energy': e, 'forces': f.reshape(-1, 3)}

    def get_forces(self, atoms=None, force_consistent=False):
        forces = self.get_property('forces', atoms)
        return forces


