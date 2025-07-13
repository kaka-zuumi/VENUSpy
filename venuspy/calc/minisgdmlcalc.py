import logging

import numpy as np

import sgdml
from sgdml.train import GDMLTrain
from sgdml.predict import GDMLPredict
from sgdml.intf.ase_calc import SGDMLCalculator

from ase.calculators.calculator import Calculator
from ase.units import kcal, mol

class miniSGDMLCalculator(Calculator):

    implemented_properties = ['energy', 'forces']

    def __init__(
        self,
        model,
        E_to_eV=kcal / mol,
        F_to_eV_Ang=kcal / mol,
        n_threads=None,
        use_torch=False,
        *args,
        **kwargs
    ):
        """
        ASE calculator for the sGDML force field.

        A calculator takes atomic numbers and atomic positions from an Atoms object and calculates the energy and forces.

        Note
        ----
        ASE uses eV and Angstrom as energy and length unit, respectively. Unless the paramerters `E_to_eV` and `F_to_eV_Ang` are specified, the sGDML model is assumed to use kcal/mol and Angstorm and the appropriate conversion factors are set accordingly.
        Here is how to find them: `ASE units <https://wiki.fysik.dtu.dk/ase/ase/units.html>`_.

        Parameters
        ----------
                model : :obj:`sGDML model`
                        A sGDML model file
                E_to_eV : float, optional
                        Conversion factor from whatever energy unit is used by the model to eV. By default this parameter is set to convert from kcal/mol.
                F_to_eV_Ang : float, optional
                        Conversion factor from whatever length unit is used by the model to Angstrom. By default, the length unit is not converted (assumed to be in Angstrom)
                use_torch : boolean, optional
                        Use PyTorch to calculate predictions
        """

        super(miniSGDMLCalculator, self).__init__(*args, **kwargs)

        self.log = logging.getLogger(__name__)

#       self.gdml_predict = GDMLPredict(model, use_torch=use_torch)
        self.gdml_predict = GDMLPredict(model, num_workers=n_threads, use_torch=use_torch)
        self.gdml_predict.prepare_parallel(n_bulk=1)

        self.log.warning(
            'Please remember to specify the proper conversion factors, if your model does not use \'kcal/mol\' and \'Ang\' as units.'
        )

        # Converts energy from the unit used by the sGDML model to eV.
        self.E_to_eV = E_to_eV

        # Converts length from eV to unit used in sGDML model.
        self.Ang_to_R = F_to_eV_Ang / E_to_eV

        # Converts force from the unit used by the sGDML model to eV/Ang.
        self.F_to_eV_Ang = F_to_eV_Ang

    def calculate(self, atoms=None, *args, **kwargs):

        super(miniSGDMLCalculator, self).calculate(atoms, *args, **kwargs)

        # convert model units to ASE default units
        r = np.array(atoms.get_positions()) * self.Ang_to_R

        e, f = self.gdml_predict.predict(r.ravel())

        # convert model units to ASE default units (eV and Ang)
        e *= self.E_to_eV
        f *= self.F_to_eV_Ang

        self.results = {'energy': e, 'forces': f.reshape(-1, 3)}

