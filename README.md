<p align="center">
  <img align="center" width="400" height="600" src="images/venuspylogo1.png">
</p>

# Summary

VENUSpy is a Python-based implementation of the VENUS software for molecular dynamics and initial sampling. VENUS has been used by computational chemists to simulate real-life conditions with appropriate statistical ensembles. Depending on what ensemble you're interested in and how many trajectories you can simulate, you may need to adjust what kind of sampling you do! Canonical, microcanonical, and normal-mode sampling can be carried out on any provided potential energy surface (PES).



# Example Usage

We suggest first-time users to use the command line interface (CLI) for simulations. It can be downloaded through git:

```
git clone https://github.com/kaka-zuumi/VENUSpy.git
```

Inside the downloaded folder, you can install VENUSpy with the Atomic Simulation Environment (ASE) package:

```
cd VENUSpy
pip install venuspy
pip install ase
```

The file `cli.py` can be run with your favourite Python environment so long as the appropriate packages can be installed. Then, input files and command line arguments can be supplied to it.

There are three key features in VENUSpy:
- [Sampling Methods](#sampling_methods)
- [Potential Energy Surfaces](#potential_energy_surfaces)
- [Hybrid Potential](#hybrid_potential)



## Sampling Methods

To illustrate the difference that sampling can make, consider how reactive water would be at room temperature. Chemical reactivity is typically controlled by the vibrational energy in the molecule's bonds. At room temperature (298K), molecules have an average of 0.592 kcal/mol in vibrational energy. We can compare vibrational energies of water sampled from:

- A 298K canonical ensemble 
- A 0.592 kcal/mol microcanonical ensemble

### Canonical sampling

<details>
<summary>Click here to expand the instructions</summary>
```
python -u cli.py H2O.input.xyz H2O.input.xtb .  --atomsInFirstGroup "1 2 3" --production 100 --interval 1 --time_step 0.15 --INITQPa "thermal" --TVIBa 298.0 --TROTa 0.0 --n_threads 1 > production.log
```
</details>

### Microcanonical sampling

<details>
<summary>Click here to expand the instructions</summary>
```
python -u cli.py H2O.input.xyz H2O.input.xtb .  --atomsInFirstGroup "1 2 3" --production 100 --interval 1 --time_step 0.15 --INITQPa "microcanonica" --EVIBa 0.592 --EROTa 0.0 --n_threads 1 > production.log
```
</details>


## Potential Energy Surfaces

To do any sampling or dynamics, a PES is necessary. Many options are available:

### MOPAC

<details>
<summary>Click here to expand the instructions</summary>

<img align="right" width="300" height="200" src="images/mopaclogo1.png">

To use MOPAC, it must first be installed somehow. On Ubuntu 24.0 for example, it can be installed with:

```
sudo apt install mopac
```

MOPAC is a general semiempirical software which means that it can be used for most reactions of interest. We will try it out on the B + C2H2 reaction. The input files describing the geometry and PES are as follows:

<details>
<summary>B.C2H2.input.xyz</summary>

```text
5

B      0.000000    0.000000    0.000000
C     -1.707100    1.879500    0.000000
C     -0.611600    2.321200    0.000000
H      0.365700    2.747600    0.000000
H     -2.684400    1.453100    0.000000
```
</details>

<details>
<summary>B.C2H2.input.mopac</summary>

```text
           method AM1
           charge 0
     multiplicity 2
          maxiter 1500
```
</details>

Then, any initial sampling and MD parameters can be given to this so long as the B and C2H2 are kept separate. For example, for a bimolecular collision initiated with 2.4 kcal/mol of collision energy and cold C2H2, the following command works:

```
python -u cli.py B.C2H2.input.xyz B.C2H2.input.mopac . --atomsInFirstGroup "1" --collisionEnergy 2.4 --impactParameter 1.0 --centerOfMassDistance 10.0 --production 100 --interval 1 --time_step 0.15 --INITQPa "thermal" --INITQPb "thermal" --TVIBa 300.0 --TROTa 300.0 --TVIBb 10.0 --TROTb 10.0 --n_threads 1 > production.log
```

Sometimes the SCF calculation in MOPAC does not converge which leads to the error: `ase.calculators.calculator.CalculationFailed: ... failed`. This happens about 1/5 times for this system; restarting it often resolves this.

</details>



### PSI4

<details>
<summary>Click here to expand the instructions</summary>

<img align="right" width="400" height="200" src="images/psi4logo1.png">

To use PSI4, it can be installed with `conda`. You can create a conda environment for it like so:

```
conda create --name psi4md psi4 ase -c conda-forge
conda activate psi4md
```

PSI4 is a general ab initio software which means that it can be used for any adiabatic reaction. Thus, it can be used for the same B + C2H2 reaction. The same geometry file `B.C2H2.input.xyz` can be used, while the PES is altered as:

<details>
<summary>B.C2H2.input.xyz</summary>

```text
5

B      0.000000    0.000000    0.000000
C     -1.707100    1.879500    0.000000
C     -0.611600    2.321200    0.000000
H      0.365700    2.747600    0.000000
H     -2.684400    1.453100    0.000000
```
</details>

<details>
<summary>B.C2H2.input.psi4</summary>

```text
referencemethod uhf
     psi4method b3lyp/def2-sv(p)
         charge 0
   multiplicity 2
```
</details>

Similar to the MOPAC implementation, any initial sampling and MD parameters can be given to this so long as the B and C2H2 are kept separate. For example, for a bimolecular collision initiated with 2.4 kcal/mol of collision energy and cold C2H2, the following command works:

```
python -u cli.py B.C2H2.input.xyz B.C2H2.input.psi4 . --atomsInFirstGroup "1" --collisionEnergy 2.4 --impactParameter 1.0 --centerOfMassDistance 10.0 --production 100 --interval 1 --time_step 0.15 --INITQPa "thermal" --INITQPb "thermal" --TVIBa 300.0 --TROTa 300.0 --TVIBb 10.0 --TROTb 10.0 --n_threads 1 > production.log
```

While PSI4 is parallelized, it is a full electronic structure calculation so it takes more than a minute to do the molecular dynamics, let alone the initial sampling. By default, if there are convergence issues at any step of the initial sampling or dynamics, the ab initio calculation is restarted with slightly different or looser parameters. We suggest skipping a full trajectory simulation if trying this as a test.

</details>



### TBLite

<details>
<summary>Click here to expand the instructions</summary>

<img align="right" width="400" height="200" src="images/xtblogo1.png">

To use TBLite, it can be installed with `pip`, like so:

```
pip install tblite
```

TBLite is a light-weight implementation of the extended tight-binding (xTB) Hamiltonian, which is a generic enough framework for most chemical reactions. Thus, it can be used for the same B + C2H2 reaction. The same geometry file `B.C2H2.input.xyz` can be used, while the PES is altered as:

<details>
<summary>B.C2H2.input.xyz</summary>

```text
5

B      0.000000    0.000000    0.000000
C     -1.707100    1.879500    0.000000
C     -0.611600    2.321200    0.000000
H      0.365700    2.747600    0.000000
H     -2.684400    1.453100    0.000000
```
</details>

<details>
<summary>B.C2H2.input.xtb</summary>

```text
      xtbmethod GFN2-xTB
         charge 0
   multiplicity 2
```
</details>

Similar to the MOPAC implementation, any initial sampling and MD parameters can be given to this so long as the B and C2H2 are kept separate. For example, for a bimolecular collision initiated with 2.4 kcal/mol of collision energy and cold C2H2, the following command works:

```
python -u cli.py B.C2H2.input.xyz B.C2H2.input.xtb . --atomsInFirstGroup "1" --collisionEnergy 2.4 --impactParameter 1.0 --centerOfMassDistance 10.0 --production 100 --interval 1 --time_step 0.15 --INITQPa "thermal" --INITQPb "thermal" --TVIBa 300.0 --TROTa 300.0 --TVIBb 10.0 --TROTb 10.0 --n_threads 1 > production.log
```

Sometimes the xTB calculation does not converge. By default, VENUSpy restarts the calculation a few times with slightly different parameters to try to save the trajectory.

</details>


### ChemPotPy

<details>
<summary>Click here to expand the instructions</summary>

To use ChemPotPy, the main package can be installed with `pip` and some helper packages must be installed with `conda`. As suggested by the developers, a new conda environment can be made for chempotpy with the appropriate packages installed like so:

```
conda create --name chempotpy
conda activate chempotpy
conda install python=3.11
conda install mkl mkl-service
conda install -c conda-forge gfortran
pip install numpy "numpy>=1.26,<1.27"
pip install charset_normalizer
pip install ase
pip install chempotpy
```

ChemPotPy is a collection of analytical potentials, originally made in Fortran and then packaged with a Python wrapper. Thus, only specific chemical reactions can be studied. We will try it out on the O + O2 reaction; find the full list of reactions available at: https://github.com/shuyinan/chempotpy.  The input files describing the geometry and PES are as follows:

<details>
<summary>O.O2.input.xyz</summary>

```text
3

O    0.00000000   0.00000000  100.00000000
O    0.00000000   0.00000000    0.59301532
O    0.00000000   0.00000000   -0.59301532
```
</details>

<details>
<summary>O.O2.input.chempotpy</summary>

```text
Q1-Sgm    chempotpy O3 O3_6_5Ap_2023 0
```
</details>

Then, any initial sampling and MD parameters can be given to this. For example, for a bimolecular collision initiated with 2.4 kcal/mol of collision energy and cold O2, the following command works:

```
python -u cli.py O.O2.input.xyz O.O2.input.chempotpy . --atomsInFirstGroup "1" --collisionEnergy 2.4 --impactParameter 1.0 --centerOfMassDistance 10.0 --production 100 --interval 1 --time_step 0.15 --INITQPa "thermal" --INITQPb "thermal" --TVIBa 300.0 --TROTa 300.0 --TVIBb 10.0 --TROTb 10.0 --n_threads 1 > production.log
```

</details>



### Python-Based ML Potentials

<details>
<summary>Click here to expand the instructions</summary>

<p>
<img align="right" width="250" height="100" src="images/pytorchlogo1.png">
</p>
<p>
<img align="right" width="300" height="200" src="images/tensorflowlogo1.jpg">
</p>

Many Python-based machine learning (ML) potentials exist now and because of the variety of different ML software, there may be conflicts between installed software. It is suggested to always create separate `conda` environments for each software. If `pip` is being used, separate Python virtual environments can be used for each software as well.

We will demonstrate interfaces with three examples: Schnet, sGDML, and Physnet.

For Schnet, first install an appropriate version (depending on the version of the model):

```
python3.11 -m venv .schnetmd
source .schnetmd/bin/activate
pip install torch==2.3 schnetpack==2.0.4 pytorch-lightning==2.2
pip install ase venuspy
```

And then do the initial sampling and MD:

```
python -u cli.py CH.C4H6.input.xyz MLmodels/CHC4H6/best_inference_model . --atomsInFirstGroup "1 2" --collisionEnergy 2.4 --impactParameter 1.0 --centerOfMassDistance 10.0 --production 100 --interval 1 --time_step 0.15 --INITQPa "thermal" --INITQPb "thermal" --TVIBa 300.0 --TROTa 300.0 --TVIBb 10.0 --TROTb 10.0 --n_threads 1 > production.log
```


For sGDML, first install the latest version with `pip`:

```
python -m venv .sgdmlmd
source .sgdmlmd/bin/activate
pip install sgdml
pip install ase venuspy
```

And then do the initial sampling and MD:

```
python -u cli.py HBr.HCl.input.xyz MLmodels/HBrHCl/model-train8000-sym2-sig0050.npz . --atomsInFirstGroup "1 2" --collisionEnergy 2.4 --impactParameter 1.0 --centerOfMassDistance 10.0 --production 100 --interval 1 --time_step 0.15 --INITQPa "thermal" --INITQPb "thermal" --TVIBa 300.0 --TROTa 300.0 --TVIBb 10.0 --TROTb 10.0 --n_threads 1 > production.log
```

For Physnet, first download the latest version from github and install tensorflow:

```
git clone https://github.com/MMunibas/PhysNet.git
conda create -n tensorflow1.14 tensorflow=1.14
conda activate tensorflow1.14
conda install ase -c conda-forge
```

You may need to change a few lines of code so that it works (there seem to be some backward compatability issues):

```
sed -i 's/import tensorflow as tf/import tensorflow.compat.v1 as tf/' PhysNet/*.py PhysNet/*/*.py PhysNet/*/*/*.py
sed -i 's/self._saver = tf.train.Saver(self.variables, save_relative_paths=True, max_to_keep=50)/self._saver = tf.train.Saver(max_to_keep=50)/' PhysNet/neural_network/NeuralNetwork.py
sed -i 's/@lru_cache/@lru_cache(maxsize=128)/' $(dirname $(which python))/../lib/python3.7/site-packages/ase/formula.py
sed -i 's/from importlib.metadata import entry_points/from importlib_metadata import entry_points/' $(dirname $(which python))/../lib/python3.7/site-packages/ase/io/formats.py
```

And then do the initial sampling and MD:

```
python -u cli.py CH.SH2.input.xyz MLmodels/CHSH2/model.physnet.config . --atomsInFirstGroup "1 2" --collisionEnergy 2.4 --impactParameter 1.0 --centerOfMassDistance 10.0 --production 100 --interval 1 --time_step 0.15 --INITQPa "thermal" --INITQPb "thermal" --TVIBa 300.0 --TROTa 300.0 --TVIBb 10.0 --TROTb 10.0 --n_threads 1 > production.log 2> /dev/null
```

</details>




## Hybrid Potential

See the attached manuscript to see details of when and how to use a hybrid potential energy surface. In general, an ab initio method would be combined with a ML method. Right now, the only ML method it is implemented with is sGDML (due to the ease in retraining it on-the-fly). For the ab initio method, let's test this out on one of the simplest non-analytical potentials, xTB. First, install both software:

```
python -m venv .hybridmd
source .hybridmd/bin/activate
pip install sgdml
pip install tblite
pip install ase venuspy
```

Add the argument `--MDtype "smoothed"` and you're good to go:

```
python -u cli.py B.C2H2.input.xyz B.C2H2.input.xtb . --MDtype "smoothed" --atomsInFirstGroup "1" --collisionEnergy 2.4 --impactParameter 1.0 --centerOfMassDistance 10.0 --production 5000 --interval 1 --time_step 0.15 --INITQPa "thermal" --INITQPb "thermal" --TVIBa 300.0 --TROTa 300.0 --TVIBb 10.0 --TROTb 10.0 --n_threads 1 > production.log
```

The xTB software, although not a completely ab initio theory, uses similar kinds of SCF methods and also runs into convergence problems. Over the course of the 5000 steps simulated, it often will have two or three energy drift/jump issues. When this happens, the sGDML surface quickly trains on the local surface and tries to save the trajectory. After some number of steps, the xTB method takes over again.

While basically irrelevant for xTB, this ability to save a higher level calculation is of great import to expensive methods like true ab initio calculations with DZ or TZ basis sets.



