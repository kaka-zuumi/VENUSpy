# VENUSpy



## Example Usage

We suggest using the command line interface (CLI) to do simulations. The file `cli.py` can be run with your favourite Python environment so long as the appropriate packages can be installed. Then, input files and command line arguments can be supplied to it.

### MOPAC

To use MOPAC, the software needs to be installed somehow. On Ubuntu 24.0 for example, it can be installed with:

```
sudo apt install mopac
```

MOPAC is a general semiempirical method which means that it can be used for most reactions of interest. We will try it out on the B + C2H2 reaction. The input files describing the geometry and PES are as follows:

<details>
<summary>input.xyz</summary>

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
<summary>input.mopac</summary>

```text
           method AM1
           charge 0
     multiplicity 2
          maxiter 1500
```
</details>

Then, any initial sampling and MD parameters can be given to this so long as the B and C2H2 are kept separate. For example, for a bimolecular collision initiated with 2.4 kcal/mol of collision energy and cold C2H2, the following command works:

```
python -u cli.py input.xyz input.mopac . --atomsInFirstGroup "1" --collisionEnergy 2.4 --impactParameter 1.0 --centerOfMassDistance 10.0 --production 100 --interval 1 --time_step 0.15 --INITQPa "thermal" --INITQPb "thermal" --TVIBa 300.0 --TROTa 300.0 --TVIBb 10.0 --TROTb 10.0 --n_threads 1 > production.log
```

Sometimes the SCF calculation in MOPAC does not converge which leads to the error: `ase.calculators.calculator.CalculationFailed: Calculator "mopac" failed with command "mopac .tmp.mop 2> /dev/null" failed`. This happens about 1/10 times for this system; restarting it often resolves this.


