"""This module defines an ASE interface to BAGEL.

"""
import os

import numpy as np

from ase import Atoms
from ase.calculators.calculator import FileIOCalculator, ReadError, Parameters
from ase.units import kcal, mol, Debye

class bagelcalculator(FileIOCalculator):
#   implemented_properties = ['energy', 'forces', 'dipole', 'magmom']
    implemented_properties = ['energy', 'forces', 'dipole']

    BAGELexe = "/mnt/lustre/koa/koastore/rsun_group/camels/BAGEL_1_AV/BAGEL/bin/BAGEL"
    command = 'mpirun ' + BAGELexe + ' bagel0.json > bagel0.out'

    discard_results_on_any_change = True

    def __init__(self, restart=None,
                 ignore_bad_restart_file=FileIOCalculator._deprecated,
                 label='bagel0', atoms=None, **kwargs):
        """Construct BAGEL-calculator object.

        Parameters:

        label: str
            Prefix for filenames (bagel0.json, bagel0.out, ...)

        Examples:

        """
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

        # Kazuumi's addition to make a custom input file (must have a 'GEOMETRY HERE' line)
        if ('custominputfile' in kwargs):
            self.custominputfile = kwargs.get('custominputfile',None)
            if not os.path.isfile(self.custominputfile):
                raise ReadError
            
            with open(self.custominputfile) as fd:
                lines = fd.readlines()

            self.pregeometryinput = ''
            self.postgeometryinput = ''
            pregeometry = True
            print("Custom BAGEL input file supplied:")
            for i, line in enumerate(lines):
                print(line,end='')
                if line.find('GEOMETRY HERE') != -1:
                    pregeometry = False
                    continue
                if (pregeometry):
                    self.pregeometryinput += line
                else:
                    self.postgeometryinput += line

            assert pregeometry == False

        else:
            self.custominputfile = None

            raise ValueError("This version of VENUSpy/BAGEL requires a custominputfile")

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        s = self.pregeometryinput

        # Write coordinates:
        Natoms = len(atoms); i = 1; lineend="},\n"
        for xyz, symbol in zip(atoms.positions, atoms.get_chemical_symbols()):
            if (i == Natoms): lineend="}\n"
            s += ' {0} "atom" : "{1}", "xyz" : [     {2},      {3},      {4} ] {5}'.format("{",symbol, *xyz, lineend)
            i += 1

#       for v, p in zip(atoms.cell, atoms.pbc):
#           if p:
#               s += 'Tv {0} {1} {2}\n'.format(*v)

        s += self.postgeometryinput

        with open('bagel0.json', 'w') as fd:
            fd.write(s)

    def get_index(self, lines, pattern):
        for i, line in enumerate(lines):
            if line.find(pattern) != -1:
                return i

    def read(self, label='bagel0'):
        FileIOCalculator.read(self, label)
        if not os.path.isfile(self.label + '.out'):
            raise ReadError

        with open(self.label + '.out') as fd:
            lines = fd.readlines()

        self.atoms = self.read_atoms_from_file(lines)
        self.read_results()

    def read_atoms_from_file(self, lines):
        """Read the Atoms from the output file stored as list of str in lines.
        Parameters:

            lines: list of str
        """

        atoms_were_found = False
        for i, line in enumerate(lines):
            if line.find('*** Geometry ***') != -1:
                atoms_were_found = True
                break

        assert atoms_were_found == True

        j = 2
        lines1 = lines[i:]
        symbols = []
        positions = []
        while not lines1[j].isspace():  # continue until we hit a blank line
            l = lines1[j].split()
            symbols.append(l[3][1:-2])
            positions.append([float(l[7][:-1]), float(l[8][:-1]), float(l[9])])
            j += 1

        return Atoms(symbols=symbols, positions=positions)

    def read_parameters_from_file(self, lines):
        """Find and return the line that defines a Mopac calculation

        Parameters:

            lines: list of str
        """
        for i, line in enumerate(lines):
            if line.find('CALCULATION DONE:') != -1:
                break

        lines1 = lines[i:]
        for i, line in enumerate(lines1):
            if line.find('****') != -1:
                return lines1[i + 1]

    def read_results(self):
        """Read the results, such as energy, forces, eigenvalues, etc.
        """
        FileIOCalculator.read(self, self.label)
        if not os.path.isfile(self.label + '.out'):
            raise ReadError

        with open(self.label + '.out') as fd:
            lines = fd.readlines()

        # To get the final energies (only for MCSCF right now)
        # awk 'BEGIN {Nstate=0} /FCI iteration/ {flag=1;Niter=0} /ci vector/ {if(flag==1){print $0, E};flag=0} {if(flag==1 && $1 == Niter && $2 == Nstate){E=$(NF-2);Niter+=1}}' casscf.log | tail

        Natoms = len(self.atoms)

        BAGEL_FCI_iteration_flag = False
        for i, line in enumerate(lines):

            # Print one copy of the output to the screen
            print(line,end="")

            if line.find('FCI iteration') != -1:
                BAGEL_FCI_iteration_flag = True
                Niter = "0"
                Nstate = "0"   # Assume we are looking for the ground state (state 0)

            elif line.find('ci vector') != -1:
                BAGEL_FCI_iteration_flag = False
#               print("Energy: " + str(self.results['energy']) + " " + line,end='')

            elif (BAGEL_FCI_iteration_flag):
                BAGEL_fields = line.split()

                if (len(BAGEL_fields) >= 5 and BAGEL_fields[0] == Niter and BAGEL_fields[1] == Nstate):
                    Niter = str(int(Niter) + 1)
                    self.results['energy'] = float(BAGEL_fields[-3]) # * kcal / mol

            elif line.find('Nuclear energy gradient') != -1:
                forces = [-float(line.split()[1])
                          for j, line in enumerate(lines[i+2:i+2 + 4 * Natoms]) if j%4>0 ]
                self.results['forces'] = np.array(
                    forces).reshape((-1, 3)) # * kcal / mol
#               print("Forces: ", self.results['forces'])

            elif line.find('Permanent dipole moment') != -1:
                if line.find('Permanent dipole moment: Unrelaxed') != -1: continue

                self.results['dipole'] = np.array(
                    [float(astring[:-1]) for astring in lines[i+1].split()[1:4]]) * Debye
#               print("Dipole: ", self.results['dipole'])


