#!/usr/bin/env python3

#import tensorflow as tf
import tensorflow.compat.v1 as tf
#tf.compat.v1.disable_eager_execution()
#tf.disable_eager_execution()  # Kazuumi change for the Physnet (tensorflow 1.14.0) + Psi4 install
#tf.disable_v2_behavior()

from PhysNet.neural_network.NeuralNetwork import NeuralNetwork as PN_NeuralNetwork
from PhysNet.neural_network.activation_fn import shifted_softplus as PN_shifted_softplus
import tensorflow.compat.v1 as tf
import numpy as np
import os
import sys
import argparse
import logging
import string
import random
from shutil import copyfile
from datetime import datetime

# Kazuumi additions:
import ase
from ase.neighborlist import neighbor_list
from ase import units


def softplus_inverse(x):
    '''numerically stable inverse of softplus transform'''
    return x + np.log(-np.expm1(-x))


'''
Calculator for the atomic simulation environment (ASE)
that evaluates energies and forces using a neural network
'''
class PNCalculator:
    #most parameters are just passed to the neural network
    def __init__(self,
                 config_file,                      #ckpt file from which to restore the model (can also be a list for ensembles)
                 atoms,                            #ASE atoms object
                 E_to_eV=units.kcal/units.mol,
                 F_to_eV_Ang=units.kcal/units.mol, 
                 charge=0,                         #system charge
                 dtype=tf.float32):                #single or double precision

        #define command line arguments
        PNparser = argparse.ArgumentParser(fromfile_prefix_chars='@')
        PNparser.add_argument("--restart", type=str, default=None,  help="restart training from a specific folder")
        PNparser.add_argument("--num_features", type=int,   help="dimensionality of feature vectors")
        PNparser.add_argument("--num_basis", type=int,   help="number of radial basis functions")
        PNparser.add_argument("--num_blocks", type=int,   help="number of interaction blocks")
        PNparser.add_argument("--num_residual_atomic", type=int,   help="number of residual layers for atomic refinements")
        PNparser.add_argument("--num_residual_interaction", type=int,   help="number of residual layers for the message phase")
        PNparser.add_argument("--num_residual_output", type=int,   help="number of residual layers for the output blocks")
        PNparser.add_argument("--cutoff", default=10.0, type=float, help="cutoff distance for short range interactions")
        PNparser.add_argument("--use_electrostatic", default=1, type=int,   help="use electrostatics in energy prediction (0/1)")
        PNparser.add_argument("--use_dispersion", default=1, type=int,   help="use dispersion in energy prediction (0/1)")
        PNparser.add_argument("--grimme_s6", default=None, type=float, help="grimme s6 dispersion coefficient")
        PNparser.add_argument("--grimme_s8", default=None, type=float, help="grimme s8 dispersion coefficient")
        PNparser.add_argument("--grimme_a1", default=None, type=float, help="grimme a1 dispersion coefficient")
        PNparser.add_argument("--grimme_a2", default=None, type=float, help="grimme a2 dispersion coefficient")
        args = PNparser.parse_args(["@"+config_file])

#       checkpoint=args.restart                                   # In the original, we would load the directory
#       checkpointfile = tf.train.latest_checkpoint(checkpoint)   # 
#       print("I see the checkpoint: ", checkpoint)
#       print("I see the checkpoint file: ", checkpointfile)
        checkpointfile=args.restart                               # but here we just load the file itself

        # If we may have multiple sessions / models, we may need to reset the graph at the start
        tf.reset_default_graph()

        #create neural network
#       self._nn = NeuralNetwork(F=args.num_features,           
        self._nn = PN_NeuralNetwork(F=args.num_features,           
               K=args.num_basis,                
               sr_cut=args.cutoff,              
               num_blocks=args.num_blocks, 
               num_residual_atomic=args.num_residual_atomic,
               num_residual_interaction=args.num_residual_interaction,
               num_residual_output=args.num_residual_output,
               use_electrostatic=(args.use_electrostatic==1),
               use_dispersion=(args.use_dispersion==1),
               s6=args.grimme_s6,
               s8=args.grimme_s8,
               a1=args.grimme_a1,
               a2=args.grimme_a2,
               activation_fn=PN_shifted_softplus, 
#              activation_fn=shifted_softplus, 
               scope="neural_network")

        self._sess = tf.Session()
        if checkpointfile is not None:
            self._nn._saver.restore(self._sess, checkpointfile)
        else:
            raise ValueError("I guess you're not as smart as you thought you were buddy.")

        # Converts energies and forces
        self.E_to_eV = E_to_eV
        self.F_to_eV_Ang = F_to_eV_Ang

        #create neighborlist
        if True:
            self._sr_cutoff = self._nn.sr_cut
            self._lr_cutoff = None
            self._use_neighborlist = False
        if False:
            self._sr_cutoff = self._nn.sr_cut
            self._lr_cutoff = self._nn.sr_cut * 2
            self._use_neighborlist = True

        #create placeholders for feeding data
        self._Q_tot      = np.array(1*[charge])
        self._Z          = tf.placeholder(tf.int32, shape=[None, ], name="Z") 
        self._R          = tf.placeholder(dtype,    shape=[None,3], name="R")        
        self._idx_i      = tf.placeholder(tf.int32, shape=[None, ], name="idx_i") 
        self._idx_j      = tf.placeholder(tf.int32, shape=[None, ], name="idx_j") 
        self._offsets    = tf.placeholder(dtype,    shape=[None,3], name="offsets") 
        self._sr_idx_i   = tf.placeholder(tf.int32, shape=[None, ], name="sr_idx_i") 
        self._sr_idx_j   = tf.placeholder(tf.int32, shape=[None, ], name="sr_idx_j") 
        self._sr_offsets = tf.placeholder(dtype,    shape=[None,3], name="sr_offsets") 
        
        #calculate atomic charges, energy and force evaluation nodes
        if self.use_neighborlist:
            Ea, Qa, Dij, nhloss = self.nn.atomic_properties(self.Z, self.R, self.idx_i, self.idx_j, self.offsets, self.sr_idx_i, self.sr_idx_j, self.sr_offsets)
        else:
            Ea, Qa, Dij, nhloss = self.nn.atomic_properties(self.Z, self.R, self.idx_i, self.idx_j, self.offsets)
        self._charges = self.nn.scaled_charges(self.Z, Qa, self.Q_tot)
        self._energy, self._forces = self.nn.energy_and_forces_from_scaled_atomic_properties(Ea, self.charges, Dij, self.Z, self.R, self.idx_i, self.idx_j)

#       #calculate atomic charges, energy and force evaluation nodes
#       if self.use_neighborlist:
#           Ea, Qa, Dij, nhloss = nn.atomic_properties(self.Z, self.R, self.idx_i, self.idx_j, self.offsets, self.sr_idx_i, self.sr_idx_j, self.sr_offsets)
#       else:
#           Ea, Qa, Dij, nhloss = nn.atomic_properties(self.Z, self.R, self.idx_i, self.idx_j, self.offsets)
#       self._charges =nn.scaled_charges(self.Z, Qa, self.Q_tot)
#       self._energy, self._forces = nn.energy_and_forces_from_scaled_atomic_properties(Ea, self.charges, Dij, self.Z, self.R, self.idx_i, self.idx_j)

        #calculate properties once to initialize everything
        self._calculate_all_properties(atoms)

    def calculation_required(self, atoms, quantities=None):
        return atoms != self.last_atoms

    def _calculate_all_properties(self, atoms):
        #find neighbors and offsets
        if self.use_neighborlist or any(atoms.get_pbc()):
            idx_i, idx_j, S = neighbor_list('ijS', atoms, self.lr_cutoff)
            offsets = np.dot(S, atoms.get_cell())
            sr_idx_i, sr_idx_j, sr_S = neighbor_list('ijS', atoms, self.sr_cutoff)
            sr_offsets = np.dot(sr_S, atoms.get_cell())
            feed_dict = {self.Z: atoms.get_atomic_numbers(), self.R: atoms.get_positions(), 
                    self.idx_i: idx_i, self.idx_j: idx_j, self.offsets: offsets,
                    self.sr_idx_i: sr_idx_i, self.sr_idx_j: sr_idx_j, self.sr_offsets: sr_offsets}
        else:
            N = len(atoms)
            idx_i = np.zeros([N*(N-1)], dtype=int)

            idx_j = np.zeros([N*(N-1)], dtype=int)
            offsets = np.zeros([N*(N-1),3], dtype=float)
            count = 0
            for i in range(N):
                for j in range(N):
                    if i != j:
                        idx_i[count] = i
                        idx_j[count] = j
                        count += 1
            feed_dict = {self.Z: atoms.get_atomic_numbers(), self.R: atoms.get_positions(), 
                    self.idx_i: idx_i, self.idx_j: idx_j, self.offsets: offsets}

        #calculate energy and forces (in case multiple NNs are used as ensemble, this forms the average)
        self._last_energy, self._last_forces, self._last_charges = self.sess.run([self.energy, self.forces, self.charges], feed_dict=feed_dict)
        self._energy_stdev = 0

        # Kazuumi's addition
        self._last_energy = self._last_energy * self.E_to_eV
        self._last_forces = self._last_forces * self.F_to_eV_Ang

        self._last_energy = np.array(1*[self.last_energy]) #prevents some problems...

        #store copy of atoms
        self._last_atoms = atoms.copy()

    def get_potential_energy(self, atoms, force_consistent=False):
        if self.calculation_required(atoms):
            self._calculate_all_properties(atoms)
        return self.last_energy

    def get_forces(self, atoms):
        if self.calculation_required(atoms):
            self._calculate_all_properties(atoms)
        return self.last_forces

    def get_charges(self, atoms):
        if self.calculation_required(atoms):
            self._calculate_all_properties(atoms)
        return self.last_charges

###################################################################################################################

    @property
    def sess(self):
        return self._sess

    @property
    def last_atoms(self):
        return self._last_atoms

    @property
    def last_energy(self):
        return self._last_energy

    @property
    def last_forces(self):
        return self._last_forces

    @property
    def last_charges(self):
        return self._last_charges

    @property
    def energy_stdev(self):
        return self._energy_stdev

    @property
    def sr_cutoff(self):
        return self._sr_cutoff

    @property
    def lr_cutoff(self):
        return self._lr_cutoff

    @property
    def use_neighborlist(self):
        return self._use_neighborlist

    @property
    def nn(self):
        return self._nn

    @property
    def Z(self):
        return self._Z

    @property
    def Q_tot(self):
        return self._Q_tot

    @property
    def R(self):
        return self._R

    @property
    def offsets(self):
        return self._offsets

    @property
    def idx_i(self):
        return self._idx_i

    @property
    def idx_j(self):
        return self._idx_j

    @property
    def sr_offsets(self):
        return self._sr_offsets

    @property
    def sr_idx_i(self):
        return self._sr_idx_i

    @property
    def sr_idx_j(self):
        return self._sr_idx_j

    @property
    def energy(self):
        return self._energy

    @property
    def forces(self):
        return self._forces

    @property
    def charges(self):
        return self._charges

