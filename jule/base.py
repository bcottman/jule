#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Bruce_H_Cottman"
__license__ = "MIT License"

from abc import ABC

import logging
import pydot_ng as pydot
from IPython.display import Image, display
import sys, os, timeit, random, math
#for parameter files
import yaml
from attrdict import AttrDict
import numpy as np
from numba import jit
import copy
import pandas as pd
from pandas.core.dtypes.generic import ABCDataFrame, ABCIndexClass, ABCSeries
from pandas.util._validators import validate_bool_kwarg
from tqdm import tqdm
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import seaborn as sns
#
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD,rmsprop
from keras.utils import print_summary,plot_model

import warnings

warnings.filterwarnings("ignore")



class juleError(Exception):
    pass


class parameters(object):
    """
    Read in from file(s) the parameters for this service.
    Currently the __init__ will read in default.yml which sets the
    data_environment parameter. In future top level parameters are set
    so this class can bootstrap to other more environment specialized files.
    currently need only:

        default.yml
        experiment-1 (optional)

    """

    def __init__(self):
        """
        Bootstrap parameter files.

        Parameters:
            None

        Returns:
            self (definition of __init__ behavior).

        Note:
            Currently bootstrap to <name>.yaml or S3.yaml from default.yaml but
            any <na00me>.yaml can be used
            without change to current cede. Only new code need be added to
            support <nth name>.yaml.
            Notice instance is different on call to class init but resulting
            parameter dictionary is always
            the same. This means class parameters can be called from anywhere
            to give same parameters.
            It also means if dafault.yaml or underlying <name>.yaml is changed,
            parameters class instance is set
            again with a resulting new parameters dictionary.

        Example:

            >>> parm_D = parameters().parameters_D

        """
        self.parameters_D = None
        default_D = self._read_parameters()

        if 'experiment_environment'in default_D and 'parameter_directory_path' in default_D:
            self.parameters_D = self._read_parameters(default_D['parameter_directory_path']\
                                                      +default_D['experiment_environment']+'.yaml')
        else:
            raise NameError(
                "read_jule_parameters: experiment_environment does not exist(read from default.yaml):{}".format(default_D))

    def _read_parameters(self,filepath="../parameters/default.yaml"):
        if os.path.exists(filepath):
            with open(filepath) as f:
                config = yaml.load(f)
                return AttrDict(config)
        else:
            raise NameError(
                "read_jule_parameters: The file does not exist:{}".format(filepath)
        )


class CNN(object):
    def __init__(self):
        self._block = 1
        self.model = None

    def block(self):
        """
            Parameters: Overridden in parameter file.

            Return: self
        """
        para_D = parameters().parameters_D
        #        print(para_D)
        self._block = para_D['Conv2D']['block']
        return self

    # default is one block
    def Conv2D(self, n_class=10, dropout=[0.25, 0.25]):
        """
        2-layer deep Conv2D: simple CNN usually used (n,m,3)
        for color image classification.
        Parameters: Overridden in parameter file.
                n_class: (int) 1"
                    number of items output in a vector. Each item
                    in the vector corresponds to the probabilty of
                    that index/class.
                dropout: (list) [0.25,0.25]

        Return: self
    """
        # todo grayscale option
        # todo 3d option
        para_D = parameters().parameters_D
        #        print(para_D)
        n_class = para_D['Conv2D']['n_class']
        dropout = para_D['Conv2D']['dropout']
        input_s = para_D['Conv2D']['input_s']
        deep = para_D['Conv2D']['deep']
        if len(dropout) != self._block + 1:
            raise ValueError('dropout list length must be one greater than iblock number')
        model = Sequential()
        # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
        # this applies 32 convolution filters of size 3x3 each.
        for iblock in range(self._block):
            model.add(Conv2D(32, (3, 3), activation='relu'
                             , input_shape=input_s))
            if deep: model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(dropout[iblock]))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(dropout[self._block]))
        model.add(Dense(n_class, activation='softmax'))
        self.model = model
        return self

    def compile(self):
        """
            Parameters: Overridden in parameter file.

            Return: self
        """
        # initiate  optimizer
        para_D = parameters().parameters_D
        opt_name = para_D['Conv2D']['optimizer']
        opt = None
        if opt_name == 'RMSprop':
            opt = rmsprop(lr=0.0001, decay=1e-6)
        elif opt_name == 'SGD':
            opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

        loss = None
        loss = para_D['Conv2D']['loss']

        self.model.compile(loss=loss, optimizer=opt)
        return self

    def plot(self):
        """
            Parameters: None

            Return: self
        """
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from matplotlib.pyplot import figure

        plot_model(self.model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB')
        # figure  in inch units
        figure(num=None, figsize=(10, self._block * 10), facecolor='w', edgecolor='k')
        image = mpimg.imread("model.png")
        plt.imshow(image)
        plt.show()
        return self

    def info(self):
        """
            Parameters:None

            Return: self
        """
        print_summary(self.model)
        return self