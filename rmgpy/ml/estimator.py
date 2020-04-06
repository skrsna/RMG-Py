#!/usr/bin/env python3

###############################################################################
#                                                                             #
# RMG - Reaction Mechanism Generator                                          #
#                                                                             #
# Copyright (c) 2002-2019 Prof. William H. Green (whgreen@mit.edu),           #
# Prof. Richard H. West (r.west@neu.edu) and the RMG Team (rmg_dev@mit.edu)   #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the 'Software'),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
###############################################################################

import contextlib
import os
from argparse import Namespace
from typing import Callable, Union

try:
    import dgl
    import torch
except ImportError as e:
    dgl, torch = None, None
    dgl_exception, torch_exception = e,e
    
import numpy as np

from rmgpy.molecule import Molecule
from rmgpy.species import Species
from rmgpy.thermo import ThermoData
from rmgpy.ml.utils import make_dgl_graph_from_smiles, MPNPool

class MLEstimator:
    """
    A machine learning based estimator for thermochemistry prediction.

    The attributes are:

    ==================== ======================= =======================
    Attribute            Type                    Description
    ==================== ======================= =======================
    `hf298_estimator`    :class:`Predictor`      Hf298 estimator
    `s298_cp_estimator`  :class:`Predictor`      S298 and Cp estimator
    `temps`              ``list``                Cp temperatures
    ==================== ======================= =======================
    """

    # These should correspond to the temperatures that the ML model was
    # trained on for Cp.
    temps = [300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1500.0]

    def __init__(self, hf298_path: str, s298_path: str, cp_path: str):
        self.hf298_estimator = load_estimator(hf298_path)
        self.s298_cp_estimator = load_estimator(s298_path)
        self.cp_estimator = load_estimator(cp_path)
    def get_thermo_data(self, molecule: Union[Molecule, str]) -> ThermoData:
        """
        Return thermodynamic parameters corresponding to a given
        :class:`Molecule` object `molecule` or a SMILES string.

        Returns: ThermoData
        """
        molecule = Molecule(smiles=molecule) if isinstance(molecule, str) else molecule

        hf298 = self.hf298_estimator(molecule.smiles)
        s298 = self.s298_cp_estimator(molecule.smiles)
        cp = np.zeros(len(self.temps))
        cp[:] = self.cp_estimator(molecule.smiles)
        cp0 = molecule.calculate_cp0()
        cpinf = molecule.calculate_cpinf()

        # Set uncertainties to 0 because the current model cannot estimate them
        thermo = ThermoData(
            Tdata=(self.temps, 'K'),
            Cpdata=(cp, 'cal/(mol*K)', np.zeros(len(self.temps))),
            H298=(hf298, 'kcal/mol', 0),
            S298=(s298, 'cal/(mol*K)', 0),
            Cp0=(cp0, 'J/(mol*K)'),
            CpInf=(cpinf, 'J/(mol*K)'),
            Tmin=(300.0, 'K'),
            Tmax=(2000.0, 'K'),
            comment='ML Estimation using DGL and tuned mpnn model'
        )

        return thermo

    def get_thermo_data_for_species(self, species: Species) -> ThermoData:
        """
        Return the set of thermodynamic parameters corresponding to a
        given :class:`Species` object `species`.

        The current ML estimator treats each resonance isomer
        identically, i.e., any of the resonance isomers can be chosen.

        Returns: ThermoData
        """
        return self.get_thermo_data(species.molecule[0])


def load_estimator(model_dir: str) -> Callable[[str], dict]:
    """
    Load saved torch model and return function for evaluating it.
    """
    if dgl is None:
        # Delay dgl ImportError until we actually try to use it
        # so that RMG can load successfully without dgl.
        raise dgl_exception
    
    if len(os.listdir(model_dir)) == 1:
        model = torch.load(os.path.join(model_dir,'model.pth'),map_location='cpu')
        predictor = MPNPool(pooling=model['params']['pooling'],output_dim=model['params']['output_dim'])
        predictor.load_state_dict(model['state_dict'])
        def estimator(smi: str):
            predictor.eval()
            with torch.no_grad():
                pred = predictor(make_dgl_graph_from_smiles(smi))
            if model['params']['output_dim'] == 1: #hf298 and #s298 are single scalars
                return float(pred[0][0].cpu().detach().numpy())
            elif model['params']['output_dim'] == 7:
                return pred[0].cpu().detach().numpy()
    
    elif len(os.listdir(model_dir)) > 1:
        saved_models = os.listdir(model_dir)
        models = dict([(model.split('_')[0],torch.load(os.path.join(model_dir,model),map_location='cpu')) for model in saved_models])
        preds = dict()
        def estimator(smi: str):
            for prop, loaded_checkpoint in models.items():
                #print("predicting for prop {}".format(prop))
                dgl_graph = make_dgl_graph_from_smiles(smi)
                predictor = MPNPool(pooling=loaded_checkpoint['pooling'])
                predictor.load_state_dict(loaded_checkpoint['state_dict'])
                predictor.eval()
                with torch.no_grad():
                    preds[prop] = float(predictor(dgl_graph)[0][0].cpu().detach().numpy())
            return preds
    return estimator

