# Copyright (C) Robert Bosch GmbH 2023.
#
# All rights reserved, also regarding any disposal, exploitation,
# reproduction, editing, distribution, as well as in the event of
# applications for industrial property rights.
#
# This program and the accompanying materials are made available under
# the terms of the Bosch Internal Open Source License v4
# which accompanies this distribution, and is available at
# http://bios.intranet.bosch.com/bioslv4.txt

from bolt.applications.scl9hi.preprocessing.datasets import Dataset, Datasets
from typing import Union
from dataclasses import dataclass
from os import environ, path, listdir
import glob, math
import numpy as np
from os import makedirs, environ
# if jobid == 5:
#     LEARNING_TASK_NAME = "mlflow_06_jta_bbox_only"
#     TRAIN_WITH_KP = False
#     TRAIN_DATASET = Dataset.JTA
#     VALIDATION_DATASET = Dataset.JTA
#     THREE_DIMS = False
# Hyperparameter
# RESULT_DIR = f"/home/scl9hi/bachelorthesis/results/{LEARNING_TASK_NAME}/"
# TRAIN_STEPS = 0  # Normal is 4000 for KI_A
# VALID_STEPS = 0  # Normal is 3333 for KI_A
# EPOCHS = 15  # Lets set it 15
# WEIGHTS = "DEFAULT"  # "DEFAULT"
# PRETRAINED_MODEL_PATH = None  # "/home/scl9hi/results/02_multitask_learning_big/models/20230523-135003_nweo_obj_det_tutorial_sol_model_000003.hdf5"
# DEBUG = True

@dataclass
class ExperimentEnv:
    task_name: str # this has to be unique

    train_dataset: Dataset
    trained_on: Dataset
    batch_size: int

    epochen: int
    kp_2d: bool
    kp_3d: bool
    inst: bool

    train_steps: Union[int, None]
    val_steps: int

    weights: str
    pretrained_model_path: Union[None, str]

    debug: bool
    model: str = "v1"
    abs_kp3d: bool = True
    is_test: bool = False
    learning_rate: float = 6e-05
    weight_decay: float = 0.005
    freeze_backend: bool = False
    use_kendal: bool = False
    custom_weights: bool = False
    no_shared_weighting: bool = False
    use_no_weighter: bool = False

class ExperimentBuilder:
    """ it is responsible for generating the corresponding experiments
    """
    def __init__(self, train_mode: bool = True, debug: bool = False,
                 batch_size=6, model: str= "v1",
                 hp_tuning=False, iteration: str ="P1", with_agora: bool = True) -> None:
        # uses the train and validation class for training
        self.train_on = [Datasets.JTA, Datasets.MOTS, Datasets.SHIFT, Datasets.KI_A, Datasets.KI_A_S, Datasets.MSCOCO]
        if with_agora:
            self.train_on.append(Datasets.AGORA)
        # Here we use the validation split for performance check
        # and we use the train for finetuning
        self.test_on = [Datasets.CITYPERSONS, Datasets.MOT17, Datasets.BDD100K, Datasets.NUIMAGES, Datasets.MSCOCO]
        self.train_mode = train_mode
        self.debug = debug
        self.batch_size = batch_size
        self.TRAIN_IMG = 30000
        self.VALD_IMG = 5000
        self.model = model
        self.hp_tuning = hp_tuning
        self.iteration = iteration
        self.generate_exps()

        self.special = {}
        # this are the experiments that needed to be stopped and rerun
        # for name in ["v4_train_05_MOTS"]:
        #     redo_name = "redo_"+name
        #     if name in self.train_exps:
        #         old_model_pth = self.get_models(self.train_exps[name])[-1]
        #         old_end_epoch = int(old_model_pth.split('/')[-1].split('_')[1].split('.')[0])+1
        #         self.special[redo_name] = self.train_exps[name]
        #         self.special[redo_name].task_name = redo_name

        #         self.special[redo_name].pretrained_model_path = old_model_pth
        #         self.special[redo_name].epochen = self.train_exps[name].epochen - old_end_epoch

    def gen_exp_name(self, dataset_name, i, kp_2d, kp_3d, inst, hp=None):
        if self.hp_tuning:
            exp_name = f'lr_{dataset_name}_{i:02d}_{hp}'
        else:
            exp_name = f'{i:02d}_{dataset_name}'
            if kp_2d:
                exp_name += '_kp2d'
            if kp_3d:
                exp_name += '_kp3d'
            if inst:
                exp_name += '_inst'
            if self.debug:
                exp_name = 'debug_'+exp_name
            else:
                exp_name = 'train_'+exp_name
        exp_name = self.model+"_"+exp_name
        return exp_name

    def generate_exp(self, exp_name, dataset, kp_2d, kp_3d, inst, train_steps, weights, pretrained_model_path,
                debug, lr=None, weight_decay=None):
        if lr is None:
            if self.model in ["v2", "v3"]:
                lr = 0.0005
            elif self.model == "v1":
                lr = 6e-05  # old v2 value: 1e-06
            elif self.model == "v4":
                lr = 5e-05
        if weight_decay is None:
            weight_decay = 0.05 if self.model == "v2" else 0.005 # old v2 value: 1e-04
        if self.hp_tuning:
            epochen = 15
        else:
            epochen = 60 if self.model == "v1" else 100
        return ExperimentEnv(task_name=exp_name,
                train_dataset=dataset,
                trained_on=dataset,
                kp_2d=kp_2d,
                kp_3d=kp_3d,
                inst=inst,
                epochen=epochen,
                abs_kp3d=dataset != Datasets.SHIFT,
                batch_size=self.batch_size,
                #train_steps=0 if self.debug or not dataset.has_debug else math.floor(self.TRAIN_IMG/self.batch_size),
                train_steps=0 if dataset == Datasets.AGORA else train_steps,
                val_steps=0 if self.debug or not dataset.has_debug or dataset == Datasets.AGORA else math.floor(self.VALD_IMG/self.batch_size),
                weights=weights,
                pretrained_model_path=pretrained_model_path,
                is_test=False,
                debug=debug,
                model=self.model,
                learning_rate=lr,
                weight_decay=weight_decay
        )
    def get_models(self, exp):
        # find best model path ...
        folder = f'{environ["RB_RESULTS_DIR"]}{self.iteration}/{exp.task_name}/models/'
        models = []
        try:
            for file in listdir(folder):
                if file.endswith(".hdf5"):
                    models.append(folder+file)
        except:
            models = []
        models.sort()
        return models

    def generate_exps(self):
        # first create train experiments
        train_exps = {}
        i = 1

        if self.hp_tuning:
            # choose Dataset# choose Dataset
            for dataset, name in zip([Datasets.KI_A], ["KI_A"]):
                kp_2d, kp_3d, inst = True, True, False

                # we already evalutated 6e-06
                for lr in [5e-04, 1e-04, 5e-05, 1e-05, 05e-06, 1e-06]:
                    weights = "DEFAULT"
                    pretrained_model_path = None
                    exp_name = self.gen_exp_name(name, i, kp_2d, kp_3d, inst, hp=lr)

                    # check if the dataset is to small to to have a .small file
                    debug = self.debug
                    if not dataset.has_debug:
                        debug = False
                    if dataset == Datasets.SHIFT:
                        train_steps = 0
                    else:
                        train_steps = 0 if self.debug or not dataset.has_debug else math.floor(self.TRAIN_IMG/self.batch_size)
                    exp = self.generate_exp(exp_name, dataset, kp_2d, kp_3d, inst, train_steps, weights, pretrained_model_path, debug, lr=lr)
                    train_exps[exp_name] = exp
                    i+=1



        else:
            # choose Dataset
            for dataset in self.train_on:
                all_datasets = vars(Datasets)
                for dataset_name, dataset_value in all_datasets.items():
                    if dataset == dataset_value:
                        # see if dataset is in our datasets used for training list
                        # create name this is a unique identifier ...

                        # first inst only, then no auxillary, then kp_2d only, then kp_3d only
                        # then all together

                        # TODO add instance seg
                        for kp_2d, kp_3d, inst in zip([False,True,False,True], # kp_2d
                                                    [False,False,True,True], # kp_3d
                                                    [False,False,False, False] # inst
                                                    ):
                            # filter out for shift, since it only has bbox
                            if ((dataset == Datasets.SHIFT and not kp_2d and not kp_3d) or (dataset == Datasets.MSCOCO and not kp_3d)) or (dataset != Datasets.MSCOCO and dataset != Datasets.SHIFT):
                                weights = "DEFAULT"
                                pretrained_model_path = None

                                exp_name = self.gen_exp_name(dataset_name, i, kp_2d, kp_3d, inst)



                                # check if the dataset is to small to to have a .small file
                                if dataset == Datasets.SHIFT:
                                    train_steps = 0
                                else:
                                    train_steps = 0 if self.debug and dataset.has_debug else math.floor(self.TRAIN_IMG/self.batch_size)

                                exp = self.generate_exp(exp_name, dataset, kp_2d, kp_3d, inst, train_steps, weights, pretrained_model_path, self.debug and dataset.has_debug)
                                train_exps[exp_name] = exp
                                i+=1
        # create tests (already pretrained, test on real world dataset ...)
        test_exps = {}
        i = 1
        for dataset in self.test_on:
            all_datasets = vars(Datasets)
            for dataset_name, dataset_value in all_datasets.items():
                if dataset == dataset_value:
                    for exp in train_exps.values():
                        # KIAS is only allowed to be testet on Citypersons
                        if ((dataset == Datasets.CITYPERSONS and exp.train_dataset == Datasets.KI_A_S) or exp.train_dataset != Datasets.KI_A_S):
                            for freeze, finetune_number in  zip([False, False, True],[None, 440, 440]):
                            #for finetune_number in  [None]:
                                if finetune_number is not None:
                                    freezestr = "freeze" if freeze else ""
                                    nr_str = f'{freezestr}{(finetune_number):03d}'
                                else:
                                    nr_str = "No"

                                models = self.get_models(exp)
                                if len(models) > 0:


                                    exp_name = f'test_{i}_{dataset_name}_{nr_str}_f_{exp.task_name}'

                                    # check if a Test is already run for this
                                    # if not self.test_already(exp_name): uncommented since we want to run multiple tests ...
                                    test_exp = ExperimentEnv(task_name=exp_name,
                                                        train_dataset=dataset,
                                                        trained_on=exp.train_dataset,
                                                        kp_2d=exp.kp_2d,
                                                        kp_3d=exp.kp_3d,
                                                        inst=inst,
                                                        epochen=12 if freeze else 1,
                                                        train_steps=finetune_number,
                                                        batch_size=self.batch_size,
                                                        val_steps=0,
                                                        weights=weights,
                                                        is_test=True,
                                                        pretrained_model_path=models[-1],
                                                        debug=self.debug and dataset.has_debug,
                                                        model=self.model,
                                                        learning_rate=3e-07 if self.model == "v2" else 6e-05,
                                                        freeze_backend=freeze)
                                    test_exps[exp_name] = test_exp
                                i+=1
        self.test_exps = test_exps
        self.train_exps = train_exps

    def test_already(self, exp_name):
        exp_folder = f'{environ["RB_RESULTS_DIR"]}{self.iteration}/tests/{exp_name}/images/bp_kp/'
        return path.exists(exp_folder)


    def get_exp(self, exp_name: str):
        if exp_name in self.train_exps.keys():
            return self.train_exps[exp_name]
        elif exp_name in self.test_exps.keys():
            return self.test_exps[exp_name]
        elif exp_name in self.special.keys():
            return self.special[exp_name]
        else:
            print(exp_name, "not found")
            raise ValueError
    def get_tasks(self):
        if self.train_mode:
            return list(self.train_exps.values())
        else:
            return list(self.test_exps.values())
    def get_special(self):
        if self.train_mode:
            return list(self.special.values())
        else:
            return list(self.special.values())
