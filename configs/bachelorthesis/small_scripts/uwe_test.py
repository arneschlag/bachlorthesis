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

# this is just a test file for testing debugging purposes

from functools import partial
import sys
from argparse import ArgumentParser
from os import environ
# preprocessing
from os import makedirs, environ
from bolt.applications.scl9hi.preprocessing.expr_builder import ExperimentBuilder, ExperimentEnv

from bolt.exec import configure_bolt
from bolt.log import add_default_stream_handler
from bolt.applications.scl9hi.preprocessing.dataloader import build_data
from bolt.applications.scl9hi.preprocessing.datasets import Datasets
from bolt.backends.torch.processors import load_model_weights

# compute
from bolt.rb_keras.experiment import Experiment, StandardTraining, StandardTest
from bolt.applications.object_detection.v1.backends.torch.matching.iou import match_per_anchor
from bolt.applications.object_detection.v1.backends.torch.similarity.iou import iou_float16, iod_float16
from torch.optim.adamw import AdamW
from bolt.backends.torch.losses.weighting import AutomaticLossWeighting, LogWeighter
from bolt.applications.object_detection.v1.backends.torch.objectives.objectives import ObjectDetectionLoss
from bolt.applications.scl9hi.postprocessing.callback_graph import PreprocessPoints
from bolt.applications.scl9hi.compute.losses import build_losses
from bolt.applications.scl9hi.compute.model import Model

#postprocessing
from bolt.applications.scl9hi.postprocessing.build_callbacks import build_callbacks


# in debug mode manually select the expermint
gettrace = getattr(sys, 'gettrace', None)

experiment: ExperimentEnv

# The following parameters must be the same between the test submitter and this file!
# model can be either v1 or v2
# v1: ResNet-50 (2015)
# v2: Swin Transformer V2 (2021)
MODEL = "v4"
TRAIN_MODE = True
DEBUG_DATA = True

# debugging
enviroment = ExperimentBuilder(train_mode=TRAIN_MODE, debug=DEBUG_DATA, batch_size=1, model=MODEL)
# values 3, 7, 8, 12, 16
task_list = enviroment.get_tasks()
experiment = task_list[12] # 3, 7, 8, 12, 16
experiment.task_name = "d_"+experiment.task_name
experiment.pretrained_model_path = "/home/scl9hi/mfsd/02_stud/01_schlag/02_results/P2/v4_train_13_KI_A_kp2d_kp3d/models/model_000037.hdf5"
# run the expermient
add_default_stream_handler()

RESULT_DIR = f'{environ["RB_RESULTS_DIR"]}{experiment.task_name}/'
makedirs(RESULT_DIR + "images/bp_kp/", exist_ok=True)
makedirs(RESULT_DIR + "keypoint_results", exist_ok=True)
makedirs(RESULT_DIR + "keypoint_stats", exist_ok=True)
makedirs(RESULT_DIR + "logs", exist_ok=True)
makedirs(RESULT_DIR + "models", exist_ok=True)
makedirs(RESULT_DIR + "obj_det_stats", exist_ok=True)
makedirs(RESULT_DIR + "3d_results", exist_ok=True)

with configure_bolt(
    "all",
    **{
        "backend": "torch",
        "image_data_format": "channels_first",
        "use_amp": True,
        "allow_tf32": True,
        "cudnn_benchmark": True,
    },
):
    # build model
    kp_df = experiment.train_dataset.kp_df if enviroment.train_mode else experiment.trained_on.kp_df
    modelbuilder = Model(experiment, kp_df)
    callbacks = build_callbacks(experiment, RESULT_DIR, kp_df, iteration="P2")

    factory = StandardTest(steps=experiment.val_steps,
            callbacks=callbacks, validation_mode=True,
            callback_graph=PreprocessPoints(kp_2d=experiment.kp_2d, kp_3d=experiment.kp_3d, output_for_object_evaluation=False, kp_df=kp_df))

    setup = Experiment(
        model_builder=modelbuilder(),
        model_processors=[
            None
            if experiment.pretrained_model_path is None
            else partial(load_model_weights, weight_path= experiment.pretrained_model_path, strict=False)
        ],
        data_generators=build_data(experiment.train_dataset, experiment.trained_on, experiment.debug, batchsize=experiment.batch_size, uwe=True),
        train_factory= None if experiment.train_steps is None else factory,
        test_factory=factory if experiment.train_steps is None else None
    )

    setup()
