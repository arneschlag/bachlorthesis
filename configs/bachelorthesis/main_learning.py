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
from bolt.backends.torch.losses.weighting import AutomaticLossWeighting, LogWeighter, KendallWeighter, NoWeighter
from bolt.applications.object_detection.v1.backends.torch.objectives.objectives import ObjectDetectionLoss
from bolt.applications.scl9hi.postprocessing.callback_graph import PreprocessPoints
from bolt.applications.scl9hi.compute.losses import build_losses
from bolt.applications.scl9hi.compute.model import Model
#postprocessing
from bolt.applications.scl9hi.postprocessing.build_callbacks import build_callbacks


# in debug mode manually select the expermint
gettrace = getattr(sys, 'gettrace', None)




experiment: ExperimentEnv
if gettrace is None or not gettrace():
    # no debugging
    parser = ArgumentParser()
    parser.add_argument('-exp_name', dest='exp_name', type=str, help='Add exp_name to specify exp_name')
    parser.add_argument('-batch_size', dest='batch_size', type=str, help='Add batch_size to specify batch_size')
    parser.add_argument('-train_mode', dest='train_mode', type=str, help='Add train_mode to specify train_mode')
    parser.add_argument('-debug_mode', dest='debug_mode', type=str, help='Add debug_mode to specify debug_mode')
    parser.add_argument('-hp', dest='hp', type=str, help='Add hp to specify hyperparameter')
    parser.add_argument('-model', dest='model', type=str, help='Add model to specify model')
    parser.add_argument('-iter', dest='iter', type=str, help="Which Iteration of Models")
    args = parser.parse_args()
    enviroment = ExperimentBuilder(train_mode=bool(args.train_mode == "True"), debug=bool(args.debug_mode == "True"),
                                   batch_size=int(args.batch_size), model=str(args.model),
                                   hp_tuning=bool(args.hp == "True"), iteration=args.iter)
    if args.exp_name.startswith('d_'):
        # special test_smaller iteration sizes
        experiment = enviroment.get_exp(exp_name=args.exp_name[2:])
        experiment.task_name = "d_no_weighter_" + experiment.task_name
        experiment.use_no_weighter = True
        #experiment.epochen = 15
    else:
        experiment = enviroment.get_exp(exp_name=args.exp_name)
    # experiment.train_steps = 0
    #experiment.task_name = "new_eval_"+experiment.task_name
    # experiment.val_steps = 40
    # experiment.epochen = 10
    ITERATION = args.iter

    # the third iteration uses kendal by default
    if ITERATION in ["P3", "P4"]:
        experiment.use_no_weighter = True

    elif ITERATION in ["P5", "P6", "P7"]:
        experiment.use_kendal = True
    TRAIN_MODE = bool(args.train_mode == "True")
    DEBUG_DATA =  bool(args.debug_mode == "True")
else:
    # The following parameters must be the same between the test submitter and this file!
    # model can be either v1 or v2
    # v1: ResNet-50 (2015)
    # v2: Swin Transformer V2 Base (2021)
    # v3: Swin Transformer V2 Tiny (2021)
    # v4: ConvNext (2022)
    MODEL = "v1"
    TRAIN_MODE = True
    DEBUG_DATA = True
    ITERATION = "P3"

    # import mlflow
    # logged_model = 'runs:/f230881613524a33aef77042e5f7e8a1/best_model'

    # # Load model as a PyFuncModel.
    # loaded_model = mlflow.pytorch.load_model(logged_model)


    #from bolt.models.processors import mlflow_load_model_weights
    # debugging
    enviroment = ExperimentBuilder(train_mode=TRAIN_MODE, debug=DEBUG_DATA, batch_size=6, model=MODEL, with_agora=True)
    # values 3, 7, 8, 12, 16
    task_list = enviroment.get_tasks()
    experiment = task_list[7] #29 (CITYPERSONS), 86 (MOT17), 131 (BDD100K), 176 (NUIMAGES), 221 (MSCOCO)
    experiment.train_steps = 256
    experiment.val_steps = 10
    experiment.task_name = "d_"+experiment.task_name
    experiment.epochen = 5
    #experiment.freeze_backend = False
    #experiment.debug = False
    #experiment.use_no_weighter = True
    experiment.use_kendal = True
    experiment.pretrained_model_path = "/home/scl9hi/mfsd/02_stud/01_schlag/02_results/P3/v1_train_08_MOTS_kp2d_kp3d/models/model_000042.hdf5"


# run the expermient
add_default_stream_handler()

# put short experiments in seperate dir
if DEBUG_DATA or not TRAIN_MODE:
    test_subdir = "tests/"
else:
    test_subdir = ""

RESULT_DIR = f'{environ["RB_RESULTS_DIR"]}{ITERATION}/{test_subdir}{experiment.task_name}/'
makedirs(RESULT_DIR + "images/bp_kp/", exist_ok=True)
makedirs(RESULT_DIR + "keypoint_results", exist_ok=True)
makedirs(RESULT_DIR + "keypoint_stats", exist_ok=True)
makedirs(RESULT_DIR + "models", exist_ok=True)
makedirs(RESULT_DIR + "obj_det_stats", exist_ok=True)
makedirs(RESULT_DIR + "3d_results", exist_ok=True)
makedirs(RESULT_DIR + "logs", exist_ok=True)

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
    model = modelbuilder()
    # weights =mlflow_load_model_weights(model=modelbuilder(), weight_path='/home/scl9hi/bachelorthesis/bolt_mfsd/configs/bachelorthesis/model.pth',
    #                         mlflow_run_id='f230881613524a33aef77042e5f7e8a1', mlflow_tracking_uri='http://10.34.211.230//mlflow/')
    callbacks = build_callbacks(experiment, RESULT_DIR, kp_df, ITERATION)

    if  experiment.train_steps is None:
        factory = StandardTest(steps=experiment.val_steps,
                callbacks=callbacks, validation_mode=True,
                callback_graph=PreprocessPoints(kp_2d=experiment.kp_2d, kp_3d=experiment.kp_3d, output_for_object_evaluation=False, kp_df=kp_df))
    else:
        optimizer = partial(AdamW, lr=experiment.learning_rate, betas=[0.9, 0.999], weight_decay=experiment.weight_decay,
                          eps=1e-06)
        num_task = 2
        if experiment.kp_2d:
            num_task += 1
        if experiment.kp_3d:
            num_task += 2
        factory = StandardTraining(
            train_steps=experiment.train_steps,  # for large dataset it is 20000
            epochs=experiment.epochen,
            optimizer=optimizer,
            loss=AutomaticLossWeighting(
                weighter=KendallWeighter(num_task) if experiment.use_kendal else NoWeighter() if experiment.use_no_weighter else LogWeighter(),
                loss_function=ObjectDetectionLoss(
                    reduce=False,
                    matcher=partial(
                        match_per_anchor,
                        split_size=20000,
                        pos_th=0.5,
                        cor_th=0.5,
                        neg_th=0.5,
                        foreground_similarity=iou_float16,
                        ignore_similarity=iod_float16,
                        forcing_th=0.25,
                    ),
                    losses=build_losses(experiment.kp_2d, experiment.kp_3d, experiment.abs_kp3d),
                ),
            ),
            valid_steps=experiment.val_steps,
            callbacks=callbacks,
            callback_graph=PreprocessPoints(kp_2d=experiment.kp_2d, kp_3d=experiment.kp_3d, output_for_object_evaluation=False, kp_df=kp_df)
        )



    setup = Experiment(
        model_builder=model,
        model_processors=[
            None
            if experiment.pretrained_model_path is None
            else partial(load_model_weights, weight_path= experiment.pretrained_model_path, strict=False)
        ],
        data_generators=build_data(experiment.train_dataset, experiment.trained_on, experiment.debug, batchsize=experiment.batch_size),
        train_factory= None if experiment.train_steps is None else factory,
        test_factory=factory if experiment.train_steps is None else None
    )

    setup()
