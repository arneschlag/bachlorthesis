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
import functools

from bolt.applications.scl9hi.postprocessing.callback_graph import DrawImage, PreprocessPoints
from bolt.applications.pose_estimation_3d.evaluation.callbacks import PoseEstimation3DEvaluation, ExtendedPoseEvaluation
from bolt.applications.pose_estimation_3d.evaluation.metrics import MPJPE, PCK

import bolt.applications.keypoint_detection.evaluation.callbacks
import bolt.applications.keypoint_detection.evaluation.evaluation
from bolt.applications.mfsd.evaluation import mfsd_evaluation
import bolt.applications.object_detection.v1.evaluation.callbacks
import bolt.applications.object_detection.v1.evaluation.protocols.caltech
import bolt.evaluation.callbacks
import bolt.rb_keras.callbacks.savelogger
from bolt.applications.object_detection.v1.evaluation.protocols.caltech import generate_protocols
from bolt.rb_keras.callbacks.mlflow import MLflowCallback, ModelLoggingSettings
from bolt.applications.object_detection.v1.evaluation.protocols.caltech import EvalProtocol
from bolt.applications.scl9hi.preprocessing.expr_builder import ExperimentEnv, ExperimentBuilder
from bolt.data.containers import OneHotAttribute
from bolt.backends.torch.callbacks import LearningRateScheduler
from bolt.applications.scl9hi.compute.lr_scheduler import CosineAnnealingLR_w_Warmup
from bolt.core.callbacks import Callback
import os
class LogLearningRate(Callback):
    def on_train_epoch_start(self, epoch, steps, log, data_generator):
        log["LR"] = self.model.optimizer.param_groups[0]['lr']
        print(f'E{epoch}: LR {log["LR"]}')



def build_callbacks(exp: ExperimentEnv, result_dir: str, kp_df, iteration: str ):
    """Are used for generating callbacks which evaluate our model.

    Depend on if training with keypoints is activated, if not no Keypoint stats will be evaluated.

    Returns
    -------
    list of callbacks
    """
    # protocols = {
    #     "reasonable": {"min_height": 50, "min_vis_ratio": 0.65},
    #     "reasonable_small": {"min_height": 50, "max_height": 75, "min_vis_ratio": 0.65},
    #     "reasonable_occ_heavy": {"min_height": 50, "min_vis_ratio": 0.2, "max_vis_ratio": 0.65},
    #     "all": {"min_height": 20, "min_vis_ratio": 0.2},
    # }
    #graph_oe = PreprocessPoints(kp_2d=exp.kp_2d, kp_3d=exp.kp_3d, output_for_object_evaluation=True, kp_df=kp_df)
    #graph_ot = PreprocessPoints(kp_2d=exp.kp_2d, kp_3d=exp.kp_3d, output_for_object_evaluation=False, kp_df=kp_df)
    if iteration == "P1":
        exp_name = "scl9hi_thesis_log_weighter_1"
    if iteration == "P2":
        exp_name = "scl9hi_thesis_log_weighter_2"
    elif iteration == "P3":
        exp_name = "scl9hi_thesis_kendal_weights_were_not_changed"
    elif iteration == "P4":
        exp_name = "scl9hi_thesis_no_weighting"
    elif iteration == "P5":
        exp_name = "scl9hi_thesis_kendal"
    elif iteration == "P6":
        exp_name = "scl9hi_thesis_kendal_2"
    elif iteration == "P7":
        exp_name = "scl9hi_thesis_kendal_3"
    callbacks = [
        # bolt.applications.object_detection.v1.evaluation.callbacks.EvalCallbackObjectDetection(
        #     protocols=generate_protocols(filters={"height_min_50": "[50, inf)*",
        #                                           "height_50_75": "[50, 75]*",
        #                                           "height_min_20": "[20, inf)*",
        #                                           "vis_min_65_height_min_50": "vis_ratio >= 0.65 & height >= 50.0",
        #                                           "vis_min_65_height_50_75": "vis_ratio >= 0.65 & height >= 50.0",
        #                                           "vis_20_65_height_min_50": "vis_ratio >= 0.2 & vis_ratio <= 0.65 & height >= 50.0",
        #                                           "vis_min_65_height_min_20": "vis_ratio >= 0.65 & height >= 20.0",
        #                                           "vis_min_65_height_20_50": "vis_ratio >= 0.65 & height >= 20.0 & height <= 50.0"}),
        #     similarity=None,
        #     store_path=None,#result_dir + "obj_det_stats/",
        #     #post_proc_graph=graph_oe,
        #     prefix="test_{prediction:}" if exp.is_test else "val_{prediction:}_{epoch:03d}",
        # ),

        mfsd_evaluation(
            prefix="test_" if exp.is_test else "val_{epoch:03d}",
            plot_path=result_dir + "obj_det_stats/",
            classes_handler=OneHotAttribute(hierarchy="person", has_bg_class=True)
        ),

        # ),

        bolt.rb_keras.callbacks.savelogger.SaveLogger(
            extension="csv",
            separator=";",
            filename=result_dir + "resnet",
        ),
        MLflowCallback(experiment_name=exp_name,
            tracking_uri=os.environ['MLFLOW_TRACKING_URI'],
            #tracking_uri='http://10.34.211.230/',
            s3_endpoint_url=os.environ['MLFLOW_S3_ENDPOINT_URL'],
            run_name=exp.task_name,
            log_files=dict(logs=result_dir + "logs/"),
            log_model_settings=ModelLoggingSettings(log_model_in_epochs=[]) # do not log the Model
        )
    ]
    callbacks.append(
        bolt.evaluation.callbacks.EvalCallbackWriteImg(
            filepath=result_dir + "images/",
            total_num=25,
            output_formats="{e:03d}.png",
            post_proc_graph=DrawImage(), # TODO move to one object, now it called for every image again
        )
    )
    if not exp.is_test:
        callbacks.append(
            bolt.rb_keras.callbacks.core.ModelCheckpoint(
                filepath=result_dir + "models/model_{epoch:06d}.hdf5",
                mode="min",
                monitor="valid_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
                delete_spare=True
            )
        )

    if exp.kp_2d:
        callbacks.append(
            bolt.applications.keypoint_detection.evaluation.callbacks.EvalCallbackKeypoints(
                output_path=result_dir + "keypoint_stats/",
                #post_proc_graph=graph_ot,
                post_proc_graph=None,
                map_metric=functools.partial(
                    bolt.applications.keypoint_detection.evaluation.evaluation.calculate_oks,
                    constants=kp_df.constants,
                    invert=True,
                ),
                scale_to_gt_resolution=False,
                evaluate_occlusion=False,
            )
        )
        callbacks.append(
            bolt.applications.keypoint_detection.evaluation.callbacks.EvalCallbackExportKeypoints(
                output_path=result_dir + "keypoint_results/",
                name_key="label",
                #post_proc_graph=graph_ot,
                post_proc_graph=None,
                limbs_subset_to_draw=kp_df.limbs,
                export_json=False,
                export_images=True,
                center_keypoints=False
            )
        )
    if exp.kp_3d:
        callbacks.append(
            PoseEstimation3DEvaluation(#post_proc_graph=graph_ot,
                                       post_proc_graph=None,
                                       net_output_key_3d="keypoints_3d",
                                       target_key_3d="keypoints_3d",
                                       dir=result_dir + "3d_results/",
                                       metrics=[MPJPE(compute_root_relative=False, per_joint=True), PCK(), MPJPE(compute_root_relative=False, per_joint=False)],
                                       print_out=True, uwe=exp.task_name == 'd_v1_debug_23_AGORA_kp2d_kp3d')
        )
    if exp.model in ["v2", "v3"]:
        # Set hyperparameters
        warmup_epochs = 5

        scheduler = functools.partial(CosineAnnealingLR_w_Warmup,
                         T_max=exp.epochen,
                         warmup_epochs=warmup_epochs,
                         target_lr=exp.learning_rate)

        callbacks.append(LearningRateScheduler(scheduler_builder=scheduler)
        )
        callbacks.append(LogLearningRate())

    return callbacks
