import subprocess, time, sys, math
from bolt.applications.scl9hi.preprocessing.expr_builder import ExperimentBuilder, ExperimentEnv
import pickle

from os import makedirs, environ

MODEL = "v1"
TRAIN_MODE = False
DEBUG_DATA = False
HP_TUNING = False
ITERATION = "P7"

if MODEL == "v1":
    GPU = "rtx_12g pascal_12g"
elif MODEL in ["v2", "v3", "v4"]:
    GPU = "rtx_48g"
#GPU = "rtx_24g"
#queue = "rb_regular"
# Train base line time per step:
# 2.5s per batch
# or 0.5s per Image

# Validation Base line time per step
# 20s per batch
# or 4s per Image

time_multi: float = 1
is_test = DEBUG_DATA or not TRAIN_MODE
if "pascal_12g" in GPU:
    # Train base line time per step:
    # 2.5s per batch (v1)
    # or 0.5s per Image

    # Validation Base line time per step
    # 2.7s per batch (v1)
    if MODEL == "v1":
        batch_size = 6
        time_multi = 2.7
    elif MODEL == "v3":
        batch_size = 4
        time_multi = 2.8
elif "rtx_12g" in GPU:
    # Train base line time per step:
    # 1.2s per batch (v1)
    # or 0.2s per Image
    # 1.3s per batch (v3)

    # Validation Base line time per step
    # 1.3s per batch (v1)
    # 1.5s per batch (v3)
    if MODEL == "v1":
        # since we dont know which gpu is used we assume the worst ...
        time_multi = 1.4
        batch_size = 6
    elif MODEL == "v3":
        # since we dont know which gpu is used we assume the worst ...
        time_multi = 2.8
        batch_size = 4

elif "rtx_24g" in GPU:
    # Train base line per step:
    # 1.5s per batch (v2)
    # or 0.25s per Image

    # Validation Base line time per step
    # 1.5s per batch
    if MODEL in ["v2", "v3", "v4"]:
        batch_size = 6
        time_multi = 1.5
    if MODEL  == "v1":
        batch_size = 12
        time_multi = 2.8
    #time_multi = 1.5
    # assume the worst ...
    #batch_size = 12
elif "rtx_48g" in GPU:
    # train base line per step:
    # 2.8s per batch (v2)
    # or 0.25s per Image
    if MODEL in ["v2", "v3", "v4"]:
        batch_size = 12

    if MODEL  == "v1":
        batch_size = 12
    time_multi = 2.8
else:
    is_test = True
    batch_size = 6

exps = ExperimentBuilder(batch_size=batch_size, train_mode=TRAIN_MODE, debug=DEBUG_DATA, model=MODEL,
                         hp_tuning=HP_TUNING, iteration=ITERATION)

exper = exps.get_tasks()
# with open("runs_to_run.pkl", "rb") as file:
#     runlist = pickle.load(file)
# tasks = []
# for task in runlist[2]:
#     if task[2] == MODEL:
#         for exp in exper:
#             needed = True
#             for item in task:
#                 if item and item not in exp.task_name:
#                     needed = False
#                     break
#                 if item == None and ("kp2d" in exp.task_name or "kp3d" in exp.task_name):
#                     needed = False
#                     break

#             if needed:
#                 tasks.append(exp)
#                 break

# for real kendal train:
# 17,11 missing for finish P5 (just inference) v1
# 12, 13, missing for finish P6 (just inference) v1
# 11, 12, 13, missing for finish P7 (just inference) v1
allowed = ["train_10"]
#allowed = ["train_10", "train_13"]
tasks = []
for exp in exper:
    for name in allowed:
        if name in exp.task_name:
            tasks.append(exp)

#tasks = [exp for exp in exper if "AGORA" in exp.task_name ]
#exper = exps.get_special()
# 147, 144, 129, (v1)
# batch_size = 12
# exp = list(exper)[12]
queue = "rb_regular"



for exp in tasks:
    test_subdir = ""
    if is_test and MODEL in ["v1", "v4"]:
        if exp.freeze_backend:
            hours = 24
            queue = "rb_regular"
            continue
        elif exp.train_steps:
            hours = 4
            queue = "rb_regular"
            continue
        else:
            hours = 2
            queue = "rb_short"
            #continue
        test_subdir = "tests/"
    elif HP_TUNING:
        hours = 48
        queue = "rb_regular"
    else:
        total_time = (exp.epochen*(((exps.TRAIN_IMG/batch_size)*time_multi) + ((exps.VALD_IMG/batch_size)*time_multi))) + (12*3600)  # add 12 hours just in case ...
        hours = math.ceil(total_time/3600)
        if hours > 168:
            queue = "rb_long"
    RESULT_DIR = f'{environ["RB_RESULTS_DIR"]}{ITERATION}/{test_subdir}{exp.task_name}/'
    makedirs(RESULT_DIR + "logs", exist_ok=True)

    #exp.task_name = "d_" + exp.task_name
    #hours = 168
    #queue = "rb_regular"
    #GPU = "rtx_12g"
    #queue = "rb_regular"
    #hours = 168
    content = f'''#
    #BSUB -q {queue}                  # fill in queue name (rb_regular, rb_large, ...)
    #BSUBP -J {exp.task_name}                    # job name
    #BSUB -e  {RESULT_DIR}logs/error.log            # optional: Have errors written to specific file
    #BSUB -o  {RESULT_DIR}logs/output.log             # optional: Have output written to specific file
    #BSUB -W {hours}:00                 # optional: fill in desired wallclock time (hours are optional). If omitted, one week is assumed
    #BSUB -M 50G                        # fill in required amount of RAM (in Mbyte)
    #BSUB -gpu num=1:mode=exclusive_process   # use n GPU (in exclusive process mode)
    #BSUB -R span[hosts=1]                    # Ensure the job only runs on one server
    # #BSUB -Is                               # optional: submit an interactive job
    # BSUB -m "{GPU}"                      # optional: submit job to specific nodes, rtx_12g stands for RTX 2080Ti node group, pascal_12g for GTX 1080 Ti, compute for both
    # # BSUB -m "hi-036l"      # just to run on the hi-036l
    # # BSUB -m "hi-032l"
    # Load Modules
    # source /home/scl9hi/bachelorthesis/bolt/bin/activate
    # Here comes your code. These are just regular bash commands
    python3 /home/scl9hi/bachelorthesis/bolt_mfsd/configs/bachelorthesis/main_learning.py -exp_name {exp.task_name} -batch_size {batch_size} -train_mode {TRAIN_MODE} -debug_mode {DEBUG_DATA} -model {MODEL} -hp {HP_TUNING} -iter {ITERATION}'''

    print("Submitting",exp.task_name)
    print(f'Estimated Time {hours}h')
    p = subprocess.Popen(["bsub"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    stdout, stderr = p.communicate(content)

    # Gib die Ausgaben des Subprozesses aus
    print("Stdout:", stdout)
    print("Stderr:", stderr)
