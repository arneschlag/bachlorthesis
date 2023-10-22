# this file will collect all trainings and testdata and will create statics

import mlflow, os
from typing import Union, List, Dict
from tqdm import tqdm
import pandas as pd
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from os import makedirs
from matplotlib.ticker import PercentFormatter
from multiprocessing import Manager
from joblib import Parallel, delayed
import seaborn as sns
import pickle
# Set the MLflow tracking server URI
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])  # Replace with your MLflow server URI

client = mlflow.MlflowClient()

# maps a experiment run to experiment id of mlflow
experiment_version = {1: ["23"], # Log Weighter 1
                      2: ["29", "32"], # Log Weighter 2
                      3: ["31"], # No Weights (Kendal Bug) 1
                      5: ["35"], # Kendalweighter 1
                      6: ["37"], # Kendalweighter 2
                      7: ["38"]} # Kendalweighter 3



def is_test(text: str):
    return text.startswith('test_')
def is_train(text: str):
    return text[3:].startswith('train_')
def is_redo(text: str):
    return text.startswith('redo_')

def model_version(text: str, version=None):

    if version == 3:
        return "no_weights_v1"
    elif version == 1 or version == 2:
        if "v1" in text:
            return "log_weighter_v1"
        else:
            return "log_weighter_v4"
    elif version in [5,6,7]:
        if "v1" in text:
            return "kendal_v1"
        else:
            return "kendal_v4"
    else:
        raise ValueError

def is_aux_task(text: str):
    return text in ["kp2d", "kp3d"]

def auxillary_task(text: str):
    if "kp2d_kp3d" in text:
        return "kp2d_kp3d"
    elif "kp2d" in text:
        return "kp2d"
    elif "kp3d" in text:
        return "kp3d"
    else:
        return None

def train_dataset(text: str):
    elements = text.split('_')
    elements.reverse()
    cache = ""
    for element in elements:
        if not is_aux_task(element):
            if element in ['A', 'S']:
                if len(cache) == 0:
                    cache = f'_{element}'
                else:
                    cache = f'_{element}{cache}'
            else:
                return element+cache
    return None
def test_dataset(text: str):
    if is_test(text):
        return text.split('_')[2]
    else:
        return None

def finetune_task(text: str):
    if is_test(text):
        return text.split('_')[3]
    else:
        return None

@dataclass
class Metric:
    # List in order to supper multiple runs, each run gets append to the list
    best_epoch: Dict[str, List[Union[None, int]]]
    # each epoch can have its own value
    value_array: Dict[str, List[List[Union[float, None, int]]]]

    def merge(self, other):
        for key in other.best_epoch:
            if key in self.best_epoch:
                self.best_epoch[key] += other.best_epoch[key]
                self.value_array[key] += other.value_array[key]
            else:
                self.best_epoch[key] = other.best_epoch[key]
                self.value_array[key] = other.value_array[key]
        return

class TestDataBase:
    def __init__(self) -> None:
        self.data = {}
    def add_finetune_task(self, task: str):
        if task not in self.data.keys():
            self.data[task] = {}
    def add_source_dataset(self, task: str, source_dataset: str):
        self.add_finetune_task(task)
        if source_dataset not in self.data[task].keys():
            self.data[task][source_dataset] = {}
    def add_target_dataset(self, task: str, source_dataset: str, target_dataset: str):
        self.add_source_dataset(task, source_dataset)
        if target_dataset not in self.data[task][source_dataset].keys():
            self.data[task][source_dataset][target_dataset] = {}
    def add_auxillary_task(self, task: str, source_dataset: str, target_dataset: str, auxillary_tasks: str):
        self.add_target_dataset(task, source_dataset, target_dataset)
        if auxillary_tasks not in self.data[task][source_dataset][target_dataset].keys():
            self.data[task][source_dataset][target_dataset][auxillary_tasks] = {}
    def add_metric(self, task: str, source_dataset: str, target_dataset: str, auxillary_tasks: str, metric: str, value: Metric):
        self.add_auxillary_task(task, source_dataset, target_dataset, auxillary_tasks)
        if metric not in self.data[task][source_dataset][target_dataset][auxillary_tasks].keys():
            self.data[task][source_dataset][target_dataset][auxillary_tasks][metric] = value
        else:
            self.data[task][source_dataset][target_dataset][auxillary_tasks][metric].merge(value)

class TrainDataBase:
    def __init__(self) -> None:
        self.data = {}
    def add_train_dataset(self, dataset: str):
        if dataset not in self.data.keys():
            self.data[dataset] = {}
    def add_task(self, dataset: str, task: Union[str, None]):
        self.add_train_dataset(dataset)
        if task not in self.data[dataset].keys():
            self.data[dataset][task] = {}
    def add_metric(self,dataset: str, task: Union[str, None], metricname: str, value: Metric):
        self.add_task(dataset, task)
        if metricname not in self.data[dataset][task].keys():
            self.data[dataset][task][metricname] = value
        else:
            self.data[dataset][task][metricname].merge(value)




def to_array(metrics):
    """sometimes it logs metrics for the same step twise, it will filter this out

    Args:
        metrics (_type_): _description_

    Returns:
        _type_: _description_
    """
    my_array = []
    step = -1
    for key in metrics:
        if key.step > step:
            my_array.append(key.value)
            step = key.step
    return my_array


manager = Manager()
metric_history_cache = manager.dict()

def get_metric_history(run_id, metric_name, metric_history_cache):
    # Check if metric history is already in cache
    if (run_id, metric_name) in metric_history_cache:
        return metric_history_cache[(run_id, metric_name)]

    # If not in cache, fetch metric history and add to cache
    metric_history = client.get_metric_history(run_id, metric_name)
    metric_history_cache[(run_id, metric_name)] = metric_history

    return metric_history

# every Model gets its Database
testdb = {"log_weighter_v1": TestDataBase(),
         "no_weights_v1": TestDataBase(),
         "kendal_v1": TestDataBase(),
         "log_weighter_v4": TestDataBase(),
         "kendal_v4": TestDataBase()}
traindb = {"log_weighter_v1": TrainDataBase(),
           "no_weights_v1": TrainDataBase(),
           "kendal_v1": TrainDataBase(),
           "log_weighter_v4": TrainDataBase(),
           "kendal_v4": TrainDataBase()}
metrics = ["reasonable_LAMR", "reasonable_AP",
           "reasonable_small_AP", "reasonable_small_LAMR",
           "reasonable_occ_heavy_AP", "reasonable_occ_heavy_LAMR",
           "valid_loss", "train_loss",
           "PCK", "mAP", "MPJPE",
           "valid_loss_abs_3d", "valid_loss_bbox_coords", "valid_loss_classes", "valid_loss_keypoints", "valid_loss_keypoints_3d",
           "valid_kendall_regularization",
           "valid_classes_weight", "valid_keypoints_weight", "valid_keypoints_3d_weight", "valid_abs_3d_weight", "valid_bbox_coords_weight"]
# Print the names of all the runs.
later_task = []
# Get all runs for the given experiment ID.
tasks = []
print("prepare downloads ...")
for version in experiment_version:
    for experiment_id in experiment_version[version]: # due to a bug, i needed to rerun the for p2 twice
        runs = client.search_runs(experiment_ids=[experiment_id])
        for run in runs:
            for metric in metrics:
                tasks.append((run.info.run_id, metric))
print("Download to local cache ...")
Parallel(n_jobs=128)(
    delayed(get_metric_history)(run_id, metric_name, metric_history_cache)
    for run_id, metric_name in tqdm(tasks)
)
print("Download to local cache done")
with open("cache.pkl", "wb") as file:
  pickle.dump(dict(metric_history_cache), file)

with open("cache.pkl", "rb") as file:
    metric_history_cache = pickle.load(file)
for version in experiment_version:
    for experiment_id in experiment_version[version]: # due to a bug, i needed to rerun the for p2 twice
        runs = client.search_runs(experiment_ids=[experiment_id])
        for run in tqdm(runs):
            # check which are finished
            name = run.info.run_name
            value = 0
            if run.info.status == "FINISHED":
                for metric in metrics:
                    #process_information(client.get_metric_history(run.info.run_id, metric))
                    if is_test(name):
                        # filer the information based whether the finetuning is done over multiple epochs
                        if finetune_task(name) == "freeze440":
                            valid_loss = to_array(get_metric_history(run.info.run_id, "valid_loss", metric_history_cache))

                            #valid_loss = to_array(client.get_metric_history(run.info.run_id, "valid_loss"))
                            if len(valid_loss) > 0:
                                epoch = pd.Series(valid_loss).idxmin()
                            else:
                                break
                        else:
                            epoch = 0
                        values = to_array(get_metric_history(run.info.run_id, metric, metric_history_cache))
                        #values = to_array(client.get_metric_history(run.info.run_id, metric))
                        if len(values) > epoch:
                            value = values[epoch]
                        else:
                            value = None

                        testdb[model_version(name, version)].add_metric(task=finetune_task(name),
                                    source_dataset=train_dataset(name),
                                    target_dataset=test_dataset(name),
                                    auxillary_tasks=auxillary_task(name),
                                    metric=metric,
                                    value=Metric(best_epoch={version: [epoch]}, value_array={version: [values]}))
                    elif is_train(name):
                        valid_loss = to_array(get_metric_history(run.info.run_id, "valid_loss", metric_history_cache))
                        #valid_loss = to_array(client.get_metric_history(run.info.run_id, "valid_loss"))
                        if len(valid_loss) > 0:
                            epoch = pd.Series(valid_loss).idxmin()
                            values = to_array(get_metric_history(run.info.run_id, metric, metric_history_cache))
                            #values = to_array(client.get_metric_history(run.info.run_id, metric))
                            if len(values) > epoch:
                                value = values[epoch]
                            else:
                                value = None
                            traindb[model_version(name, version)].add_metric(dataset=train_dataset(name),
                                                                    task=auxillary_task(name),
                                                                    metricname=metric,
                                                                    value=Metric(best_epoch={version: [epoch]}, value_array={version: [values]}))
                        else:
                            break
                    elif is_redo(name):
                        if [name, run, version] not in later_task:
                            later_task.append([name, run, version])

            else:
                # used if some experiments hung up and werent finished ...
                #if name in ["v1_train_08_MOTS_kp2d_kp3d", "v1_train_12_KI_A_kp3d", "v1_train_10_KI_A", "v1_train_11_KI_A_kp2d"]:
                    # Set the experiment's status to FINISHED.
                    # Update the experiment.
                #    client.update_run(run.info.run_id, status="FINISHED")
                print("Warning!", run.info.run_name, "is still running")

        #Now check for reruns
        for name, run, v in later_task:
            orig_name = name[5:]
            if is_train(orig_name):
                # new element
                valid_loss = to_array(get_metric_history(run.info.run_id, "valid_loss", metric_history_cache))
                epoch = pd.Series(valid_loss).idxmin()

                # TODO reconfigure
                database_element = traindb[model_version(orig_name, v)].data[train_dataset(name)][auxillary_task(name)]
                prev = database_element["valid_loss"]

                savepoint_epoch = prev.best_epoch[1][0]


                # if the new one improved upon the old one:
                for metric in metrics:
                    if metric != "valid_loss":
                        values = to_array(get_metric_history(run.info.run_id, metric, metric_history_cache))

                        if prev.value_array[v][0][savepoint_epoch] > valid_loss[epoch]:
                            database_element[metric].best_epoch[v][0] = len(prev.value_array[v][0][:savepoint_epoch+1])+epoch
                        database_element[metric].value_array[v][0] = database_element[metric].value_array[v][0][:savepoint_epoch+1] + values
                values = to_array(get_metric_history(run.info.run_id, "valid_loss", metric_history_cache))
                database_element["valid_loss"].best_epoch[v][0] = epoch+len(prev.value_array[v][0][:savepoint_epoch+1])
                database_element["valid_loss"].value_array[v][0] = database_element["valid_loss"].value_array[v][0][:savepoint_epoch+1] + values
                traindb[model_version(orig_name, v)].data[train_dataset(name)][auxillary_task(name)] = database_element
            else:
                print("Redo of tests is not supported")
        print("All Data loaded to local database")

with open("runs.pkl", "wb") as file:
  pickle.dump([testdb, traindb], file)


with open("runs.pkl", "rb") as file:
    testdb, traindb = pickle.load(file)


# missing experiments ...
# go threw the testdatabase, to find experiments that need to be run:
ids = {}
for id in experiment_version.keys():
    for version in traindb:
        for dataset in traindb[version].data:
            for task in traindb[version].data[dataset]:
                ids[id, version, dataset, task] = {}
for id in experiment_version.keys():
    for version in testdb:
        for finetune in testdb[version].data:
            for source_dataset in testdb[version].data[finetune]:
                for target_dataset in testdb[version].data[finetune][source_dataset]:
                    for task in testdb[version].data[finetune][source_dataset][target_dataset]:
                        if target_dataset not in ids[id, version, source_dataset, task]:
                            ids[id, version, source_dataset, task][target_dataset] = {}
                        if finetune not in ids[id, version, source_dataset, task][target_dataset]:
                            if len(testdb[version].data[finetune][source_dataset][target_dataset][task]["reasonable_LAMR"].value_array) > id-1:

                                num_tests = len(testdb[version].data[finetune][source_dataset][target_dataset][task]["reasonable_LAMR"].value_array[id])
                                ids[id, version, source_dataset, task][target_dataset][finetune] = num_tests
                            else:
                                ids[id, version, source_dataset, task][target_dataset][finetune] = 1
submitter_list = {1:[], 2:[], 3: []}
for key, value in ids.items():
    for target_dataset, value_1 in value.items():
        if "freeze440" not in value_1:
            submitter_list[key[0]].append((target_dataset, "freeze440", key[1], key[2], key[3]))
            submitter_list[key[0]].append((target_dataset, "freeze440", key[1], key[2], key[3]))
            submitter_list[key[0]].append((target_dataset, "freeze440", key[1], key[2], key[3]))
        elif value_1["freeze440"] < 3:
            for i in range(3-value_1["freeze440"]):
                submitter_list[key[0]].append((target_dataset, "freeze440", key[1], key[2], key[3]))

        if "440" not in value_1:
            submitter_list[key[0]].append((target_dataset, "440", key[1], key[2], key[3]))
        elif value_1["440"] < 1:
            submitter_list[key[0]].append((target_dataset, "440", key[1], key[2], key[3]))

        if "No" not in value_1:
            submitter_list[key[0]].append((target_dataset, "No", key[1], key[2], key[3]))
        elif value_1["No"] < 1:
            submitter_list[key[0]].append((target_dataset, "No", key[1], key[2], key[3]))
with open("runs_to_run.pkl", "wb") as file:
    pickle.dump(submitter_list, file)
print("jobs pending P1:",len(submitter_list[1]))
print("jobs pending P2:",len(submitter_list[2]))
print("jobs pending P3(kendal):",len(submitter_list[3]))

FILL = None
def get_value(metr_obj):
    tmp = []
    for run in metr_obj.best_epoch.keys():
        for i in range(len(metr_obj.best_epoch[run])):
            epoch = metr_obj.best_epoch[run][i]
            if len(metr_obj.value_array[run][i]) > epoch:
                value = metr_obj.value_array[run][i][epoch]
            else:
                value = FILL
            tmp.append(value)

    # average over all runs ...
    # filter out nan values
    tmp = list([item for item in tmp if item != None])

    value = round((sum(tmp)/len(tmp)), 2)
    return value, tmp

# Iterate through the metrics first
for model, database in testdb.items():
    makedirs(model, exist_ok=True)
    for metric in ["reasonable_LAMR", "reasonable_AP",
           "reasonable_small_AP", "reasonable_small_LAMR"]:
        makedirs(f'{model}/{metric}', exist_ok=True)

        for finetune_task_name, finetune_data in sorted(database.data.items()):
            makedirs(f'{model}/{metric}/{finetune_task_name}/', exist_ok=True)

            diff_kp2d = []
            diff_kp2dkp3d = []
            diff_kp3d = []
            for source_dataset_name, dataset_data in sorted(finetune_data.items()):
                # create plot
                plt.style.use('seaborn-v0_8-pastel')
                fig, ax = plt.subplots()
                # we need to calculate the offset of each item
                x = np.arange(len(dataset_data.keys()))
                bar_width = 0.2
                num_groups = len(dataset_data.keys())


                #tasks = list(dataset_data.items())[0][1].keys()
                total_values = [] # list to keep track of max and min of values
                tasks = {}
                target_names = []
                for target_dataset_name, target_data in sorted(dataset_data.items()):

                    # break if baseline is not processed yet ...
                    if None not in target_data.keys():
                        break

                    if metric in target_data[None].keys():
                        metr_obj = target_data[None][metric]
                        base_median, base_values = get_value(metr_obj)
                    else:
                        base_median, base_values = FILL, [FILL]

                    for aux_task_name in [None, "kp2d", "kp3d", "kp2d_kp3d"]:
                        if aux_task_name in target_data and metric in target_data[aux_task_name]:
                            metr_obj = target_data[aux_task_name][metric]
                            median, all_values = get_value(metr_obj)
                        else:
                            median, all_values = FILL, [FILL]

                    # for auxillary_task_name, auxillary_data in sorted(target_data.items(), key=lambda x: (x[0] is None, x[0])):
                    #     if metric in auxillary_data.keys():
                    #         metr_obj = auxillary_data[metric]
                    #         median, all_values = get_value(metr_obj)
                    #     else:
                    #         median, all_values = FILL, [FILL]

                        total_values.append(median)

                        if aux_task_name not in tasks.keys():
                            tasks[aux_task_name] = []
                        tasks[aux_task_name].append((median, all_values))  #todo
                        if target_dataset_name not in target_names:
                            target_names.append(target_dataset_name)

                        if aux_task_name:
                            if base_median is not None and median is not None:
                                diff = base_median-median
                                if aux_task_name == "kp2d":
                                    diff_kp2d.append(diff)
                                if aux_task_name == "kp3d":
                                    diff_kp3d.append(diff)
                                if aux_task_name == "kp2d_kp3d":
                                    diff_kp2dkp3d.append(diff)
                i = 0
                for task_name, median_values_list in tasks.items():
                    offset = bar_width * (i - (num_groups - 1) / 2)
                    medians = np.rot90(np.array(median_values_list, dtype=object))[1]
                    filtered_medians = [0 if v is None else v for v in medians]
                    if task_name is None:
                        handler = ""
                    elif task_name == "kp2d_kp3d":
                        handler = ' + kp2d + kp3d'
                    else:
                        handler = f' + {task_name}'
                    rects = ax.bar(x + offset, filtered_medians, bar_width, label=f'ObjDet{handler} - {metric.split("_")[-1]}')
                    #ax.bar_label(rects, padding=0, label_type='none')
                    for pos, median_values in zip(list(x), median_values_list):
                        median, values = median_values
                        filtered_values = [0 if v is None else v for v in values]
                        rects = ax.boxplot(filtered_values, positions=[pos+offset])
                    i+=1
                ax.set_xlabel('Test Dataset')
                ax.set_ylabel('Values')
                ax.set_title(f'Metric: {metric}, Traindataset: {source_dataset_name}\n Finetuning {finetune_task_name} Model {model}')
                ax.set_xticks(x)
                ax.set_xticklabels(target_names)
                ax.legend()
                plt.ylim(0, 100)

                #ax.set_ylim((min(total_values)-1, max(total_values)+1))
                #ax.set_facecolor('lightgray')


                plt.tight_layout()
                #plt.savefig(f'{model}/{metric}/{finetune_task_name}/plot_from_{source_dataset_name}.pdf')
                plt.savefig(f'metrics/{model}_{metric}_{finetune_task_name}_plot_from_{source_dataset_name}.pdf')
                plt.close()

            overall_avg_diff_kp2d = sum(diff_kp2d) / len(diff_kp2d) if len(diff_kp2d) > 0 else 0
            overall_avg_diff_kp2dkp3d = sum(diff_kp2dkp3d) / len(diff_kp2dkp3d) if len(diff_kp2dkp3d) > 0 else 0
            overall_avg_diff_kp3d = sum(diff_kp3d) / len(diff_kp3d) if len(diff_kp3d) > 0 else 0
            print(f'Overall Average Differences, finetune: {finetune_task_name}:')
            print(f'kp2d vs. Base {metric} Difference: {overall_avg_diff_kp2d}')
            print(f'kp2dkp3d vs. Base {metric} Difference: {overall_avg_diff_kp2dkp3d}')
            print(f'kp3d vs. Base {metric} Difference: {overall_avg_diff_kp3d}')


            for task, data, avg in zip(["kp2d", "kp3d", "kp2d_kp3d"], [diff_kp2d, diff_kp3d, diff_kp2dkp3d], [overall_avg_diff_kp2d, overall_avg_diff_kp3d, overall_avg_diff_kp2dkp3d]):
                plt.style.use('default')
                fig, axs = plt.subplots()
                fig.suptitle(f'{task} - baseline, finetune: {finetune_task_name},\n model: {model}, num_tests: {len(data)}', fontsize=20)

                # We can set the number of bins with the *bins* keyword argument.
                # Create a scatter plot
                sns.kdeplot(data, fill=True, ax=axs)

                y_values = [0] * len(data)  # Create x-axis values
                axs.scatter(data, y_values)


                axs.set_ylabel("Density")
                 # Plot the average value as a vertical line
                axs.axvline(avg, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {avg:.2f}')
                axs.set_xlabel(f'{metric} (base-{task}) lower is better' if "AP" in metric else f'{metric} (base-{task}) higher is better')


                plt.tight_layout()
                #plt.savefig(f'{model}/{metric}/{finetune_task_name}/differences_{task}.pdf')
                plt.savefig(f'metrics/{model}_{metric}_{finetune_task_name}_differences_{task}.pdf')
                plt.close()



task_to_color = {}
task_to_color["kp2d"] = 'b'
task_to_color["kp2d_kp3d"] = 'g'
task_to_color["kp3d"] = 'r'
task_to_color["None"] = 'y'

def extract_refined_metrics(traindb, experiment_name, dataset_name, metric_name):
    refined_metrics = {}

    # Get the TrainDataBase object for the specified experiment
    experiment_data = traindb[experiment_name].data

    # Check if the specified dataset exists
    if dataset_name in experiment_data:
        for task_name, metrics in experiment_data[dataset_name].items():
            if metric_name in metrics:
                metric = metrics[metric_name]

                # Extract values for all runs and refine up to max_epochs
                for run, values in metric.value_array.items():
                    # Use a combined key to represent task and run
                    combined_key = f"{task_name}_run{run}"
                    refined_metrics[combined_key] = values[0]

    return refined_metrics

def compute_average_performance(refined_metrics):
    task_averages = {}
    task_data = {}
    for task_run, values in refined_metrics.items():
        task = task_run.split("_run")[0]
        if task not in task_data:
            task_data[task] = []
        task_data[task].append(values)
    for task, metrics in task_data.items():
        avg_values = [sum(epoch_values) / len(epoch_values) for epoch_values in zip(*metrics)]
        task_averages[task] = avg_values
    return task_averages

def generate_train_plot(traindb, run, datensatz, metric):
    refined_metrics  = extract_refined_metrics(traindb, run, datensatz, metric)
    refined_metrics = compute_average_performance(refined_metrics)
    plt.figure(figsize=(12, 7))
    saveplt = False
    for task_run, values in refined_metrics.items():
        if len(values) > 0:
            saveplt = True
            plt.plot(values, label=task_run, color=task_to_color[task_run])
    if saveplt:
        exp_name = run.replace('v4', 'v2')

        plt.title(f'{exp_name}: {metric} for {datensatz} dataset')
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'plots/{run}_{metric}_{datensatz}.pdf', format='pdf')
    plt.close()

# plot graphs to be used as pdf ...
for run in ["no_weights_v1", "log_weighter_v1", "kendal_v1", "log_weighter_v4", "kendal_v4"]:
    for metric in ["reasonable_AP", "reasonable_LAMR", "valid_loss", "valid_loss_classes", "PCK", "mAP", "MPJPE"]:
        for datensatz in ["SHIFT", "KI_A", "KI_A_S", "JTA", "MOTS", "AGORA"]:
            generate_train_plot(traindb, run, datensatz, metric)




# make a table that plots the differences in training between the different
print("plotting Table Difference during Training")
def generate_latex_table(input_string):
    # Split the input string into lines
    lines = input_string.strip().split("\n")

    # Extract headers and data
    headers = lines[0].split(",")
    data = [line.split(",") for line in lines[1:]]

    # Begin the LaTeX table
    latex_table = """
\\begin{table}
\\centering
\\begin{tabular}{%s}
\\hline
""" % ("|" + "c|"*len(headers))

    # Add headers to the table with _ replaced by \_
    latex_table += " & ".join(["\\rot{" + header.replace("_", "\\_") + "}" for header in headers]) + "\\\\\n\\hline\n"

    # Function to generate cell color
    def cell_color(value):
        if value.replace('.', '', 1).isdigit() or (value.startswith('-') and value[1:].replace('.', '', 1).isdigit()):
            return "\\cellcolorgrade{" + value + "}" + value
        else:
            return value.replace("_", "\\_")

    # Add data rows to the table
    for row in data:
        latex_table += " & ".join([cell_color(item) for item in row]) + "\\\\\n\\hline\n"

    # End the LaTeX table
    latex_table += """
\\end{tabular}
\\caption{Your caption here}
\\label{tab:your_label_here}
\\end{table}
"""
    return latex_table
# go over trainings database
# only works for two repetitions ...
for metric in ["reasonable_LAMR", "reasonable_AP"]:
    type_ = "log_weighter_v1"
    line = f'Dataset,Diff_Base,ObjDet-KP2D,Diff_KP2D,ObjDet-KP3D,Diff_KP3D,ObjDet-KP2D_KP3D,Diff_KP2D_KP3D\n'
    for dataset in sorted(traindb[type_].data):
        # calculate BASELINE
        valid_loss = traindb[type_].data[dataset][None]["valid_loss"]
        base_lamrs = []
        for version in valid_loss.best_epoch:
            epoch = valid_loss.best_epoch[version][0]

            metric_lamr = traindb[type_].data[dataset][None][metric].value_array[version][0][epoch]
            base_lamrs.append(metric_lamr)

        base_lamr = sum(base_lamrs)/len(base_lamrs)
        line += f'{dataset},{str(round(abs(base_lamrs[0]-base_lamrs[1]),1))}'
        for task, item in sorted(traindb[type_].data[dataset].items(), key=lambda x: (x[0] is None, x[0])):
            if task:
                # find lowest loss
                valid_loss = traindb[type_].data[dataset][task]["valid_loss"]

                lamrs = []
                for version in valid_loss.best_epoch:
                    epoch = valid_loss.best_epoch[version][0]

                    metric_lamr = traindb[type_].data[dataset][task][metric].value_array[version][0][epoch]

                    lamrs.append(metric_lamr)
                lamr = sum(lamrs)/len(lamrs)
                if metric == "reasonable_AP":
                    line += f',{str(round(lamr-base_lamr,1))},{str(round(abs(lamrs[0]-lamrs[1]),1))}'
                else:
                    line += f',{str(round(base_lamr-lamr,1))},{str(round(abs(lamrs[0]-lamrs[1]),1))}'
        line += '\n'
    #print(line)
    print(metric)
    print(generate_latex_table(line))
