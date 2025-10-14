from typing import Dict, Callable
import warnings
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from datetime import datetime # debugging
import json
import copy

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Metric as TMetric
from torchmetrics import MetricCollection as TMetricCol
# import torch.distributed as dist

from sklearn import metrics as sk_metrics

from spanet.options import Options
from spanet.dataset.evaluator import SymmetricEvaluator
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
from spanet.network.utilities.divergence_losses import assignment_cross_entropy_loss, jensen_shannon_divergence

class NanReduceMetric(TMetric):
    """
    Distributed-safe nan-aware reduction metric.
    Supports both 'mean' and 'sum' reductions across devices.

    Example:
        nanmean = NanReduceMetric(mode='mean')
        nansum  = NanReduceMetric(mode='sum')

    This avoids poisoning with NaNs and correctly syncs across GPUs.

    credit: ChatGPT but also https://github.com/Lightning-AI/torchmetrics/pull/506
    """
    full_state_update = False  # avoids syncing all samples individually

    def __init__(self, mode: str = "mean"):
        super().__init__()
        if mode not in ("mean", "sum"):
            raise ValueError(f"Unsupported mode '{mode}'. Use 'mean' or 'sum'.")
        self.mode = mode

        # Shared states across both modes
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, values: torch.Tensor):
        """Update internal state with new batch values."""
        if not torch.is_tensor(values):
            values = torch.tensor(values, dtype=torch.float32)

        # Handle NaNs safely — exclude them from the reduction
        mask = ~torch.isnan(values)
        valid_values = values[mask]

        if valid_values.numel() > 0:
            self.sum += valid_values.sum()
            self.count += valid_values.numel()

    def compute(self):
        """Compute the global reduction (mean or sum)."""
        if self.mode == "sum":
            return self.sum
        elif self.mode == "mean":
            return self.sum / torch.clamp(self.count, min=1)


class JetReconstructionValidation(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetReconstructionValidation, self).__init__(options, torch_script)
        self.evaluator = SymmetricEvaluator(self.training_dataset.event_info)
        if self.balance_particles:
            self.particle_index_tensor_np = self.particle_index_tensor.cpu().detach().numpy()
            self.particle_weights_tensor_np = self.particle_weights_tensor.cpu().detach().numpy()
        # self.validation_step_metrics_outputs = []
        self.aggregate_metrics = {} # store metrics like 'classif_i_acc_num' 
        # self.aggregate_props = {} # debugging proportion changes
        # self.save_nan_info = {} # debugging purposes...
        # self.aggregate_probabs = {} # store probabilities per classif. (DEBUG ONLY)
        # self.debug_dir = f"{self.logger.log_dir}/debug_validation"
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # self.debug_dir = f"debug_validation_{timestamp}"
        # os.makedirs(self.debug_dir, exist_ok=True)

        # Create global metric objects (avoids 'nan' issues...)
        # self.nanmean_metric = NanReduceMetric(mode="mean")
        # self.nansum_metric = NanReduceMetric(mode="sum")
        self.nansafe_metrics = {}

    @property
    def particle_metrics(self) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        return {
            "accuracy": sk_metrics.accuracy_score,
            "sensitivity": sk_metrics.recall_score,
            "specificity": lambda t, p: sk_metrics.recall_score(~t, ~p),
            "f_score": sk_metrics.f1_score
        }

    @property
    def particle_score_metrics(self) -> Dict[str, Callable[[np.ndarray, np.ndarray], float]]:
        return {
            # "roc_auc": sk_metrics.roc_auc_score,
            # "average_precision": sk_metrics.average_precision_score
        }

    # Old (pre 16may25)
    # def compute_metrics(self, jet_predictions, particle_scores, stacked_targets, stacked_masks, classifications, classification_targets):

    # New (post 16may25)
    def compute_metrics(self, jet_predictions, particle_scores, stacked_targets, stacked_masks, stacked_weights):
        event_permutation_group = self.event_permutation_tensor.cpu().numpy()
        num_permutations = len(event_permutation_group)
        num_targets, batch_size = stacked_masks.shape
        particle_predictions = particle_scores >= 0.5

        # Compute all possible target permutations and take the best performing permutation
        # First compute raw_old accuracy so that we can get an accuracy score for each event
        # This will also act as the method for choosing the best permutation to compare for the other metrics.
        jet_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=bool)
        weighted_jet_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=bool)
        particle_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=bool)
        for i, permutation in enumerate(event_permutation_group):
            for j, (prediction, target, weight) in enumerate(zip(jet_predictions, stacked_targets[permutation], stacked_weights[permutation])):
                jet_accuracies[i, j] = np.all(prediction == target, axis=1)
                weighted_jet_accuracies[i, j] = np.all(prediction == target, axis=1) * weight

            particle_accuracies[i] = stacked_masks[permutation] == particle_predictions

        jet_accuracies = jet_accuracies.sum(1)
        weighted_jet_accuracies = weighted_jet_accuracies.sum(1)
        particle_accuracies = particle_accuracies.sum(1)

        # Select the primary permutation which we will use for all other metrics.
        chosen_permutations = self.event_permutation_tensor[jet_accuracies.argmax(0)].T
        chosen_permutations = chosen_permutations.cpu()
        permuted_masks = torch.gather(torch.from_numpy(stacked_masks), 0, chosen_permutations).numpy()

        # Compute final accuracy vectors for output
        num_particles = stacked_masks.sum(0)
        tot_target_weights = (stacked_masks * stacked_weights).sum(0)
        jet_accuracies = jet_accuracies.max(0)
        weighted_jet_accuracies = weighted_jet_accuracies.max(0)
        particle_accuracies = particle_accuracies.max(0)

        # Create the logging dictionaries
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
    
            metrics = {f"jet/accuracy_{i}_of_{j}": (jet_accuracies[num_particles == j] >= i).mean()
                    for j in range(1, num_targets + 1)
                    for i in range(1, j + 1)}

            metrics.update({f"particle/accuracy_{i}_of_{j}": (particle_accuracies[num_particles == j] >= i).mean()
                            for j in range(1, num_targets + 1)
                            for i in range(1, j + 1)})

        particle_scores = particle_scores.ravel()
        particle_targets = permuted_masks.ravel()
        particle_predictions = particle_predictions.ravel()

        for name, metric in self.particle_metrics.items():
            metrics[f"particle/{name}"] = metric(particle_targets, particle_predictions)

        for name, metric in self.particle_score_metrics.items():
            metrics[f"particle/{name}"] = metric(particle_targets, particle_scores)
        
        # Old (pre 16may25s) --> need to add ? why did i even add this ?
        # for key in classifications:
        #     accuracy = (classifications[key] == classification_targets[key])
        #     metrics[f"classifications/{key}_accuracy"] = accuracy.mean()


        # Compute the sum accuracy of all complete events to act as our target for
        # early stopping, hyperparameter optimization, learning rate scheduling, etc.
        metrics["validation_accuracy"] = metrics[f"jet/accuracy_{num_targets}_of_{num_targets}"]

        has_targets = tot_target_weights > 0
        weighted_avg_jet_accuracy = weighted_jet_accuracies[has_targets] / tot_target_weights[has_targets]
        metrics["validation_average_jet_accuracy"] = np.mean(weighted_avg_jet_accuracy)

        return metrics
    
    # def on_validation_start(self):
    #     ''' allows us to save output in the right place ... '''
    #     self.debug_dir_val = os.path.join(self.trainer.logger.log_dir, "debug_val")
    #     os.makedirs(self.debug_dir_val, exist_ok=True)

    def validation_step(self, batch, batch_idx) -> Dict[str, np.float32]:
        # Run the base prediction step
        sources, num_jets, targets, regression_targets, classification_targets, item = batch
        jet_predictions, particle_scores, regressions, classifications, classification_scores, outputs = self.predict(sources)

        batch_size = num_jets.shape[0]
        num_targets = len(targets)

        # Stack all of the targets into single array, we will also move to numpy for easier the numba computations.
        stacked_targets = np.zeros(num_targets, dtype=object)
        stacked_masks = np.zeros((num_targets, batch_size), dtype=bool)
        stacked_weights = np.zeros((num_targets, batch_size), dtype=float)
        for i, (target, mask, weight) in enumerate(targets):
            stacked_targets[i] = target.detach().cpu().numpy()
            stacked_masks[i] = mask.detach().cpu().numpy()
            stacked_weights[i] = weight.detach().cpu().numpy()

        regression_targets = {
            key: value.detach().cpu().numpy()
            for key, value in regression_targets.items()
        }

        classification_targets = {
            key: value.detach().cpu().numpy()
            for key, value in classification_targets.items()
        }

        metrics = self.evaluator.full_report_string(jet_predictions, stacked_targets, stacked_masks, prefix="Purity/")
        # nt_types = ['0t','1t','2t','3t','4t','*t']
        # for nt_type in nt_types:
        #     nt_key = f"proportion_{nt_type}_from_purity"
        #     if f"{nt_key}_from_metrics_before_compute_metrics" not in self.aggregate_props:
        #         self.aggregate_props[f"{nt_key}_from_metrics_before_compute_metrics"] = [copy.deepcopy(metrics[f"Purity/{nt_type}/event_proportion"])]
        #     else:
        #         self.aggregate_props[f"{nt_key}_from_metrics_before_compute_metrics"].append(copy.deepcopy(metrics[f"Purity/{nt_type}/event_proportion"]))

        #     metrics[f"{nt_key}_from_metrics_before_compute_metrics"] = copy.deepcopy(metrics[f"Purity/{nt_type}/event_proportion"])
        #     # metrics[f"{nt_key}_from_metrics_before_compute_metrics_DEVICE_{self.device}"] = copy.deepcopy(metrics[f"Purity/{nt_type}/event_proportion"])
        #     self.log(f"{nt_key}_from_metrics_before_compute_metrics_DEVICE_{self.device}", copy.deepcopy(metrics[f"Purity/{nt_type}/event_proportion"]), sync_dist=False, on_epoch=True)


        # Apply permutation groups for each target
        for target, prediction, decoder in zip(stacked_targets, jet_predictions, self.branch_decoders):
            for indices in decoder.permutation_indices:
                if len(indices) > 1:
                    prediction[:, indices] = np.sort(prediction[:, indices])
                    target[:, indices] = np.sort(target[:, indices])

        # Old (pre 16may25)
        # metrics.update(self.compute_metrics(jet_predictions, particle_scores, stacked_targets, stacked_masks, classifications, classification_targets))
        metrics.update(self.compute_metrics(jet_predictions, particle_scores, stacked_targets, stacked_masks, stacked_weights))
        # nt_types = ['0t','1t','2t','3t','4t','*t']
        # for nt_type in nt_types:
        #     nt_key = f"proportion_{nt_type}_from_purity"
        #     metrics[f"{nt_key}_from_metrics_before_compute_metrics_2"] = copy.deepcopy(metrics[f"Purity/{nt_type}/event_proportion"])

        # hardcoded for debugging purposes (specifically for 4t):
        # debugging the changing proportions... --> tests whether the purity calculation is dodgy
        # nt_types = ['0t','1t','2t','3t','4t','*t']
        # our_nt = stacked_masks.T.astype(int).sum(axis=1)
        # for nt_type in nt_types:

        #     nt_key = f"proportion_{nt_type}_from_purity"
        #     metrics[f"{nt_key}_from_metrics_before_tensor"] = copy.deepcopy(metrics[f"Purity/{nt_type}/event_proportion"])
        #     if f"{nt_key}_from_metrics_before_tensor" not in self.aggregate_props:
        #         self.aggregate_props[f"{nt_key}_from_metrics_before_tensor"] = [ copy.deepcopy(metrics[f"Purity/{nt_type}/event_proportion"]) ]
        #     else:
        #         self.aggregate_props[f"{nt_key}_from_metrics_before_tensor"].append(copy.deepcopy(metrics[f"Purity/{nt_type}/event_proportion"]))

        #     save_val = torch.tensor(
        #         copy.deepcopy(metrics[f"Purity/{nt_type}/event_proportion"]),
        #         device=self.device,
        #         dtype=torch.float32
        #     )
        #     self.log(nt_key, copy.deepcopy(metrics[f"Purity/{nt_type}/event_proportion"]), sync_dist=True, on_epoch=True)
        #     metrics[f"{nt_key}_from_metrics"] = copy.deepcopy(metrics[f"Purity/{nt_type}/event_proportion"])

        #     if nt_key not in self.aggregate_props:
        #         self.aggregate_props[nt_key] = [save_val]
        #     else:
        #         self.aggregate_props[nt_key].append(save_val)
            
        #     # calculate proportion ourselves...
        #     nt_key = f"proportion_{nt_type}_from_us"
        #     use_n = nt_type.replace("t","")
        #     if use_n == "*":
        #         nt_mask = our_nt >= 1
        #     else:
        #         nt_mask = our_nt == int(use_n)

        #     our_num, our_den = np.sum(nt_mask), nt_mask.shape[0]
        #     our_prop = our_num / our_den
        #     self.log(nt_key, our_prop, sync_dist=True, on_epoch=True)
        #     self.log(f"{nt_key}_NUM", our_num, sync_dist=True, on_epoch=True, reduce_fx=torch.sum)
        #     self.log(f"{nt_key}_DEN", our_den, sync_dist=True, on_epoch=True, reduce_fx=torch.sum)
        #     metrics[f"{nt_key}_from_metrics"] = our_prop
        #     metrics[f"{nt_key}_NUM_from_metrics"] = our_num
        #     metrics[f"{nt_key}_DEN_from_metrics"] = our_den

        #     our_num = torch.tensor(our_num, device=self.device, dtype=torch.int32)
        #     our_den = torch.tensor(our_den, device=self.device, dtype=torch.int32)
        #     our_prop = torch.tensor(our_prop, device=self.device, dtype=torch.float32)
        #     # our_proportion = np.sum(nt_mask) / nt_mask.shape[0]
        #     if nt_key not in self.aggregate_props:
        #         self.aggregate_props[nt_key] = [our_prop]
        #         # also save the raw number of such events
        #         self.aggregate_props[f"{nt_key}_NUM"] = [our_num]
        #         self.aggregate_props[f"{nt_key}_DEN"] = [our_den]
        #     else:
        #         self.aggregate_props[nt_key].append(our_prop)
        #         self.aggregate_props[f"{nt_key}_NUM"].append(our_num)
        #         self.aggregate_props[f"{nt_key}_DEN"].append(our_den)
            

        for key in regressions:
            delta = regressions[key] - regression_targets[key]
            
            percent_error = np.abs(delta / regression_targets[key])
            self.log(f"REGRESSION/{key}_percent_error", percent_error.mean(), sync_dist=True)

            absolute_error = np.abs(delta)
            self.log(f"REGRESSION/{key}_absolute_error", absolute_error.mean(), sync_dist=True)

            percent_deviation = delta / regression_targets[key]
            self.logger.experiment.add_histogram(f"REGRESSION/{key}_percent_deviation", percent_deviation, self.global_step)

            absolute_deviation = delta
            self.logger.experiment.add_histogram(f"REGRESSION/{key}_absolute_deviation", absolute_deviation, self.global_step)

        # print(f"Validation batch {batch_idx} class. accs:")
        for key in classifications:
            accuracy = (classifications[key] == classification_targets[key])
            self.log(f"CLASSIFICATION/{key}_accuracy_val", accuracy.mean(), sync_dist=True)
            # print(f" - {key}:")
            # print(f" - - acc: {accuracy.mean():.4f}")

            # print(f" - - proportions (target):")
            # unique, counts = np.unique(classification_targets[key], return_counts=True)
            # proportions = dict(zip(unique, counts / counts.sum()))
            # for u in proportions:
            #     print(f"   - {u}: {proportions[u]:.4f}")
            
            # print(f" - - proportions (prediction):")
            # unique, counts = np.unique(classifications[key], return_counts=True)
            # proportions = dict(zip(unique, counts / counts.sum()))
            # for u in proportions:
            #     print(f"   - {u}: {proportions[u]:.4f}")

            # enable calculation of uncertainties
            # 1. simple binomial error --> assumes N.p ≥ 5 and N.(1-p) ≥ 5
            acc_error = np.sqrt(accuracy.mean() * (1 - accuracy.mean()) / accuracy.shape[0])
            self.log(f"CLASSIFICATION/{key}_acc_error_val", acc_error, sync_dist=True)
            # print(f" - - err: {acc_error:.4f}")
            # 2. store number of correct and total for epoch end calculation
            #    this will allow us to calculate the uncertainty without assuming anything
            accuracy_num, accuracy_den = accuracy.sum(), accuracy.shape[0]
            # print(f" - - correct: {accuracy_num}")
            # print(f" - - total: {accuracy_den}")

            # Add loss for validation step
            cweight = None if self.balance_classifications else self.classification_weights[key]
            closs = self.calculate_classification_loss(
                outputs.classifications[key],
                classification_targets[key],
                cweight
            )
            # record probabilities every epoch end
            # if f"classification_{key}_probs" not in self.aggregate_probabs:
            #     probs = torch.softmax(outputs.classifications[key], dim=1)
            #     self.aggregate_probabs[f"classification_{key}_probs"] = [probs.detach().cpu()]
            #     self.aggregate_probabs[f"classification_{key}_targs"] = [torch.from_numpy(classification_targets[key])]
            # else:
            #     probs = torch.softmax(outputs.classifications[key], dim=1)
            #     self.aggregate_probabs[f"classification_{key}_probs"] += [probs.detach().cpu()]
            #     self.aggregate_probabs[f"classification_{key}_targs"] += [torch.from_numpy(classification_targets[key])]

            # print(f" - - loss: {closs}")
            self.log(f"loss/classification/{key}_val", closs, sync_dist=True)
            if f"CLASSIFICATION/{key}_acc_num_val" not in self.aggregate_metrics:
                self.aggregate_metrics[f"CLASSIFICATION/{key}_acc_num_val"] = accuracy_num
                self.aggregate_metrics[f"CLASSIFICATION/{key}_acc_den_val"] = accuracy_den
                # self.aggregate_metrics[f"lossagg/classification/{key}_val"] = closs
            else:
                self.aggregate_metrics[f"CLASSIFICATION/{key}_acc_num_val"] += accuracy_num
                self.aggregate_metrics[f"CLASSIFICATION/{key}_acc_den_val"] += accuracy_den
                # self.aggregate_metrics[f"lossagg/classification/{key}_val"] +=  closs
            # todo: add other metrics?
            
            # classification_metrics = {
            #     "sensitivity":sk_metrics.recall_score,
            #     "specificity":lambda t, p: sk_metrics.recall_score(~t, ~p),
            #     "f1_score":sk_metrics.f1_score
            # }
            # for cm in classification_metrics:
            #     self.log(
            #         f"CLASSIFICATION/{key}_{cm}_val",
            #         classification_metrics[cm](classifications[key], classification_targets[key]),
            #         sync_dist=True
            #     )

        if len(self.nansafe_metrics) == 0:
            for name, value in metrics.items():
                if "NUM_from_metrics" in name or "DEN_from_metrics" in name:
                    self.nansafe_metrics[name] = NanReduceMetric(mode="sum").to(self.device)
                else:
                    self.nansafe_metrics[name] = NanReduceMetric(mode="mean").to(self.device)
            self.nansafe_metrics_metcol = TMetricCol(self.nansafe_metrics)
        
        for name, tmet in self.nansafe_metrics_metcol.items():
            tmet(metrics[name])
            self.log(name, tmet, on_epoch=True)

        # for name, value in metrics.items():
        #     # NOT SKIPPING NANS
        #     # HANDLING THEM WITH NANSUM OR NANMEAN WORKS BETTER THAN THE ASYNCHRONISATION
        #     # OCCURING FROM IGNORING A NAN MANUALLY LIKE THIS...
        #     # if "NUM_from_metrics" in name or "DEN_from_metrics" in name:
        #     #     self.log(name, value, sync_dist=True, on_epoch=True, reduce_fx=torch.nansum)
        #     # else:
        #     #     self.log(name, value, sync_dist=True, on_epoch=True, reduce_fx=torch.nanmean)

        #     if "NUM_from_metrics" in name or "DEN_from_metrics" in name:
        #         self.nansum_metric(value)
        #         self.log(name, self.nansum_metric, prog_bar=True)
        #     else:
        #         self.nanmean_metric(value)
        #         self.log(name, self.nanmean_metric, prog_bar=True)

        #     # if len(self.nansafe_metrics) == 0:
        #     #     self.nansafe_metrics
            
        #     if np.isnan(value):
        #         self.save_nan_info[f"{name}_{self.device}_{batch_idx}"] = copy.deepcopy({
        #             "name":name,
        #             "value":"nan",
        #             "epoch":int(self.current_epoch),
        #             "device":str(self.device),
        #             "batch":int(batch_idx)
        #         })

            # if not np.isnan(value):
            #     self.log(name, value, sync_dist=True, on_epoch=True)
            #     if "NUM_from_metrics" in name or "DEN_from_metrics" in name:
            #         self.log(name, value, sync_dist=True, on_epoch=True, reduce_fx=torch.sum)
            #     # elif "DEVICE" in name:
            #     #     self.log(name, value, sync_dist=False, on_epoch=True)
            #     else:
            #         self.log(name, value, sync_dist=True, on_epoch=True)
            # else:
            #     print(f"{name} has NAN")
            #     self.save_nan_info[f"{name}_{self.device}_{self.batch_idx}"] = copy.deepcopy({
            #             "name":name,
            #             "value":"nan",
            #             "epoch":int(self.current_epoch),
            #             "device":str(self.device),
            #             "batch":int(self.batch_idx)
            #         })
                # print(f" - device: {self.device}")
                # print(f" - epoch: {self.current_epoch}")
                # print(f" - ")


        # self.validation_step_metrics_outputs.append(metrics)

        return metrics

    def plot_probability(self):
        ''' ignoring weights... '''
        for key in self.aggregate_probabs:
            if "targs" in key: continue

            all_probs = torch.cat(self.aggregate_probabs[key], dim=0)
            all_targs = torch.cat(self.aggregate_probabs[key.replace("probs", "targs")], dim=0)
            # what do we want to plot?
            # for each target --> the associated probability
            
            plt.figure(figsize=(10,10))
            use_bins = np.arange(0,1+0.01,0.01)
            chosen_probs = all_probs[torch.arange(all_probs.shape[0]), all_targs]
            unique = np.unique(all_targs)
            for u in unique:
                plt.hist(chosen_probs[all_targs == u], 
                    bins=use_bins, label=f"{u}", histtype="step")
            plt.xlabel("Prob. of target")
            plt.savefig(os.path.join(self.debug_dir_val, f"probs_epoch_{self.current_epoch}.pdf"))
            plt.close()
        
        # self.aggregate_probabs.clear()

    def calculate_classification_loss(
        self, prediction, target, weight
    ):
        if isinstance(prediction, np.ndarray):
            prediction = torch.from_numpy(prediction).float().to(self.device)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target).long().to(self.device)
        if isinstance(weight, np.ndarray):
            weight = torch.from_numpy(weight).long().to(self.device)
        return F.cross_entropy(
            prediction, target, ignore_index=-1, weight=weight
        )
        

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def plot_proportions(self):
        ''' for debugging , don't use for big runs '''

        nt_types = ['0t','1t','2t','3t','4t','*t']
        for nt_type in nt_types:

            plt.figure(figsize=(10,10))
            prop_purity = self.aggregate_props[f"proportion_{nt_type}_from_purity"]
            prop_ours   = self.aggregate_props[f"proportion_{nt_type}_from_us"]
            plt.plot(np.arange(len(prop_purity)), np.array(prop_purity), color="tab:blue", label="from_purity")
            plt.plot(np.arange(len(prop_ours)), np.array(prop_ours), color="tab:red", label="from_ours")
            plt.xlabel("Batch step (in validation)")
            plt.legend()
            plt.savefig(
                os.path.join(self.debug_dir_val, f"val_{nt_type}_proportions_epoch_{self.current_epoch}.pdf")
            )
            plt.close()

    def check_proportions(self):
        print(f"Checking explicit proportions on {self.device}")
        save_per_device = {}
        plot_per_device = {}
        for key in self.aggregate_props:
            # if "_DEN" 
            # value = self.aggregate_props[key]
            # try:
            if isinstance(self.aggregate_props[key][0], np.generic):
                self.aggregate_props[key] = [
                    torch.tensor(val, dtype=torch.float32, device=self.device) for val in self.aggregate_props[key]]

            # print(f"Checking proportions: {key}")
            values_per_gpu = torch.stack([val for val in self.aggregate_props[key]])
            # except:
                # print()
            if "_NUM" not in key and "_DEN" not in key:
                our_prop_per_gpu = values_per_gpu.mean() # per-batch mean
                # print(f" - {key} (per-batch mean): {value.mean()}")
                our_prop_all_gpu = self.all_gather(our_prop_per_gpu)
                # our_prop_all_gpu = self.all_reduce(our_prop_per_gpu)
                # save_per_device[f"{key}_{self.device}"] = our_prop_per_gpu.tolist()
                save_per_device[f"{key}_gathered"] = our_prop_all_gpu.tolist()
                save_per_device[f"{key}_reduced"] = our_prop_all_gpu.sum().tolist()
                
            if "_NUM" in key:
                plot_per_device[f"{key}_{self.device}"] = values_per_gpu
                our_num_per_gpu = values_per_gpu.sum()
                our_num_all_gpu = self.all_gather(our_num_per_gpu)
                # our_num_red_gpu = self.all_reduce(our_num_per_gpu)

                # save_per_device[f"{key}_{self.device}"] = our_num_per_gpu.tolist()
                save_per_device[f"{key}_gathered"] = our_num_all_gpu.tolist()
                save_per_device[f"{key}_reduced"] = our_num_all_gpu.sum().tolist()

                our_dens_per_gpu = torch.stack([
                    val for val in self.aggregate_props[key.replace("_NUM","_DEN")]
                ])
                our_den_per_gpu = our_dens_per_gpu.sum()

                our_prop_per_gpu = our_num_per_gpu / our_den_per_gpu # per-dataset mean
                our_prop_all_gpu = self.all_gather(our_prop_per_gpu)
                # our_prop_red_gpu = self.all_reduce(our_prop_per_gpu)

                # save_per_device[f"{key}_PROP_{self.device}"] = our_prop_per_gpu.tolist()
                save_per_device[f"{key}_PROP_gathered"] = our_prop_all_gpu.tolist()
                save_per_device[f"{key}_PROP_reduced"] = our_prop_all_gpu.sum().tolist()

            if "_DEN" in key:
                # print(f"{key} {self.device} average per-dataset: {values_per_gpu.mean()}")
                plot_per_device[f"{key}_{self.device}"] = values_per_gpu
                our_den_per_gpu = values_per_gpu.sum()
                our_den_all_gpu = self.all_gather(our_den_per_gpu)
                # our_den_red_gpu = self.all_reduce(our_dem_per_gpu)

                # save_per_device[f"{key}_{self.device}"] = our_den_per_gpu.tolist()
                save_per_device[f"{key}_gathered"] = our_den_all_gpu.tolist()
                save_per_device[f"{key}_reduced"] = our_den_all_gpu.sum().tolist()
        
        savefile = os.path.join(
            self.debug_dir_val, 
            f"val_infoepoch_{self.current_epoch}_device_{self.device}.json"
            # f"val_info_epoch_{self.current_epoch}.json"
        )
        with open(savefile, 'w') as file:
            json.dump(save_per_device, file, indent=4)

        for k in plot_per_device:
            # if "DEN" 
            plt.figure(figsize=(10,10))
            plt.plot(
                np.arange(plot_per_device[k].shape[0]),
                np.array(plot_per_device[k].tolist()),
                color="tab:red",
                label=f"{k}_{self.device}"
            )
            plt.xlabel("val step")
            plt.legend()
            os.makedirs(
                os.path.join(self.debug_dir_val,"plots"),
                exist_ok=True
            )
            plt.savefig(
                os.path.join(self.debug_dir_val,"plots",f"val_info_epoch_{self.current_epoch}_{k}_{self.device}.pdf")
            )
            plt.close()

        # self.plot_proportions()
    
    def save_nan_file(self):
        nan_file = os.path.join(self.debug_dir_val, f"nan_info_epoch_{self.current_epoch}_DEVICE_{self.device}.json")
        with open(nan_file, "w") as file:
            json.dump(self.save_nan_info, file, indent=4)
        return None
    
    def on_validation_epoch_end(self):
        ''' 
            Called at end of epoch --> we want to aggregate some metrics here for
            more robust calculations
        '''
        # Allow us to calculate aggregate metrics afterwards
        # print(f"Validation epoch end, aggregate metrics:")
        # print(f"Checking metrics at epoch end")
        # printout and plots not really built for multigpu
        for key in self.aggregate_metrics:
            value = self.aggregate_metrics[key]
            # print(f" - {key}: {value}")
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            # Ensure it's a scalar (Python float/int)
            if isinstance(value, np.ndarray):
                value = value.item()  # <- convert 0-dim array to scalar

            self.aggregate_metrics[key] = value  # overwrite with numpy value

            # might bloat the logging metrics a bit...
            self.log(key, self.aggregate_metrics[key], sync_dist=True, on_epoch=True)
            if "acc_num" in key:
                acc_num = value
                acc_den = self.aggregate_metrics[key.replace("acc_num", "acc_den")]
        
                if isinstance(acc_den, torch.Tensor):
                    acc_den = acc_den.detach().cpu().numpy()
                if isinstance(acc_den, np.ndarray):
                    acc_den = acc_den.item()
                acc = acc_num / acc_den
                self.log(key.replace("acc_num", "acc_acc"), acc, sync_dist=True, on_epoch=True)

                err = np.sqrt(acc * (1 - acc) / acc_den)
                self.log(key.replace("acc_num", "acc_err"), err, sync_dist=True, on_epoch=True)
        # reset the counter per epoch
        # self.check_proportions()
        # self.save_nan_file()
        # self.save_nan_info.clear()
        # self.aggregate_props.clear()
        self.aggregate_metrics.clear()
        # self.plot_probability()
        # self.aggregate_probabs.clear()

    ##################
    # Allow ourselves to calculate loss for validation so we can compare...
    ##################




#    def on_validation_epoch_end(self):
#        # merge metrics from different mini batches into one dict
#        metrics_merged = defaultdict(list) 
#        for m in self.validation_step_metrics_outputs:
#            for key, value in m.items():
#                metrics_merged[key].append(value)
#
#        # average each metric over number of mini batches
#        metrics_averaged = {}
#        for key, values in metrics_merged.items():
#            metrics_averaged[f"mean_{key}"] = np.mean(values)
#
#        # log metrics
#        for name, value in metrics_averaged.items():
#            if not np.isnan(value):
#                self.log(name, value, sync_dist=True)
#
#        self.validation_step_metrics_outputs.clear()


