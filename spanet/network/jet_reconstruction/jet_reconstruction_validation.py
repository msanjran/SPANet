from typing import Dict, Callable
import warnings
from collections import defaultdict
import os
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from sklearn import metrics as sk_metrics

from spanet.options import Options
from spanet.dataset.evaluator import SymmetricEvaluator
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
from spanet.network.utilities.divergence_losses import assignment_cross_entropy_loss, jensen_shannon_divergence


class JetReconstructionValidation(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetReconstructionValidation, self).__init__(options, torch_script)
        self.evaluator = SymmetricEvaluator(self.training_dataset.event_info)
        if self.balance_particles:
            self.particle_index_tensor_np = self.particle_index_tensor.cpu().detach().numpy()
            self.particle_weights_tensor_np = self.particle_weights_tensor.cpu().detach().numpy()
        # self.validation_step_metrics_outputs = []
        self.aggregate_metrics = {} # store metrics like 'classif_i_acc_num' 
        # self.aggregate_probabs = {} # store probabilities per classif. (DEBUG ONLY)
        # self.debug_dir = f"{self.logger.log_dir}/debug_validation"
        # self.debug_dir = "debug_validation"
        # os.makedirs(self.debug_dir, exist_ok=True)

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

        # Apply permutation groups for each target
        for target, prediction, decoder in zip(stacked_targets, jet_predictions, self.branch_decoders):
            for indices in decoder.permutation_indices:
                if len(indices) > 1:
                    prediction[:, indices] = np.sort(prediction[:, indices])
                    target[:, indices] = np.sort(target[:, indices])

        # Old (pre 16may25)
        # metrics.update(self.compute_metrics(jet_predictions, particle_scores, stacked_targets, stacked_masks, classifications, classification_targets))
        metrics.update(self.compute_metrics(jet_predictions, particle_scores, stacked_targets, stacked_masks, stacked_weights))

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

        for name, value in metrics.items():
            if not np.isnan(value):
                self.log(name, value, sync_dist=True, on_epoch=True)

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
            plt.savefig(os.path.join(self.debug_dir, f"probs_epoch_{self.current_epoch}.pdf"))
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

    def on_validation_epoch_end(self):
        ''' 
            Called at end of epoch --> we want to aggregate some metrics here for
            more robust calculations
        '''
        # Allow us to calculate aggregate metrics afterwards
        # print(f"Validation epoch end, aggregate metrics:")
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


