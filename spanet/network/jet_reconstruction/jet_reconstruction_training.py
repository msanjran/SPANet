from typing import Tuple, Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from sklearn import metrics as sk_metrics

from spanet.options import Options
from spanet.dataset.types import Batch, Source, AssignmentTargets
from spanet.dataset.regressions import regression_loss
from spanet.network.jet_reconstruction.jet_reconstruction_network import JetReconstructionNetwork
from spanet.network.utilities.divergence_losses import assignment_cross_entropy_loss, jensen_shannon_divergence


def numpy_tensor_array(tensor_list):
    output = np.empty(len(tensor_list), dtype=object)
    output[:] = tensor_list

    return output


class JetReconstructionTraining(JetReconstructionNetwork):
    def __init__(self, options: Options, torch_script: bool = False):
        super(JetReconstructionTraining, self).__init__(options, torch_script)

        self.log_clip = torch.log(10 * torch.scalar_tensor(torch.finfo(torch.float32).eps)).item()

        self.event_particle_names = list(self.training_dataset.event_info.product_particles.keys())
        self.product_particle_names = {
            particle: self.training_dataset.event_info.product_particles[particle][0]
            for particle in self.event_particle_names
        }
        self.aggregate_metrics = {} # store metrics like 'classif_i_acc_num' 

    def particle_symmetric_loss(self, assignment: Tensor, detection: Tensor, target: Tensor, mask: Tensor, weight: Tensor) -> Tensor:
        assignment_loss = assignment_cross_entropy_loss(assignment, target, mask, weight, self.options.focal_gamma)
        detection_loss = F.binary_cross_entropy_with_logits(detection, mask.float(), weight=weight, reduction='none')

        return torch.stack((
            self.options.assignment_loss_scale * assignment_loss,
            self.options.detection_loss_scale * detection_loss
        ))

    def compute_symmetric_losses(self, assignments: List[Tensor], detections: List[Tensor], targets):
        symmetric_losses = []

        # TODO think of a way to avoid this memory transfer but keep permutation indices synced with checkpoint
        # Compute a separate loss term for every possible target permutation.
        for permutation in self.event_permutation_tensor.cpu().numpy():

            # Find the assignment loss for each particle in this permutation.
            current_permutation_loss = tuple(
                self.particle_symmetric_loss(assignment, detection, target, mask, weight)
                for assignment, detection, (target, mask, weight)
                in zip(assignments, detections, targets[permutation])
            )

            # The loss for a single permutation is the sum of particle losses.
            symmetric_losses.append(torch.stack(current_permutation_loss))

        # Shape: (NUM_PERMUTATIONS, NUM_PARTICLES, 2, BATCH_SIZE)
        return torch.stack(symmetric_losses)

    def combine_symmetric_losses(self, symmetric_losses: Tensor) -> Tuple[Tensor, Tensor]:
        # Default option is to find the minimum loss term of the symmetric options.
        # We also store which permutation we used to achieve that minimal loss.
        # combined_loss, _ = symmetric_losses.min(0)
        total_symmetric_loss = symmetric_losses.sum((1, 2))
        index = total_symmetric_loss.argmin(0)

        combined_loss = torch.gather(symmetric_losses, 0, index.expand_as(symmetric_losses))[0]

        # Simple average of all losses as a baseline.
        if self.options.combine_pair_loss.lower() == "mean":
            combined_loss = symmetric_losses.mean(0)

        # Soft minimum function to smoothly fuse all loss function weighted by their size.
        if self.options.combine_pair_loss.lower() == "softmin":
            weights = F.softmin(total_symmetric_loss, 0)
            weights = weights.unsqueeze(1).unsqueeze(1)
            combined_loss = (weights * symmetric_losses).sum(0)

        return combined_loss, index

    def symmetric_losses(
        self,
        assignments: List[Tensor],
        detections: List[Tensor],
        targets: Tuple[Tuple[Tensor, Tensor, Tensor], ...]
    ) -> Tuple[Tensor, Tensor]:
        # We are only going to look at a single prediction points on the distribution for more stable loss calculation
        # We multiply the softmax values by the size of the permutation group to make every target the same
        # regardless of the number of sub-jets in each target particle
        assignments = [prediction + torch.log(torch.scalar_tensor(decoder.num_targets))
                       for prediction, decoder in zip(assignments, self.branch_decoders)]

        # Convert the targets into a numpy array of tensors so we can use fancy indexing from numpy
        targets = numpy_tensor_array(targets)

        # Compute the loss on every valid permutation of the targets
        symmetric_losses = self.compute_symmetric_losses(assignments, detections, targets)

        # Squash the permutation losses into a single value.
        return self.combine_symmetric_losses(symmetric_losses)

    def symmetric_divergence_loss(self, predictions: List[Tensor], masks: Tensor) -> Tensor:
        divergence_loss = []

        for i, j in self.event_info.event_transpositions:
            # Symmetric divergence between these two distributions
            div = jensen_shannon_divergence(predictions[i], predictions[j])

            # ERF term for loss
            loss = torch.exp(-(div ** 2))
            loss = loss.masked_fill(~masks[i], 0.0)
            loss = loss.masked_fill(~masks[j], 0.0)

            divergence_loss.append(loss)

        return torch.stack(divergence_loss).mean(0)
        # return -1 * torch.stack(divergence_loss).sum(0) / len(self.training_dataset.unordered_event_transpositions)

    def add_kl_loss(
            self,
            total_loss: List[Tensor],
            assignments: List[Tensor],
            masks: Tensor,
            weights: Tensor
    ) -> List[Tensor]:
        if len(self.event_info.event_transpositions) == 0:
            return total_loss

        # Compute the symmetric loss between all valid pairs of distributions.
        kl_loss = self.symmetric_divergence_loss(assignments, masks)
        kl_loss = (weights * kl_loss).sum() / masks.sum()

        with torch.no_grad():
            self.log("loss/symmetric_loss", kl_loss, sync_dist=True)
            if torch.isnan(kl_loss):
                raise ValueError("Symmetric KL Loss has diverged.")
            
        # print(f"kl_loss:")
        # print(f" - shape: {kl_loss.shape}")
        # print(f" - value: {kl_loss}")

        return total_loss + [self.options.kl_loss_scale * kl_loss]

    def add_regression_loss(
            self,
            total_loss: List[Tensor],
            predictions: Dict[str, Tensor],
            targets:  Dict[str, Tensor]
    ) -> List[Tensor]:
        regression_terms = []
        # TODO: add custom_weights for regression_loss
        # print(f"regression_loss:")

        for key in targets:
            current_target_type = self.training_dataset.regression_types[key]
            current_prediction = predictions[key]
            current_target = targets[key]

            current_mean = self.regression_decoder.networks[key].mean
            current_std = self.regression_decoder.networks[key].std

            current_mask = ~torch.isnan(current_target)

            current_loss = regression_loss(current_target_type)(
                current_prediction[current_mask],
                current_target[current_mask],
                current_mean,
                current_std
            )
            current_loss = torch.mean(current_loss)

            with torch.no_grad():
                self.log(f"loss/regression/{key}", current_loss, sync_dist=True)
            
            # print(f" - {key} shape: {current_loss.shape}")
            # print(f" - {key} value: {current_loss}")

            regression_terms.append(self.options.regression_loss_scale * current_loss)

        return total_loss + regression_terms

    def add_classification_loss(
            self,
            total_loss: List[Tensor],
            predictions: Dict[str, Tensor],
            targets: Dict[str, Tensor],
            use_custom_weights: Tensor = None
    ) -> List[Tensor]:
        classification_terms = []

        # print(f"classification_loss:")
        for key in targets:
            current_prediction = predictions[key] # shape (N_batch, N_classes)
            current_target = targets[key].long() # shape (N_batch,)

            # print(f" - {key} prediction shape: {current_prediction.shape}")
            # print(f" - {key} targer shape: {current_target.shape}")
            
            # have to specify long, otherwise error
            #  - RuntimeError: "nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Int'

            # print(current_prediction)
            # print(current_target)
            
            weight = None if not self.balance_classifications else self.classification_weights[key]
            # if weight is None:
            #     weight = torch.ones_like(use_custom_weights)
            # weight *= use_custom_weights
            # print(use_custom_weights)
            if use_custom_weights is not None:
                # overrides weight with custom weights
                weight = use_custom_weights
                current_loss = F.cross_entropy(
                    current_prediction,
                    current_target,
                    ignore_index=-1,
                    reduction='none'
                )
                current_loss = current_loss * weight
                # now apply reduction --> default is mean
                # https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
                current_loss = current_loss.mean()
            else:
                # default, original case
                current_loss = F.cross_entropy(
                    current_prediction,
                    current_target,
                    ignore_index=-1,
                    weight=weight
                )

            classification_terms.append(self.options.classification_loss_scale * current_loss)

            with torch.no_grad():
                self.log(f"loss/classification/{key}_train", current_loss, sync_dist=True)

                # Calculate accuracy on training split
                # should really take weight into account...
                classification_accuracy = (current_prediction.argmax(1) == current_target).float().mean()
                self.log(
                    f"CLASSIFICATION/{key}_accuracy_train", 
                    classification_accuracy,
                    sync_dist=True)
                
                # enable calculation of uncertainties
                # 1. simple binomial error --> assumes N.p ≥ 5 and N.(1-p) ≥ 5
                correct = (current_prediction.argmax(1) == current_target).float()
                acc_error = torch.sqrt(classification_accuracy * (1 - classification_accuracy) / correct.shape[0])
                self.log(f"CLASSIFICATION/{key}_acc_error_train", acc_error.mean(), sync_dist=True)
                # 2. store number of correct and total for epoch end calculation
                #    this will allow us to calculate the uncertainty without assuming anything
                accuracy_num, accuracy_den = correct.sum(), correct.shape[0]
                if f"CLASSIFICATION/{key}_acc_num_train" not in self.aggregate_metrics:
                    self.aggregate_metrics[f"CLASSIFICATION/{key}_acc_num_train"] = accuracy_num
                    self.aggregate_metrics[f"CLASSIFICATION/{key}_acc_den_train"] = accuracy_den
                else:
                    self.aggregate_metrics[f"CLASSIFICATION/{key}_acc_num_train"] += accuracy_num
                    self.aggregate_metrics[f"CLASSIFICATION/{key}_acc_den_train"] += accuracy_den
                    
                # todo: add other metrics?

                # classification_metrics = {
                #     "sensitivity":sk_metrics.recall_score,
                #     "specificity":lambda t, p: sk_metrics.recall_score(~t, ~p),
                #     "f1_score":sk_metrics.f1_score
                # }
                # for cm in classification_metrics:
                #     self.log(
                #         f"CLASSIFICATION/{key}_{cm}_train",
                #         classification_metrics[cm](current_prediction, current_target),
                #         sync_dist=True
                #     )

                
            
            # print(f" - {key} shape: {current_loss.shape}")
            # print(f" - {key} value: {current_loss}")

        return total_loss + classification_terms

    def training_step(self, batch: Batch, batch_nb: int) -> Dict[str, Tensor]:
        # ===================================================================================================
        # Network Forward Pass
        # ---------------------------------------------------------------------------------------------------
        outputs = self.forward(batch.sources)

        # ===================================================================================================
        # Initial log-likelihood loss for classification task
        # ---------------------------------------------------------------------------------------------------
        symmetric_losses, best_indices = self.symmetric_losses(
            outputs.assignments,
            outputs.detections,
            batch.assignment_targets,
        )

        # Construct the newly permuted masks based on the minimal permutation found during NLL loss.
        permutations = self.event_permutation_tensor[best_indices].T
        masks = torch.stack([target.mask for target in batch.assignment_targets])
        masks = torch.gather(masks, 0, permutations)

        # ===================================================================================================
        # Balance the loss based on the distribution of various classes in the dataset.
        # ---------------------------------------------------------------------------------------------------

        # Default unity weight on correct device.
        weights = torch.ones_like(symmetric_losses)

        # Balance based on the particles present - only used in partial event training
        if self.balance_particles:
            class_indices = (masks * self.particle_index_tensor.unsqueeze(1)).sum(0)
            weights *= self.particle_weights_tensor[class_indices]

        # Balance based on the number of jets in this event
        if self.balance_jets:
            weights *= self.jet_weights_tensor[batch.num_vectors]
        
        # print(f"weights:")
        # print(f" - shape: {weights.shape}")

        # # Balance using custom weights
        # # weights *= self.custom_weights_tensor[batch.item] # shape [B,]

        # # print(f"custom_weights tensor:")
        # # print(f" - shape: {self.custom_weights_tensor[batch.item].shape}")

        # # Take the weighted average of the symmetric loss terms.
        # print(f"masks:")
        # print(f" - shape (before unsqueeze): {masks.shape}")
        masks = masks.unsqueeze(1)
        # print(f" - shape (after unsqueeze): {masks.shape}")
        symmetric_losses = (weights * symmetric_losses).sum(-1) / torch.clamp(masks.sum(-1), 1, None)
        assignment_loss, detection_loss = torch.unbind(symmetric_losses, 1)

        # print(f"assignent loss tensor:")
        # print(f" - shape: {assignment_loss.shape}")
        # print(f"detection loss tensor:")
        # print(f" - shape: {detection_loss.shape}")

        # ===================================================================================================
        # Some basic logging
        # ---------------------------------------------------------------------------------------------------
        with torch.no_grad():
            for name, l in zip(self.training_dataset.assignments, assignment_loss):
                self.log(f"loss/{name}/assignment_loss", l, sync_dist=True)

            for name, l in zip(self.training_dataset.assignments, detection_loss):
                self.log(f"loss/{name}/detection_loss", l, sync_dist=True)

            if torch.isnan(assignment_loss).any():
                raise ValueError("Assignment loss has diverged!")

            if torch.isinf(assignment_loss).any():
                raise ValueError("Assignment targets contain a collision.")

        # ===================================================================================================
        # Start constructing the list of all computed loss terms.
        # ---------------------------------------------------------------------------------------------------
        total_loss = []

        if self.options.assignment_loss_scale > 0:
            total_loss.append(assignment_loss)

        if self.options.detection_loss_scale > 0:
            total_loss.append(detection_loss)

        # ===================================================================================================
        # Auxiliary loss terms which are added to reconstruction loss for alternative targets.
        # ---------------------------------------------------------------------------------------------------
        if self.options.kl_loss_scale > 0:
            total_loss = self.add_kl_loss(total_loss, outputs.assignments, masks, weights)

        if self.options.regression_loss_scale > 0:
            total_loss = self.add_regression_loss(total_loss, outputs.regressions, batch.regression_targets)

        if self.options.classification_loss_scale > 0:
            use_custom_weights = None
            if self.custom_weights_tensor is not None:
                use_custom_weights = self.custom_weights_tensor[batch.item]
            total_loss = self.add_classification_loss(total_loss, outputs.classifications, batch.classification_targets, 
                use_custom_weights=use_custom_weights)
            # total_loss = self.add_classification_loss(total_loss, outputs.classifications, batch.classification_targets) 
        
        # print(f"total loss:")
        # print(f" - len (before combining): {len(total_loss)}")

        # ===================================================================================================
        # Combine and return the loss
        # ---------------------------------------------------------------------------------------------------
        total_loss = torch.cat([loss.view(-1) for loss in total_loss])

        # print(f" - shape (after combining): {total_loss.shape}")
        # print(f" - values (after combining): {total_loss}")

        self.log("loss/total_loss", total_loss.sum(), sync_dist=True)

        return total_loss.mean()
    
    def log_gradients(self, norm_type=2):
        '''
        Log the gradient norms of the model parameters.
        by default the norm_type is 2...
        '''
        total_norm = 0.0
        # layer_norms = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                # print(f"{name}: {param.grad}")
                param_norm = param.grad.data.norm(norm_type)
                # layer_norms[name] = param_norm.item()
                total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        self.log("gradients/total_norm", total_norm, sync_dist=True)
        # too many bloody parameters --> too expensive to keep
        # for name, norm in layer_norms.items():
        #     self.log(f"gradients/{name}_norm", norm, sync_dist=True)

    def on_after_backward(self) -> None:
        '''
        PyTorch Lightning hook to be called after loss.backward() and before optimizer.step()
        - want to log the grad norm here
        '''
        self.log_gradients()
        return None

    def on_train_epoch_end(self):
        ''' 
            Called at end of epoch --> we want to aggregate some metrics here for
            more robust calculations
        '''
        # Allow us to calculate aggregate metrics afterwards
        for key in self.aggregate_metrics:
            value = self.aggregate_metrics[key]

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