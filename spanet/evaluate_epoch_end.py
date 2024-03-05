## File: evaluate_epoch_end.py
## Aim: file containing all the functions used to calculate accuracy and loss for the training and validation splits

## Import modules

import torch
import h5py
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from spanet import JetReconstructionModel, Options
from spanet.dataset.types import Batch
import time 

# from collections import defaultdict

def import_model(log_directory, cuda, checkpoint_epoch, test_file, batch_size, data_mode, accuracy_loss):
    

    options = Options.load(f"{log_directory}/options.json")
    options.testing_file = test_file
    # options.validation_file = test_file
    options.batch_size = batch_size
    checkpoint_path = f"{log_directory}/checkpoints/{checkpoint_epoch}"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint = checkpoint["state_dict"]

    print(f"Model training: {options.training_file}")
    print(f"Model validation: {options.validation_file}")
    print(f"Model testing: {options.testing_file}")
    print(f"Model batch size: {options.batch_size}")
    print(f"Checkpoint: {checkpoint_path}")

    if data_mode == 0:
        print(f"Inferring {accuracy_loss} metrics on {options.training_file}")
    elif data_mode == 1:
        print(f"Inferring {accuracy_loss} metrics on {options.validation_file}")
    elif data_mode == 2:
        print(f"Inferring {accuracy_loss} metrics on {options.testing_file}")

    model = JetReconstructionModel(options)

    model.load_state_dict(checkpoint)

    # for evaluation purposes
    model = model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # if we want to use a gpu
    if cuda:
        model = model.cuda()

    return model

def evaluate_on_split_acc(model: JetReconstructionModel, data_mode):
    '''
    Hardcoded version 
    '''
    accuracies = {}
    acc_jet = {}
    acc_particle = {}
    i_of_j = []
    hadronic_tops = 4
    for j in range(hadronic_tops+1):
        for i in range(1, j+1):
            i_of_j_name = f'{i}_of_{j}'
            acc_jet[i_of_j_name] = { 'accuracy':[], 'len':[], 'sum':[] }
            acc_particle[i_of_j_name] = { 'accuracy':[], 'len':[], 'sum':[] }
            i_of_j.append(i_of_j_name)
    acc_particle['sk_metrics'] = {'accuracy':[], 'f_score':[], 'sensitivity':[], 'specificity':[]}
    
    accuracies['jet'] = acc_jet
    accuracies['particle'] = acc_particle
    
    if data_mode == 0:
        split_dataloader = model.train_dataloader()
    elif data_mode == 1:
        split_dataloader = model.val_dataloader()
    elif data_mode == 2:
        split_dataloader = model.test_dataloader()
  
    for batch in tqdm(split_dataloader, desc="Split acc"):
        metrics = accuracy_calculator(model, batch)
        # somewhat hardcoded
        for accuracy_type in accuracies:
            for x in i_of_j:
                accuracies[f'{accuracy_type}'][x]['accuracy'].append(metrics[f'{accuracy_type}/accuracy_{x}'])
                accuracies[f'{accuracy_type}'][x]['len'].append(metrics[f'{accuracy_type}/accuracy_{x}_len'])
                accuracies[f'{accuracy_type}'][x]['sum'].append(metrics[f'{accuracy_type}/accuracy_{x}_sum'])
        for sk_metric in accuracies['particle']['sk_metrics']:
            accuracies['particle']['sk_metrics'][sk_metric].append(metrics[f'particle/{sk_metric}'])

    return accuracies

def evaluate_on_split_loss(model: JetReconstructionModel, data_mode):

    losses = {}
    losses['total'] = []
    
    hadronic_tops = 4
    for i in range(hadronic_tops):
        losses[f't{i+1}'] = {'assignment':[],'detection':[]}

    if data_mode == 0:
        split_dataloader = model.train_dataloader()
    elif data_mode == 1:
        split_dataloader = model.val_dataloader()
    elif data_mode == 2:
        split_dataloader = model.test_dataloader()

    for batch in tqdm(split_dataloader, desc="Split acc and loss"):

        assignment_loss, detection_loss, total_loss = loss_calculator(model, batch)
        
        for i in range(hadronic_tops):
            losses[f't{i+1}']['assignment'].append(assignment_loss[i].numpy())
            losses[f't{i+1}']['detection'].append(detection_loss[i].numpy())
            
        losses['total'].append(total_loss.numpy())
    
    return losses

def accuracy_calculator(model: JetReconstructionModel, batch: Batch):
        '''
        Accuracy calculator (simplified) for h2t, checking out the logging and time it would take...
        '''
        sources, num_jets, targets, regression_targets, classification_targets = batch
        sources = [[x[0].to(model.device), x[1].to(model.device)] for x in sources]
        jet_predictions, particle_scores, regressions, classifications = model.predict(sources)

        batch_size = num_jets.shape[0]
        num_targets = len(targets)

        stacked_targets = np.zeros(num_targets, dtype=object)
        stacked_masks = np.zeros((num_targets, batch_size), dtype=np.bool_)
        for i, (target, mask) in enumerate(targets):
            stacked_targets[i] = target.detach().cpu().numpy()
            stacked_masks[i] = mask.detach().cpu().numpy()

        metrics = model.evaluator.full_report_string(jet_predictions, stacked_targets, stacked_masks, prefix="Purity/")

        for target, prediction, decoder in zip(stacked_targets, jet_predictions, model.branch_decoders):
            for indices in decoder.permutation_indices:
                if len(indices) > 1:
                    prediction[:, indices] = np.sort(prediction[:, indices])
                    target[:, indices] = np.sort(target[:, indices])

        metrics.update(update_metrics(model, jet_predictions, particle_scores, stacked_targets, stacked_masks))

        return metrics

def update_metrics(model: JetReconstructionModel, jet_predictions, particle_scores, stacked_targets, stacked_masks):
    '''
    Analog to mss_compute_metrics (simplified version for h2t)
    '''
    event_permutation_group = model.event_permutation_tensor.cpu().numpy() # [[0,1], [1,0]]
    num_permutations = len(event_permutation_group) # 2
    num_targets, batch_size = stacked_masks.shape # 2, 32
    particle_predictions = particle_scores >= 0.5 # [ [T,F,T... (len batch size)] , [F,F,T,...(len batch size)] ]

    jet_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=np.bool_) 
    particle_accuracies = np.zeros((num_permutations, num_targets, batch_size), dtype=np.bool_)
    for i, permutation in enumerate(event_permutation_group):
        for j, (prediction, target) in enumerate(zip(jet_predictions, stacked_targets[permutation])):
            jet_accuracies[i, j] = np.all(prediction == target, axis=1)
        particle_accuracies[i] = stacked_masks[permutation] == particle_predictions

    jet_accuracies = jet_accuracies.sum(1) # [[1,1,1... (len batch size)],[1,0,2... (len batch size)]]
    particle_accuracies = particle_accuracies.sum(1) # [[1,1,2... (len batch size)],[1,0,1... (len batch size)]]

    chosen_permutations = model.event_permutation_tensor[jet_accuracies.argmax(0)].T
    chosen_permutations = chosen_permutations.cpu() # moves it accessible to cpu memory
    permuted_masks = torch.gather(torch.from_numpy(stacked_masks), 0, chosen_permutations).numpy()

    num_particles = stacked_masks.sum(0)
    jet_accuracies = jet_accuracies.max(0) 
    particle_accuracies = particle_accuracies.max(0)

    metrics = {f"jet/accuracy_{i}_of_{j}": (jet_accuracies[num_particles == j] >= i).mean()
                for j in range(1, num_targets + 1)
                for i in range(1, j + 1)}
    
    # not ignoring particle accuracy
    metrics.update({f"particle/accuracy_{i}_of_{j}": (particle_accuracies[num_particles == j] >= i).mean()
                for j in range(1, num_targets + 1)
                for i in range(1, j + 1)})
    particle_scores = particle_scores.ravel()
    particle_targets = permuted_masks.ravel()
    particle_predictions = particle_predictions.ravel()
    for name, metric in model.particle_metrics.items():
        metrics[f"particle/{name}"] = metric(particle_targets, particle_predictions)

    # empty function
    # for name, metric in model.particle_score_metrics.items():
    #     metrics[f"particle/{name}"] = metric(particle_targets, particle_scores)
    
    for j in range(1, num_targets+1):
        for i in range(1, j+1):
            metrics[f"jet/accuracy_{i}_of_{j}_sum"] = (jet_accuracies[num_particles == j] >= i).sum()
            metrics[f"jet/accuracy_{i}_of_{j}_len"] = len((jet_accuracies[num_particles == j] >= i))
            metrics[f"particle/accuracy_{i}_of_{j}_sum"] = (particle_accuracies[num_particles == j] >= i).sum()
            metrics[f"particle/accuracy_{i}_of_{j}_len"] = len((particle_accuracies[num_particles == j] >= i))

    metrics[f"jet/num_particles_0"] = (num_particles == 0).sum()
    metrics[f"jet/num_particles_all"] = len(num_particles)

    metrics["validation_accuracy"] = metrics[f"jet/accuracy_{num_targets}_of_{num_targets}"]

    return metrics

def loss_calculator(model: JetReconstructionModel, batch: Batch):
        '''
        Simplified version of the loss calculations so we can use them both on the training and validation datasets.
        Batch comes from the dataset used, and dataset_name has to be manually put in for the logging
        '''
        # sources = [[x[0].to(self.device), x[1].to(self.device)] for x in batch.sources] # don't know how to implement on GPU
        outputs = model.forward(batch.sources)
        
        symmetric_losses, best_indices = model.symmetric_losses(
            outputs.assignments,
            outputs.detections,
            batch.assignment_targets
            )

        permutations = model.event_permutation_tensor[best_indices].T
        masks = torch.stack([target.mask for target in batch.assignment_targets])
        masks = torch.gather(masks, 0, permutations)

        weights = torch.ones_like(symmetric_losses)

        if model.balance_particles:
            class_indices = (masks * model.particle_index_tensor.unsqueeze(1)).sum(0) # (0,1,1,2,2,2,1,3,3,2,1,0,2... (len batch size))
            weights *= model.particle_weights_tensor[class_indices]
        if model.balance_jets:
            weights *= model.jet_weights_tensor[batch.num_vectors]
            
        masks = masks.unsqueeze(1)

        symmetric_losses = (weights * symmetric_losses).sum(-1) / masks.sum(-1)
        assignment_loss, detection_loss = torch.unbind(symmetric_losses, 1)
            
        if torch.isnan(assignment_loss).any():
            raise ValueError("Assignment loss has diverged!")

        total_loss = []

        if model.options.assignment_loss_scale > 0:
            total_loss.append(assignment_loss)

        if model.options.detection_loss_scale > 0:
            total_loss.append(detection_loss)
            
        total_loss = torch.cat([loss.view(-1) for loss in total_loss])

        return assignment_loss, detection_loss, total_loss.sum()


def make_h5_file_acc(metrics_dict, filename):

    with h5py.File(filename, 'w') as h5_file:
        
        # elif accuracy_loss == 'loss':
        #     losses = h5_file.create_group('losses')
        acc_or_loss = 'accuracies'
        metrics_group = h5_file.create_group(acc_or_loss)

        # hardcoded for accuracies !!!!
        for metrics_type in metrics_dict: # jet, particle

            h5_file.create_group(f"{acc_or_loss}/{metrics_type}")

            for metrics_subtype in metrics_dict[metrics_type]: # i_of_j, (sk_metrics)

                h5_file.create_group(f"{acc_or_loss}/{metrics_type}/{metrics_subtype}")

                for metrics_subsubtype in metrics_dict[metrics_type][metrics_subtype]: # accuracy, len, sum, (accuracy, f_score, sensitivity, specificity)

                    h5_file[f"{acc_or_loss}/{metrics_type}/{metrics_subtype}"].create_dataset(f"{metrics_subsubtype}", data=np.array(metrics_dict[metrics_type][metrics_subtype][metrics_subsubtype]))

        
def make_h5_file_loss(metrics_dict, filename):

    with h5py.File(filename, 'w') as h5_file:
            
        acc_or_loss = 'losses'
        metrics_group = h5_file.create_group(acc_or_loss)

        # hardcoded for accuracies !!!!
        for metrics_type in metrics_dict: # t1, t2, total

            if metrics_type == 'total':
                h5_file[f"{acc_or_loss}"].create_dataset(f"{metrics_type}", data=np.array(metrics_dict[metrics_type]))
                continue

            h5_file.create_group(f"{acc_or_loss}/{metrics_type}")

            for metrics_subtype in metrics_dict[metrics_type]: # assignment, detection

                h5_file[f"{acc_or_loss}/{metrics_type}"].create_dataset(f"{metrics_subtype}", data=np.array(metrics_dict[metrics_type][metrics_subtype]))

    
# --------------------


def main(
    log_directory, 
    checkpoint_epoch, 
    output_directory, 
    test_file,
    batch_size,
    data_mode,
    accuracy_loss,
    cuda=False):

    print("-"*20)
    if data_mode == 0:
        split_name = 'train'
    elif data_mode == 1:
        split_name = 'val'
    elif data_mode == 2:
        split_name = 'test'
    if accuracy_loss == 'loss':
        cuda=False
    start = time.time()

    # e.g., log_directory = "/Users/rw22031/PhD-COMPUTING/ttX3-dev/SPA-Net/SPANet/spanet_output/version_7"
    checkpoint_epoch = checkpoint_epoch.split('/')[-1] # in case a directory was given instead (easier for batch jobs)
    if output_directory == 'log_all':
        output_file = f"{log_directory}/checkpoints/{output_directory}/{checkpoint_epoch}_{split_name}_infer_{accuracy_loss}.h5"
    else:
        output_file = f"{output_directory}/{checkpoint_epoch}_{split_name}_infer_{accuracy_loss}.h5"

    print(f"Importing model...")
    model = import_model(log_directory, cuda, checkpoint_epoch, test_file, batch_size, data_mode, accuracy_loss)
 
    print(f"Inferring on checkpoint {checkpoint_epoch}...")
    if accuracy_loss == 'acc':
        accuracies = evaluate_on_split_acc(model, data_mode)
        print(f"Making output file: {output_file}...")
        make_h5_file_acc(accuracies, output_file)
    
    elif accuracy_loss == 'loss':
        losses = evaluate_on_split_loss(model, data_mode)
        print(f"Making output file: {output_file}...")
        make_h5_file_loss(losses, output_file)

    end = time.time()
    duration = end - start


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-l", "--log_directory", type=str,
                        help="Pytorch Lightning Log directory containing the checkpoint and options file.")
    parser.add_argument("-e", "--checkpoint_epoch", type=str,
                        help="Name of epoch file in the checkpoints directory.")
    parser.add_argument("-o", "--output_directory", type=str,
                        help="Name of output directory to save the output to.")
    parser.add_argument("-t", "--test_file", type=str,
                        help="Name with path of the test file we want to also evaluate on.")
    parser.add_argument("-b", "--batch_size", type=int,
                        help="Override batch size in hyperparameters.")
    parser.add_argument("-dm", "--data_mode", type=int, choices=[0,1,2],
                       help="Choose which data split to use (0 = train, 1 = validation, 2 = testing).")
    parser.add_argument("-al", "--accuracy_loss", type=str, choices= ['acc', 'loss'],
                       help="Choose which data split to use (0 = train, 1 = validation, 2 = testing).")
    parser.add_argument("-c", "--cuda", action="store_true",
                       help="Choose to use GPU or not in the inference.")
    arguments = parser.parse_args()                    
    main(**arguments.__dict__)
