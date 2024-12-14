from argparse import ArgumentParser
from typing import Optional
from numpy import ndarray as Array
import os

import h5py

from spanet.dataset.jet_reconstruction_dataset import JetReconstructionDataset
from spanet.dataset.types import Evaluation, SpecialKey, Outputs
from spanet.evaluation import evaluate_on_test_dataset, load_model


def create_hdf5_output(
    output_file: str,
    dataset: JetReconstructionDataset,
    evaluation: Evaluation,
    full_outputs: Optional[Outputs]
):
    print(f"Creating output file at: {output_file}")
    with h5py.File(output_file, 'w') as output:
        # Copy over the source features from the input file.
        with h5py.File(dataset.data_file, 'r') as input_dataset:
            for input_name in input_dataset[SpecialKey.Inputs]:
                for feature_name in input_dataset[SpecialKey.Inputs][input_name]:
                    output.create_dataset(
                        f"{SpecialKey.Inputs}/{input_name}/{feature_name}",
                        data=input_dataset[SpecialKey.Inputs][input_name][feature_name]
                    )

        # Construct the assignment structure. Output both the top assignment and associated probabilities.
        for event_particle in dataset.event_info.event_particles:
            for i, product_particle in enumerate(dataset.event_info.product_particles[event_particle]):
                output.create_dataset(
                    f"{SpecialKey.Targets}/{event_particle}/{product_particle}",
                    data=evaluation.assignments[event_particle][:, i]
                )

            output.create_dataset(
                f"{SpecialKey.Targets}/{event_particle}/assignment_probability",
                data=evaluation.assignment_probabilities[event_particle]
            )

            output.create_dataset(
                f"{SpecialKey.Targets}/{event_particle}/detection_probability",
                data=evaluation.detection_probabilities[event_particle]
            )

            output.create_dataset(
                f"{SpecialKey.Targets}/{event_particle}/marginal_probability",
                data=(
                    evaluation.detection_probabilities[event_particle] *
                    evaluation.assignment_probabilities[event_particle]
                )
            )

        # Simply copy over the structure of the regressions and classifications.
        for name, regression in evaluation.regressions.items():
            output.create_dataset(f"{SpecialKey.Regressions}/{name}", data=regression)

        for name, classification in evaluation.classifications.items():
            output.create_dataset(f"{SpecialKey.Classifications}/{name}", data=classification)

        if full_outputs is not None:
            for name, vector in full_outputs.vectors.items():
                output.create_dataset(f"{SpecialKey.Embeddings}/{name}", data=vector)


def main(log_directory: str,
         output_file: Optional[str],
         test_file: Optional[str],
         event_file: Optional[str],
         batch_size: Optional[int],
         output_vectors: bool,
         gpu: bool,
         fp16: bool,
         checkpoint: Optional[str],
         output_directory: Optional[str]):
    
    # load model at particular checkpoint
    if checkpoint is not None:
        checkpoint = os.path.basename(checkpoint)
        model = load_model(log_directory, test_file, event_file, batch_size, gpu, fp16=fp16, checkpoint=checkpoint)
    else:
        model = load_model(log_directory, test_file, event_file, batch_size, gpu, fp16=fp16)

    if output_vectors:
        evaluation, full_outputs = evaluate_on_test_dataset(model, return_full_output=True, fp16=fp16)
    else:
        evaluation = evaluate_on_test_dataset(model, return_full_output=False, fp16=fp16)
        full_outputs = None

    # always save in 'version_x'
    output_directory = os.path.join(output_directory, os.path.basename(log_directory), 'predict')
    os.makedirs(output_directory, exist_ok=True)
    if output_file is None:
        output_file = os.path.basename(model.options.testing_file).replace(".h5", "")
        ckpt = checkpoint if checkpoint is not None else "default"
        output_file = os.path.join( output_directory, f"{output_file}_predictions_{ckpt}.h5" )
    else:
        output_file = os.path.basename(output_file) # ignore path we give --> we want o_dir/version_x/predictions/output.h5
        output_file = os.path.join( output_directory, output_file)
        
    create_hdf5_output(output_file, model.testing_dataset, evaluation, full_outputs)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("log_directory", type=str,
                        help="Pytorch Lightning Log directory containing the checkpoint and options file.")

    parser.add_argument("-o", "--output_file", type=str, default=None,
                        help="The output HDF5 to create with the new predicted jets for each event.")

    parser.add_argument("-tf", "--test_file", type=str, default=None,
                        help="Replace the test file in the options with a custom one. "
                             "Must provide if options does not define a test file.")

    parser.add_argument("-ef", "--event_file", type=str, default=None,
                        help="Replace the event file in the options with a custom event.")

    parser.add_argument("-bs", "--batch_size", type=int, default=None,
                        help="Replace the batch size in the options with a custom size.")

    parser.add_argument("-g", "--gpu", action="store_true",
                        help="Evaluate network on the gpu.")
    
    parser.add_argument("-fp16", "--fp16", action="store_true",
                        help="Use Automatic Mixed Precision for inference.")

    parser.add_argument("-v", "--output_vectors", action="store_true",
                        help="Include embedding vectors in output in an additional section of the HDF5.")
    
    parser.add_argument("-cp", "--checkpoint", type=str, default=None,
                        help="Checkpointed epoch we want to use for inference.")

    parser.add_argument("-od", "--output_directory", type=str, 
                        default="/nfs/dust/cms/user/sanjrani/SPANet_Investigations/investigation2/pepper_analysis/output/h4t_systematics/spanet/output",
                        help="Where to save output to (creates 'version_x' directory inside it)")

    arguments = parser.parse_args()
    main(**arguments.__dict__)
