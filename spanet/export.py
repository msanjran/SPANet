from argparse import ArgumentParser
from typing import List

import torch
from torch.nn import functional as F
from torch.utils._pytree import tree_map

import pytorch_lightning as pl

from spanet import JetReconstructionModel
from spanet.dataset.types import Source
from spanet.evaluation import load_model

# debug
import warnings
from IPython import embed

class WrappedModel(pl.LightningModule):
    def __init__(
            self,
            model: JetReconstructionModel,
            input_log_transform: bool = False,
            output_log_transform: bool = False,
            output_embeddings: bool = False
    ):
        super(WrappedModel, self).__init__()

        self.model = model
        self.input_log_transform = input_log_transform
        self.output_log_transform = output_log_transform
        self.output_embeddings = output_embeddings

    def apply_input_log_transform(self, sources):
        new_sources = []
        for (data, mask), name in zip(sources, self.model.event_info.input_names):
            new_data = torch.stack([
                mask * torch.log(data[:, :, i] + 1) if log_transformer else data[:, :, i]
                for i, log_transformer in enumerate(self.model.event_info.log_features(name))
            ], -1)

            new_sources.append(Source(new_data, mask))
        return new_sources

    def forward(self, sources: List[Source]):
        if self.input_log_transform:
            print(f"7.1. input log transform: ...")
            sources = self.apply_input_log_transform(sources)
            print(f"7.1. input log transform: done")

        print(f"7.2. model(sources): ...")
        outputs = self.model(sources)
        print(f"7.2. model(sources): done")
        
        embed()

        if self.output_log_transform:
            
            print(f"7.3. assignments: ...")
            assignments = [assignment for assignment in outputs.assignments]
            print(f"Len. assignments: {len(assignments)}")
            print(f"Assignments[0]: {assignments[0]}")
            
            print(f"7.4. detections: ...")
            detections = [F.logsigmoid(detection) for detection in outputs.detections]

            print(f"7.5. classifications: ...")
            classifications = [
                F.log_softmax(outputs.classifications[key], dim=-1)
                for key in self.model.training_dataset.classifications.keys()
            ]

        else:
            print(f"7.6. assignments: ...")
            assignments = [assignment.exp() for assignment in outputs.assignments]
            
            print(f"7.7. detections: ...")
            detections = [torch.sigmoid(detection) for detection in outputs.detections]
            
            print(f"7.8. classifications: ...")
            classifications = [
                F.softmax(outputs.classifications[key], dim=-1)
                for key in self.model.training_dataset.classifications.keys()
            ]
            
        print(f"7.9. regressions: ...")
        regressions = [
            outputs.regressions[key]
            for key in self.model.training_dataset.regressions.keys()
        ]
        
        print(f"7.10. embedding_vectors: ...")
        embedding_vectors = list(outputs.vectors.values()) if self.output_embeddings else []

        return *assignments, *detections, *regressions, *classifications, *embedding_vectors


def onnx_specification(model, output_log_transform: bool = False, output_embeddings: bool = False):
    input_names = []
    output_names = []

    dynamic_axes = {}

    for input_name in model.event_info.input_names:
        for input_type in ["data", "mask"]:
            current_input = f"{input_name}_{input_type}"
            input_names.append(current_input)
            dynamic_axes[current_input] = {
                0: 'batch_size',
                1: f'num_{input_name}'
            }

    for output_name in model.event_info.event_particles.names:
        if output_log_transform:
            output_names.append(f"{output_name}_assignment_log_probability")
        else:
            output_names.append(f"{output_name}_assignment_probability")

    for output_name in model.event_info.event_particles.names:
        if output_log_transform:
            output_names.append(f"{output_name}_detection_log_probability")
        else:
            output_names.append(f"{output_name}_detection_probability")

    for regression in model.training_dataset.regressions.keys():
        output_names.append(regression)

    for classification in model.training_dataset.classifications.keys():
        output_names.append(classification)

    if output_embeddings:
        output_names.append("EVENT/embedding_vector")

        for particle, products in model.event_info.product_particles.items():
            output_names.append(f"{particle}/PARTICLE/embedding_vector")

            for product in products:
                output_names.append(f"{particle}/{product}/embedding_vector")

    return input_names, output_names, dynamic_axes


def main(
        log_directory: str,
        output_file: str,
        input_log_transform: bool,
        output_log_transform: bool,
        output_embeddings: bool,
        gpu: bool,
        opset: int,
        checkpoint: str,
):
    major_version, minor_version, *_ = torch.__version__.split(".")
    if int(major_version) == 2 and int(minor_version) == 0:
        raise RuntimeError("ONNX export with Torch 2.0.x is not working. Either install 2.1 or 1.13.")

    # load specific checkpoint
    if checkpoint is None:
        model = load_model(log_directory, cuda=gpu)
    else:
        checkpoint = checkpoint.split('/')[-1]
        print(f"Loading {log_directory.split('/')[-1]} checkpoint: {checkpoint.split('-')[0]}")
        model = load_model(log_directory, cuda=gpu, checkpoint=checkpoint)

    # Create wrapped model with flat inputs and outputs
    print(f"1. WrappedModel: ...")
    wrapped_model = WrappedModel(model, input_log_transform, output_log_transform, output_embeddings)
    print(f"1. WrappedModel: done")
    
    print(f"2. wrapped_model.to: ...")
    wrapped_model.to(model.device)
    print(f"2. wrapped_model.to: done")
    
    print(f"3. wrapped_model.eval: ...")
    wrapped_model.eval()
    print(f"3. wrapped_model.eval: done")
    
    print(f"4. for parameter in wrapped_model.parameters: ...")
    for parameter in wrapped_model.parameters():
        parameter.requires_grad_(False)
    print(f"4. for parameter in wrapped_model.parameters: ...")
    
    print(f"5. onnx specification: ...")
    input_names, output_names, dynamic_axes = onnx_specification(model, output_log_transform, output_embeddings)
    print(f"5. onnx specification: done")
    
    print(f"6. batches: ...")
    batch = next(iter(model.train_dataloader()))
    print(f"Batch: {batch}")
    sources = batch.sources
    print(f"Batch sources: {sources}")
    if gpu:
        sources = tree_map(lambda x: x.cuda(), batch.sources)
    sources = tree_map(lambda x: x[:1], sources)
    print(f"sources tree_map: {sources}")
    print(f"6. batches: done")

    print("-" * 60)
    print(f"Compiling network to ONNX model: {output_file}")
    if not input_log_transform:
        print("WARNING -- No input log transform! User must apply log transform manually. -- WARNING")
    print("-" * 60)
    
    # warnings.simplefilter("error")
    print(f"7. to_onnx: ...")
    wrapped_model.to_onnx(
        output_file,
        sources,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset
    )
    print(f"7. to_onnx: done")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("log_directory", type=str,
                        help="Pytorch Lightning Log directory containing the checkpoint and options file.")

    parser.add_argument("output_file", type=str,
                        help="Name to output the ONNX model to.")

    parser.add_argument("-g", "--gpu", action="store_true",
                        help="Trace the network on a gpu.")

    parser.add_argument("--opset", type=int, default=15,
                        help="ONNX opset version to use. Needs to be >= 14 for SPANet")
    
    parser.add_argument("--input-log-transform", action="store_true",
                        help="Exported model will apply log transformations to input features automatically.")

    parser.add_argument("--output-log-transform", action="store_true",
                        help="Exported model will output log probabilities. This is more numerically stable.")

    parser.add_argument("--output-embeddings", action="store_true",
                        help="Exported model will also output the embeddings for every part of the event.")
    
    parser.add_argument("-cp", "--checkpoint", type=str, default=None,
                        help="Checkpointed epoch we want to use for inference.")

    arguments = parser.parse_args()
    main(**arguments.__dict__)
