#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
from collections import OrderedDict

import coremltools as ct
from coremltools.converters.mil import Block, Program, Var
from coremltools.converters.mil.frontend.milproto.load import load as _milproto_to_pymil
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Placeholder
from coremltools.converters.mil.mil import types as types
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
# from coremltools.converters.mil.testing_utils import random_gen_input_feature_type

import gc

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import numpy as np
import os
import shutil
import time


def _verify_output_correctness_of_chunks(full_model, first_chunk_model,
                                         second_chunk_model):
    """ Verifies the end-to-end output correctness of full (original) model versus chunked models
    """
    # Generate inputs for first chunk and full model
    # input_dict = {}
    # for input_desc in full_model._spec.description.input:
    #     input_dict[input_desc.name] = random_gen_input_feature_type(input_desc)

    # # Generate outputs for first chunk and full model
    # outputs_from_full_model = full_model.predict(input_dict)
    # outputs_from_first_chunk_model = first_chunk_model.predict(input_dict)

    # # Prepare inputs for second chunk model from first chunk's outputs and regular inputs
    # second_chunk_input_dict = {}
    # for input_desc in second_chunk_model._spec.description.input:
    #     if input_desc.name in outputs_from_first_chunk_model:
    #         second_chunk_input_dict[
    #             input_desc.name] = outputs_from_first_chunk_model[
    #                 input_desc.name]
    #     else:
    #         second_chunk_input_dict[input_desc.name] = input_dict[
    #             input_desc.name]

    # # Generate output for second chunk model
    # outputs_from_second_chunk_model = second_chunk_model.predict(
    #     second_chunk_input_dict)

    # Verify correctness across all outputs from second chunk and full model
    print("Skipping output correctness check.")
    # for out_name in outputs_from_full_model.keys():
    #     torch2coreml.report_correctness(
    #         original_outputs=outputs_from_full_model[out_name],
    #         final_outputs=outputs_from_second_chunk_model[out_name],
    #         log_prefix=f"{out_name}")


def _load_prog_from_mlmodel(model):
    """ Load MIL Program from an MLModel
    """
    model_spec = model.get_spec()
    start_ = time.time()
    logger.info(
        "Loading MLModel object into a MIL Program object (including the weights).."
    )
    prog = _milproto_to_pymil(
        model_spec=model_spec,
        specification_version=model_spec.specificationVersion,
        file_weights_dir=model.weights_dir,
    )
    logger.info(f"Program loaded in {time.time() - start_:.1f} seconds")

    return prog

class Chunk():
    def __init__(self, start_op_idx, end_op_idx, cumulative_size_in_mb):
        self.start_op_idx = start_op_idx
        self.end_op_idx = end_op_idx
        self.cumulative_size_in_mb = cumulative_size_in_mb

    def save(self, prog, name):
        mlmodel = ct.convert(
            prog,
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_ONLY,
            minimum_deployment_target=ct.target.iOS16, # TODO: Is this needed?
        )
        mlmodel.save(name)

    def __repr__(self):
        return f"<Chunk start={self.start_op_idx} end={self.end_op_idx} size={self.cumulative_size_in_mb}MB>"

def _get_op_idx_split_location(prog: Program):
    """ Find the op that approximately bisects the graph as measure by weights size on each side
    """
    main_block = prog.functions["main"]
    total_size_in_mb = 0

    for op in main_block.operations:
        if op.op_type == "const" and isinstance(op.val.val, np.ndarray):
            size_in_mb = op.val.val.size * op.val.val.itemsize / (1024 * 1024)
            total_size_in_mb += size_in_mb
    if total_size_in_mb < 3072:
        print("You /may/ not need to split this model in order to get it to run on the Neural Engine.")
    half_size = 1800 # Under XXGB. 2400 too high. 1800 works.
    if total_size_in_mb < half_size:
        half_size = total_size_in_mb / 2
    print(f"Total size: {total_size_in_mb}MB")
    print(f"Target split size: {half_size}MB")

    # Find the first non const op (single child), where the total cumulative size exceeds
    # the half size for the first time
    cumulative_size_in_mb = 0
    for op in main_block.operations:
        if op.op_type == "const" and isinstance(op.val.val, np.ndarray):
            size_in_mb = op.val.val.size * op.val.val.itemsize / (1024 * 1024)
            cumulative_size_in_mb += size_in_mb

        if (cumulative_size_in_mb > half_size and op.op_type != "const"
                and len(op.outputs) == 1
                and len(op.outputs[0].child_ops) == 1):
            op_idx = main_block.operations.index(op)
            return op_idx, cumulative_size_in_mb, total_size_in_mb

def _get_op_idx_split_locations(prog: Program):
    """ Find the op that approximately bisects the graph as measure by weights size on each side
    """
    main_block = prog.functions["main"]
    total_size_in_mb = 0

    for op in main_block.operations:
        if op.op_type == "const" and isinstance(op.val.val, np.ndarray):
            size_in_mb = op.val.val.size * op.val.val.itemsize / (1024 * 1024)
            total_size_in_mb += size_in_mb
    if total_size_in_mb < 1800:
        print("You /may/ not need to split this model in order to get it to run on the Neural Engine.")
    chunk_size = 1800 # Under XXGB. 2400 too high. 1800 works.
    next_split_size = chunk_size
    print(f"Total size: {total_size_in_mb}MB")
    print(f"Target split size: {chunk_size}MB")

    # Find the first non const op (single child), where the total cumulative size exceeds
    # the half size for the first time
    chunks = []
    cumulative_size_in_mb = 0
    for op in main_block.operations:
        if op.op_type == "const" and isinstance(op.val.val, np.ndarray):
            size_in_mb = op.val.val.size * op.val.val.itemsize / (1024 * 1024)
            cumulative_size_in_mb += size_in_mb

        if (cumulative_size_in_mb > next_split_size and op.op_type != "const"
                and len(op.outputs) == 1
                and len(op.outputs[0].child_ops) == 1):
            op_idx = main_block.operations.index(op)
            start_op_idx = 0 if len(chunks) == 0 else chunks[-1].end_op_idx
            chunks.append(Chunk(start_op_idx, op_idx, cumulative_size_in_mb))
            next_split_size = cumulative_size_in_mb + chunk_size
            # return op_idx, cumulative_size_in_mb, total_size_in_mb
    return chunks


def _get_first_chunk_outputs(block, start_op_idx, op_idx):
    # Get the list of all vars that go across from first program (all ops from 0 to op_idx (inclusive))
    # to the second program (all ops from op_idx+1 till the end). These all vars need to be made the output
    # of the first program and the input of the second program
    boundary_vars = set()
    for i in range(start_op_idx, op_idx + 1):
        op = block.operations[i]
        for var in op.outputs:
            if var.val is None:  # only consider non const vars
                for child_op in var.child_ops:
                    child_op_idx = block.operations.index(child_op)
                    if child_op_idx > op_idx:
                        boundary_vars.add(var)
    return list(boundary_vars)


@block_context_manager
def _add_fp32_casts(block, boundary_vars):
    new_boundary_vars = []
    for var in boundary_vars:
        if var.dtype != types.fp16:
            new_boundary_vars.append(var)
        else:
            fp32_var = mb.cast(x=var, dtype="fp32", name=var.name)
            new_boundary_vars.append(fp32_var)
    return new_boundary_vars


def _make_first_chunk_prog(prog, op_idx):
    """ Build first chunk by declaring early outputs and removing unused subgraph
    """
    block = prog.functions["main"]
    boundary_vars = _get_first_chunk_outputs(block, 0, op_idx)

    # Due to possible numerical issues, cast any fp16 var to fp32
    new_boundary_vars = _add_fp32_casts(block, boundary_vars)

    block.outputs.clear()
    block.set_outputs(new_boundary_vars)
    PASS_REGISTRY["common::dead_code_elimination"](prog)
    return prog


def _make_second_chunk_prog(prog, previous_start_op_idx, start_op_idx, end_op_idx, is_last=False):
    """ Build second chunk by rebuilding a pristine MIL Program from MLModel
    """
    block = prog.functions["main"]
    block.opset_version = ct.target.iOS16

    # First chunk outputs are second chunk inputs (e.g. skip connections)
    boundary_vars = _get_first_chunk_outputs(block, previous_start_op_idx, start_op_idx)

    output_boundary_vars = _get_first_chunk_outputs(block, start_op_idx, end_op_idx)
    if not is_last:
        new_boundary_vars = _add_fp32_casts(block, output_boundary_vars)
        block.outputs.clear()
        block.set_outputs(new_boundary_vars)

    # This op will not be included in this program. Its output var will be made into an input
    boundary_op = block.operations[start_op_idx]

    logger.info(f"boundary vars (inputs) {previous_start_op_idx}-{start_op_idx} {[v.name for v in boundary_vars]}")
    logger.info(f"boundary op: {boundary_op}".rstrip())
    logger.info(f"boundary vars (outputs) {start_op_idx}-{end_op_idx} {[v.name for v in output_boundary_vars]}")

    # Add all boundary ops as inputs
    with block:
        for var in boundary_vars:
            new_placeholder = Placeholder(
                sym_shape=var.shape,
                dtype=var.dtype if var.dtype != types.fp16 else types.fp32,
                name=var.name,
            )
            # logger.info(f"New placeholder: {var.name}")

            block._input_dict[
                new_placeholder.outputs[0].name] = new_placeholder.outputs[0]

            block.function_inputs = tuple(block._input_dict.values())
            new_var = None
            if var.dtype == types.fp16:
                new_var = mb.cast(x=new_placeholder.outputs[0],
                                  dtype="fp16",
                                  before_op=var.op)
            else:
                new_var = new_placeholder.outputs[0]

            # logger.info(f"Replacing {var.name} with {new_var.name}")
            block.replace_uses_of_var_after_op(
                anchor_op=boundary_op,
                old_var=var,
                new_var=new_var,
            )

    PASS_REGISTRY["common::dead_code_elimination"](prog)

    # Remove any unused inputs
    new_input_dict = OrderedDict()
    for k, v in block._input_dict.items():
        if len(v.child_ops) > 0:
            new_input_dict[k] = v
    block._input_dict = new_input_dict
    block.function_inputs = tuple(block._input_dict.values())
    logger.info(f"final inputs: {[v.name for v in block.function_inputs]}")

    return prog

def save_chunk(prog_chunk, filename):
    model_chunk = ct.convert(
        prog_chunk,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS16, # TODO: Is this needed?
    )
    model_chunk.save(filename)

def _recursively_split_program(prog, model, name, chunk_idx):
    filename = f"{name}_chunk{chunk_idx}.mlpackage"

    split_res = _get_op_idx_split_location(prog)
    if split_res is None:
        logger.info("No need to split further.")
        save_chunk(prog, filename)
        logger.info("Saved last chunk")
        return

    op_idx, first_chunk_weights_size, total_weights_size = split_res
    main_block = prog.functions["main"]
    incision_op = main_block.operations[op_idx]

    prog_chunk1 = _make_first_chunk_prog(prog, op_idx)
    save_chunk(prog_chunk1)
    del prog_chunk1
    gc.collect()

    # Build the second chunk
    # NOTE: this part is where it gets tricky: reloading from the full model isntead of a subset
    prog_chunk2 = _make_second_chunk_prog(_load_prog_from_mlmodel(model), op_idx)
    _recursively_split_program(prog_chunk2, model, name, chunk_idx+1)

def main(args):
    os.makedirs(args.o, exist_ok=True)

    # Check filename extension
    mlpackage_name = os.path.basename(args.mlpackage_path)
    name, ext = os.path.splitext(mlpackage_name)
    assert ext == ".mlpackage", f"`--mlpackage-path` (args.mlpackage_path) is not an .mlpackage file"

    # Load CoreML model
    logger.info("Loading model from {}".format(args.mlpackage_path))
    start_ = time.time()
    model = ct.models.MLModel(
        args.mlpackage_path,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    logger.info(
        f"Loading {args.mlpackage_path} took {time.time() - start_:.1f} seconds"
    )

    # Load the MIL Program from MLModel
    prog = _load_prog_from_mlmodel(model)

    logger.info(f"Total ops: {len(prog.functions['main'].operations)}")

    chunks: list[Chunk] = _get_op_idx_split_locations(prog)
    logger.info(f"{args.mlpackage_path} will chunked into {len(chunks)} pieces.")
    for c in chunks:
        logger.info(c)

    # From start to first chunk op idx.
    # prog_chunk1 = _make_first_chunk_prog(prog, chunks[0].end_op_idx)
    # logger.info(f"Converting and saving first chunk up to op {chunks[0].end_op_idx}")
    # filename = f"{name}_chunk1.mlpackage"
    # chunks[0].save(prog_chunk1, filename)
    # logger.info(f"Saved first chunk {filename} of {len(chunks)} total")

    # del prog_chunk1
    # gc.collect()

    prev_start_index = 0
    for idx, chunk in [(i+1, c) for i,c in enumerate(chunks)]:
        filename = f"{name}_chunk{idx}.mlpackage"
        full_prog = _load_prog_from_mlmodel(model)
        logger.info(f"Converting and saving chunk #{idx} from op {chunk.start_op_idx}-{chunk.end_op_idx}")
        is_last = chunk.end_op_idx == chunks[-1].end_op_idx
        prog_chunkn = _make_second_chunk_prog(full_prog, prev_start_index, chunk.start_op_idx, chunk.end_op_idx, is_last=is_last)
        chunk.save(prog_chunkn, filename)

        del full_prog
        del prog_chunkn
        gc.collect()
        logger.info(f"Saved chunk {filename} of {len(chunks)} total")

        prev_start_index = chunk.start_op_idx


    # Compute the incision point by bisecting the program based on weights size
    # op_idx, first_chunk_weights_size, total_weights_size = _get_op_idx_split_location(
    #     prog)
    # main_block = prog.functions["main"]
    # incision_op = main_block.operations[op_idx]
    # logger.info(f"{args.mlpackage_path} will chunked into two pieces.")
    # logger.info(
    #     f"The incision op: name={incision_op.name}, type={incision_op.op_type}, index={op_idx}/{len(main_block.operations)}"
    # )
    # logger.info(f"First  chunk size = {first_chunk_weights_size:.2f} MB")
    # logger.info(
    #     f"Second chunk size = {total_weights_size - first_chunk_weights_size:.2f} MB"
    # )

    # # Build first chunk (in-place modifies prog by declaring early exits and removing unused subgraph)
    # prog_chunk1 = _make_first_chunk_prog(prog, op_idx)

    # # Build the second chunk
    # prog_chunk2 = _make_second_chunk_prog(_load_prog_from_mlmodel(model),
    #                                       op_idx)

    # if not args.check_output_correctness:
    #     # Original model no longer needed in memory
    #     del model
    #     gc.collect()

    # # Convert the MIL Program objects into MLModels
    # logger.info("Converting the two programs")
    # model_chunk1 = ct.convert(
    #     prog_chunk1,
    #     convert_to="mlprogram",
    #     compute_units=ct.ComputeUnit.CPU_ONLY,
    #     minimum_deployment_target=ct.target.iOS16, # TODO: Is this needed?
    # )
    # del prog_chunk1
    # gc.collect()
    # logger.info("Conversion of first chunk done.")

    # model_chunk2 = ct.convert(
    #     prog_chunk2,
    #     convert_to="mlprogram",
    #     compute_units=ct.ComputeUnit.CPU_ONLY,
    #     minimum_deployment_target=ct.target.iOS16, # TODO: Is this needed?
    # )
    # del prog_chunk2
    # gc.collect()
    # logger.info("Conversion of second chunk done.")

    # # Verify output correctness
    # if args.check_output_correctness:
    #     logger.info("Verifying output correctness of chunks")
    #     _verify_output_correctness_of_chunks(
    #         full_model=model,
    #         first_chunk_model=model_chunk1,
    #         second_chunk_model=model_chunk2,
    #     )

    # # Remove original (non-chunked) model if requested
    # if args.remove_original:
    #     logger.info(
    #         "Removing original (non-chunked) model at {args.mlpackage_path}")
    #     shutil.rmtree(args.mlpackage_path)
    #     logger.info("Done.")

    # # Save the chunked models to disk
    # out_path_chunk1 = os.path.join(args.o, name + "_chunk1.mlpackage")
    # out_path_chunk2 = os.path.join(args.o, name + "_chunk2.mlpackage")

    # logger.info(
    #     f"Saved chunks in {args.o} with the suffix _chunk1.mlpackage and _chunk2.mlpackage"
    # )
    # model_chunk1.save(out_path_chunk1)
    # model_chunk2.save(out_path_chunk2)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlpackage-path",
        required=True,
        help=
        "Path to the mlpackage file to be split into two mlpackages of approximately same file size.",
    )
    parser.add_argument(
        "-o",
        required=True,
        help=
        "Path to output directory where the two model chunks should be saved.",
    )
    parser.add_argument(
        "--remove-original",
        action="store_true",
        help=
        "If specified, removes the original (non-chunked) model to avoid duplicating storage."
    )
    parser.add_argument(
        "--check-output-correctness",
        action="store_true",
        help=
        ("If specified, compares the outputs of original Core ML model with that of pipelined CoreML model chunks and reports PSNR in dB. ",
         "Enabling this feature uses more memory. Disable it if your machine runs out of memory."
         ))

    args = parser.parse_args()
    main(args)
