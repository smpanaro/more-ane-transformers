import coremltools as ct
from coremltools.converters.mil.mil import Function, Program
from coremltools.converters.mil.mil import Builder as mb
import numpy as np

"""
For filing a GH issue about how compiled pipelines include
multiple copies of the weights.
"""

def _make_model(input_name, input_length,
                output_name, output_length,
                convert_to):

    weight_tensor = np.arange(input_length * output_length, dtype='float32')
    weight_tensor = weight_tensor.reshape(output_length, input_length)

    prog = Program()
    func_inputs = {input_name: mb.placeholder(shape=(input_length,))}
    with Function(func_inputs) as ssa_fun:
        input = ssa_fun.inputs[input_name]
        y = mb.linear(x=input, weight=weight_tensor, name=output_name)
        ssa_fun.set_outputs([y])
        prog.add_function("main", ssa_fun)

    return ct.convert(prog, convert_to=convert_to)


if __name__ == "__main__":
    # Create models
    m1 = _make_model("x", 20, "y1", 10, "mlprogram")
    m2 = _make_model("y1", 10, "y2", 2, "mlprogram")

    pipeline_model = ct.utils.make_pipeline(m1, m2)

    pipeline_model.save("test-pipeline.mlpackage")