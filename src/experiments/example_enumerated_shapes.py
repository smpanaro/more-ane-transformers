import coremltools as ct
import torch
import numpy as np
from  os_signpost import Signposter
signposter = Signposter("com.smpanaro.more-ane-transformers", Signposter.Category.PointsOfInterest)

"""
Very lightly-modified example of enumerated shapes from the coremltools docs.
To see both shapes run on the ANE.

10/25/2023 Result: Yes, they do when they are large enough.
"""

# Define a model for this example.
class TestConvModule(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=10, kernel_size=3):
        super(TestConvModule, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels,
                                    kernel_size)

    def forward(self, x):
        return self.conv(x)

# Trace the model with random input.
example_input = torch.rand(1, 3, 50, 50)
traced_model = torch.jit.trace(TestConvModule().eval(), example_input)

# Set the input_shape to use EnumeratedShapes.
input_shape = ct.EnumeratedShapes(shapes=[[1, 3, 250, 250],
                                          [1, 3, 500, 500],
                                          [1, 3, 670, 670]],
                                          default=[1, 3, 670, 670])

# Convert the model with input_shape.
model = ct.convert(traced_model,
                   inputs=[ct.TensorType(shape=input_shape, name="input")],
                   outputs=[ct.TensorType(name="output")],
                   convert_to="mlprogram",
                   )

model.save("enumerated_shapes.mlpackage")

# Test the model with predictions.
input_1 = np.random.rand(1, 3, 250, 250)
input_2 = np.random.rand(1, 3, 670, 670)

with signposter.use_interval("input_1"):
    output_1 = model.predict({"input": input_1})["output"]
    print("output shape {} for input shape {}".format(output_1.shape, input_1.shape))
with signposter.use_interval("input_2"):
    output_2 = model.predict({"input": input_2})["output"]
    print("output shape {} for input shape {}".format(output_2.shape, input_2.shape))