import torch
import numpy as np

class MLModelProxy:
    """Just the fun bits from coremltools that allows us to use pre-compiled models."""
    def __init__(self, model_path, compute_unit):
        from coremltools.libcoremlpython import _MLModelProxy
        self.__proxy__ = _MLModelProxy(model_path, compute_unit.name)

    def _update_float16_multiarray_input_to_float32(self, input_data):
        for k, v in input_data.items():
            if isinstance(v, np.ndarray) and v.dtype == np.float16:
                input_data[k] = v.astype(np.float32)

    def _convert_tensor_to_numpy(self, input_dict):
        _HAS_TORCH = True
        def convert(given_input):
            if isinstance(given_input, np.ndarray):
                sanitized_input = given_input
            elif _HAS_TORCH and isinstance(given_input, torch.Tensor):
                sanitized_input = given_input.detach().numpy()
            # elif (_HAS_TF_1 or _HAS_TF_2) and isinstance(given_input, _tf.Tensor):
            #     sanitized_input = given_input.eval(session=_tf.compat.v1.Session())
            else:
                sanitized_input = np.array(given_input)
            return sanitized_input

        model_input_to_types = {}
        # Don't pass bad input!
        # TODO: Steal a spec out of the mlpackage.
        # for inp in self._spec.description.input:
        #     type_value = inp.type.multiArrayType.dataType
        #     type_name = inp.type.multiArrayType.ArrayDataType.Name(type_value)
        #     if type_name != "INVALID_ARRAY_DATA_TYPE":
        #         model_input_to_types[inp.name] = type_name

        for given_input_name, given_input in input_dict.items():
            # if not given_input_name in model_input_to_types:
            #     continue
            input_dict[given_input_name] = convert(given_input)

    def predict(self, data):
        """
        You are responsible for not passing bad input! If you're not sure and the
        model seems to be hanging, try running with a mlpackage
        """
        # self._verify_input_dict(data)
        self._convert_tensor_to_numpy(data)
        self._update_float16_multiarray_input_to_float32(data)
        return self.__proxy__.predict(data)

