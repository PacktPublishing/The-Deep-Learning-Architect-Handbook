from typing import List

import numpy as np
import tensorrt as trt
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, TensorType

from transformer_deploy.backends.trt_utils import load_engine

model = "roneneldan/TinyStories-3M"
tensorrt_path = "/models/transformer_tensorrt_text_generation/1/model.plan"

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model_input_names = self.tokenizer.model_input_names
        trt_logger = trt.Logger(trt.Logger.VERBOSE)
        runtime = trt.Runtime(trt_logger)
        self.model = load_engine(
            runtime=runtime, engine_file_path=tensorrt_path
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            query = [t.decode("UTF-8") for t in pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy().tolist()]
            tokens = self.tokenizer(
                text=query, return_tensors=TensorType.PYTORCH, return_attention_mask=False
            )
            # tensorrt uses int32 as input type, ort also because we force the format
            input_ids = tokens.input_ids.type(dtype=torch.int32)
            input_ids = input_ids.to("cuda")
            output_seq: torch.Tensor = self.model({"input_ids": input_ids})['output'].cpu().argmax(2)
            decoded_texts: List[str] = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in output_seq]
            tensor_output = [pb_utils.Tensor("OUTPUT_TEXT", np.array(t, dtype=object)) for t in decoded_texts]
            responses.append(pb_utils.InferenceResponse(tensor_output))
        return responses
