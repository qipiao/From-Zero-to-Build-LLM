import torch

from transformers import PretrainedConfig

class ModelConfig(PretrainedConfig):
    model_type = "PengAoMind"
    def __init__(
            self, 
            d_model=512, 
            num_hidden_layers=8, 
            intermediate_size=2048, 
            num_experts=4,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        self.dropout = kwargs.get("dropout", 0.0)
        self.vocab_size = kwargs.get("vocab_size", 6400)