from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn

from src import llama


@dataclass
class RecurrentParams:
    lora_layers: int = 2


class Recurrent(nn.Module):
    def __init__(self, params: RecurrentParams):
        super().__init__()


class Agent(nn.Module):
    def __init__(self, llama_model: llama.Model, params: RecurrentParams):
        super().__init__()

        self.__parameters = params
        self.__llama_model = llama_model
        self.__recurrent_model = Recurrent(params)

    def add_lora_layers_to_llama_model(self):
        self.__llama_model.freeze()

        if self.__parameters.lora_layers < 0:
            self.__parameters.lora_layers = len(self.__llama_model.model.layers)

        for l in self.__llama_model.model.layers[:self.__parameters.lora_layers]:
            l.self_attn.q_proj = llama.LoRALinear.from_linear(l.self_attn.q_proj)
            l.self_attn.v_proj = llama.LoRALinear.from_linear(l.self_attn.v_proj)

            if hasattr(l, "block_sparse_moe"):
                l.block_sparse_moe.gate = llama.LoRALinear.from_linear(l.block_sparse_moe.gate)
