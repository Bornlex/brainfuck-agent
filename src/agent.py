from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn

from src import llama, utils


@dataclass
class RecurrentParams:
    lora_layers: int = 2


@dataclass
class AgentParams:
    llama_path: str
    transformer_layers: int = -1
    recurrent_params: RecurrentParams | None = None


class Recurrent(nn.Module):
    def __init__(self, params: RecurrentParams):
        super().__init__()


class LanguageAgent(nn.Module):
    def __init__(self, params: AgentParams):
        super().__init__()

        self.__parameters = params
        self.__llama_model = self.__load_lama_model(
            params.llama_path,
            params.transformer_layers
        )

    @property
    def tokenizer(self):
        pass

    @property
    def llama(self):
        return self.__llama_model

    def load_and_build_model(self, llama_model_path: str, transformer_layers: int):
        llama_model, tokenizer, _ = utils.load(llama_model_path)
        model.add_lora_layers_to_llama_model()

        model.freeze()

        if lora_layers < 0:
            lora_layers = len(model.model.layers)

        for l in model.model.layers[len(model.model.layers) - lora_layers:]:
            l.self_attn.q_proj = llama.LoRALinear.from_linear(l.self_attn.q_proj)
            l.self_attn.v_proj = llama.LoRALinear.from_linear(l.self_attn.v_proj)

            if hasattr(l, "block_sparse_moe"):
                l.block_sparse_moe.gate = llama.LoRALinear.from_linear(l.block_sparse_moe.gate)

        return model, tokenizer

    def add_lora_layers_to_llama_model(self):
        self.__llama_model.freeze()

        if self.__parameters.lora_layers < 0:
            self.__parameters.lora_layers = len(self.__llama_model.model.layers)

        for l in self.__llama_model.model.layers[:self.__parameters.lora_layers]:
            l.self_attn.q_proj = llama.LoRALinear.from_linear(l.self_attn.q_proj)
            l.self_attn.v_proj = llama.LoRALinear.from_linear(l.self_attn.v_proj)

            if hasattr(l, "block_sparse_moe"):
                l.block_sparse_moe.gate = llama.LoRALinear.from_linear(l.block_sparse_moe.gate)
