import glob
from huggingface_hub import snapshot_download
import json
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path
import transformers
from typing import Generator, Tuple

from src import llama


def split_train_val_test(content: list, k_train: float = 0.6):
    l1 = int(len(content) * k_train)
    l2 = (len(content) - l1) // 2

    return content[:l1], content[l1:l1 + l2], content[l1 + l2:]


def make_shards(weights: dict, max_file_size_gibibyte: int = 15):
    max_file_size_bytes = max_file_size_gibibyte << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)

    return shards


def save_model(save_dir: str, weights, tokenizer, config):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights, max_file_size_gibibyte=5)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_size = sum(v.nbytes for v in weights.values())
    index_data = {"metadata": {"total_size": total_size}, "weight_map": {}}

    for i, shard in enumerate(shards):
        shard_name = shard_file_format.format(i + 1, shards_count)
        mx.save_safetensors(
            str(save_dir / shard_name), shard, metadata={"format": "mlx"}
        )
        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name
        del shard

    tokenizer.save_pretrained(save_dir)
    with open(save_dir / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)

    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }
    with open(save_dir / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def load_from_huggingface(hf_repo: str) -> Path:
    return Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
        )
    )


def read_config(file_path: Path) -> Tuple[llama.LlamaArgs, dict | None]:
    with open(file_path, "r") as f:
        config = json.loads(f.read())

    return llama.LlamaArgs.from_dict(config), config.get("quantization", None)


def quantize(model, quantization: dict):
    pass


def load(path_or_hf_repo: str):
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = load_from_huggingface(path_or_hf_repo)

    llama_args, quantization = read_config(model_path / "config.json")

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if len(weight_files) == 0:
        raise FileNotFoundError("No safetensors found in {}".format(model_path))

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf).items())

    model = llama.Model(llama_args)
    if quantization is not None:
        print('model is not quantized yet, quantizing now')
        class_predicate = (
            lambda p, m: isinstance(m, (nn.Linear, nn.Embedding))
            and f"{p}.scales" in weights
        )
        nn.quantize(
            model,
            **quantization,
            class_predicate=class_predicate,
        )

    model.load_weights(list(weights.items()))

    mx.eval(model.parameters())
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer, llama_args


def generate(
    prompt: mx.array, model: nn.Module, temp: float = 0.0
) -> Generator[mx.array, None, None]:
    """
    Generate text based on the given prompt and model.

    Args:
        prompt (mx.array): The input prompt.
        model (nn.Module): The model to use for generation.
        temp (float): The temperature for sampling. If temp is 0, use max sampling.

    Yields:
        mx.array: The generated text.
    """

    def sample(logits: mx.array) -> mx.array:
        return (
            mx.argmax(logits, axis=-1)
            if temp == 0
            else mx.random.categorical(logits * (1 / temp))
        )

    y = prompt
    cache = None
    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits)

        yield y


def completion(model, prompt, tokenizer, args) -> str:
    prompt = mx.array(tokenizer.encode(prompt))

    tokens = []
    skip = 0
    for token, n in zip(
        generate(prompt, model, args.temperature),
        range(args.max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break

        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        if len(s) - skip > 1:
            print(s[skip:-1], end="", flush=True)
            skip = len(s) - 1

    if len(tokens) == 0:
        return "No tokens generated for this prompt"

    return tokenizer.decode(tokens)[skip:]
