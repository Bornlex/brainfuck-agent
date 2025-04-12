import argparse

from src import llama, utils, agent


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a brainfuck programming model.")
    parser.add_argument(
        "--model",
        default="mlx_model",
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to fine-tune (also impacts the number of layers to be kept from le Llama base model).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Minibatch size."
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=1000,
        help="Iterations to train for."
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Max number of tokens to generate.'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Temperature for sampling.'
    )

    return parser.parse_args()


def load_and_build_model(
        llama_model_path: str,
        lora_layers: int = -1,
        tokenizer_config: dict | None = None
):
    llama_model, tokenizer, _ = utils.load(llama_model_path, tokenizer_config)
    model = agent.Agent(llama_model, None)
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


def prompt(model, tokenizer, arguments):
    while True:
        user_prompt = input('> ')
        if user_prompt == 'exit':
            return

        c = utils.completion(
            model,
            user_prompt,
            tokenizer,
            arguments
        )
        print(f'\t{c}', flush=True)


if __name__ == '__main__':
    args = parse_arguments()
    m, t = load_and_build_model(
        args.model,
        args.lora_layers,
    )

    generation = utils.generate(
        t.encode('This is a test, say the brainfuck, the name of the programming language!'),
        m,
    )

    prompt(m, t, args)
