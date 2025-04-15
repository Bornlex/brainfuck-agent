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

    llama_args = agent.AgentParams(
        llama_path=args.model,
    )

    agent_model = agent.LanguageAgent()

    generation = utils.generate(
        t.encode('This is a test, say the brainfuck, the name of the programming language!'),
        m,
    )

    prompt(m, t, args)
