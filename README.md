# brainfuck agent

## Model

The model is in two parts:
- a language understanding part
- a recurrent instruction generation part

### Language understanding

The language understand part comes from a pre-trained model. We truncate the model to actually keep only the first
layers so that it transforms a sentence into a tensor that we can then use.

The base language model we use can be found here: [MLX-Community/Mistral-7B-Instruct-v0.2-4bit](https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.2-4bit).

It is a 4bit quantized version of the Mistral-7B-v0.2 model.

The config of this model is the following:
```json
{
    "vocab_size": 32000,
    "max_position_embeddings": 32768,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "sliding_window": null,
    "num_key_value_heads": 8,
    "hidden_act": "silu",
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-05,
    "use_cache": true,
    "rope_theta": 1000000.0,
    "attention_dropout": 0.0,
    "return_dict": true,
    "output_hidden_states": false,
    "output_attentions": false,
    "torchscript": false,
    "torch_dtype": "bfloat16",
    "use_bfloat16": false,
    "tf_legacy_loss": false,
    "pruned_heads": {},
    "tie_word_embeddings": false,
    "chunk_size_feed_forward": 0,
    "is_encoder_decoder": false,
    "is_decoder": false,
    "cross_attention_hidden_size": null,
    "add_cross_attention": false,
    "tie_encoder_decoder": false,
    "max_length": 20,
    "min_length": 0,
    "do_sample": false,
    "early_stopping": false,
    "num_beams": 1,
    "num_beam_groups": 1,
    "diversity_penalty": 0.0,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    "typical_p": 1.0,
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "encoder_no_repeat_ngram_size": 0,
    "bad_words_ids": null,
    "num_return_sequences": 1,
    "output_scores": false,
    "return_dict_in_generate": false,
    "forced_bos_token_id": null,
    "forced_eos_token_id": null,
    "remove_invalid_values": false,
    "exponential_decay_length_penalty": null,
    "suppress_tokens": null,
    "begin_suppress_tokens": null,
    "architectures": [
        "MistralForCausalLM"
    ],
    "finetuning_task": null,
    "id2label": {
        "0": "LABEL_0",
        "1": "LABEL_1"
    },
    "label2id": {
        "LABEL_0": 0,
        "LABEL_1": 1
    },
    "tokenizer_class": null,
    "prefix": null,
    "bos_token_id": 1,
    "pad_token_id": null,
    "eos_token_id": 2,
    "sep_token_id": null,
    "decoder_start_token_id": null,
    "task_specific_params": null,
    "problem_type": null,
    "_name_or_path": "/Users/prince_canuma/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/snapshots/41b61a33a2483885c981aa79e0df6b32407ed873",
    "transformers_version": "4.39.0.dev0",
    "model_type": "mistral",
    "quantization": {
        "group_size": 64,
        "bits": 4
    }
}
```

### Recurrent instruction generation

The recurrent instruction generation part is a recurrent neural network that takes the output of the language model
and generates one instruction at a time,

## Training

During training, the language understanding part if frozen and the rest is trained in a reinforcement learning fashion.

Here is how to start training:

```bash
uv run python main.py --model mlx-community/Mistral-7B-Instruct-v0.2-4bit --lora-layers 2
```

## Environment

We use our lovely brainfuck environment: [GitHub brainfuck interpreter](https://github.com/Bornlex/brainfuck).


