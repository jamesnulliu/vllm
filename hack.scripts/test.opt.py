import torch
from vllm import LLM, SamplingParams

torch.compiler.reset()

N_PROBS_PER_TOKEN=30

# Sample prompts.

prompts = [
    "Hello my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=1,
    max_tokens=5,
    logprobs=N_PROBS_PER_TOKEN,
)


def main():
    llm = LLM(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        trust_remote_code=True,
        max_logprobs=N_PROBS_PER_TOKEN,
        # "raw_logprobs", "raw_logits", "processed_logprobs", "processed_logits"
        # "soft_thinking", "soft_thinking_with_temperature"
        logprobs_mode="soft_thinking_with_temperature"
    )

    outputs = llm.generate(prompts, sampling_params)

    print("\nGenerated Outputs:\n" + "-" * 60)

    for output in outputs:
        output = output.outputs[0]
        print(f"Token IDs: {output.token_ids}")
        print(f"Output: {output.text!r}")
        print(f"Logprobs: {output.logprobs}")
        print("-" * 60)


if __name__ == "__main__":
    main()
