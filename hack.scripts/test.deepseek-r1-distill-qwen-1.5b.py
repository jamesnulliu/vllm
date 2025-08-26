import torch
from vllm import LLM, SamplingParams
from vllm.hack.probs import logprobs_to_df

torch.compiler.reset()

N_PROBS_PER_TOKEN = 30
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

prompts = [
    "Hello my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

prompts_as_chat = [
    [
        {
            "role": "system",
            # Deepseek R1
            "content": "This is a conversation between User and Assistant. "
            "The user asks a question, and the Assistant solves it. "
            "The assistant first thinks about the reasoning process in the "
            "mind and then provides the user with the answer. The reasoning "
            "process and answer are enclosed within <think> </think> and "
            "<answer> </answer> tags, respectively, i.e., <think> reasoning "
            "process here </think> <answer> answer here </answer>.",
        },
        {
            "role": "user",
            "content": "The proper divisors of 12 are 1, 2, 3, 4 and 6. "
            "A proper divisor of an integer $N$ is a positive divisor of $N$ "
            "that is less than $N$. What is the sum of the proper divisors "
            "of the sum of the proper divisors of 284?",
        },
    ],
]

sampling_params = SamplingParams(
    temperature=0.6,
    max_tokens=256,
    logprobs=N_PROBS_PER_TOKEN,
)


def main():
    llm = LLM(
        model=MODEL_NAME,
        gpu_memory_utilization=0.8,
        enforce_eager=False,
        trust_remote_code=True,
        max_logprobs=N_PROBS_PER_TOKEN,
        # "raw_logprobs", "raw_logits", "processed_logprobs", "processed_logits"
        # "soft_thinking", "soft_thinking_with_temperature"
        logprobs_mode="soft_thinking_with_temperature",
    )

    outputs = llm.generate(prompts, sampling_params)

    print("\nGenerated Outputs:\n" + "-" * 60)

    for i, output in enumerate(outputs):
        output = output.outputs[0]
        print(f"Token IDs: {output.token_ids}")
        print(f"Output: {output.text!r}")
        logprobs_to_df(output.logprobs).to_csv(f"output_{i}.csv")
        print("-" * 60)


if __name__ == "__main__":
    main()
