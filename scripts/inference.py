from vllm import LLM, SamplingParams
from vllm.router_log_helper import RouterLog

# Sample prompts.
prompts = [
    "Hello, my name is",
]


sampling_params = SamplingParams(temperature=0.0, max_tokens=16)

def main():

    # Create an LLM with prefix caching enabled.
    llm = LLM(
        # model="Qwen/Qwen3-235B-A22B",
        model="Qwen/Qwen3-30B-A3B",
        # model="Qwen/Qwen2.5-7B",
        # enable_prefix_caching=True,
        tensor_parallel_size=2,
        enforce_eager=True,
    )

    outputs = llm.generate(prompts, sampling_params)

    cached_generated_texts = []
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        cached_generated_texts.append(generated_text)
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)


if __name__ == "__main__":
    main()
    RouterLog.save("experiments/result.pt")
