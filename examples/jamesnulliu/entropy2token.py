import torch

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.8, 
    top_p=0.95, 
    keep_entropy=True
)

def main():
    # Create an LLM.
    llm = LLM(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        trust_remote_code=True,
    )
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Entropy shape: {output.entropy.shape}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
