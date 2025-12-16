from vllm import LLM, SamplingParams

# Load the model
llm = LLM("facebook/opt-125m")

# Define the prompt
prompt = "Once upon a time,"

# Set sampling parameters
sampling_params = SamplingParams(max_tokens=1)

# Generate text
outputs = llm.generate([prompt], sampling_params)

# Print the generated completions
for output in outputs:
    print("Prompt:", output.prompt)
    for choice in output.generated_choices:
        print("Generated Text:", choice.text)
