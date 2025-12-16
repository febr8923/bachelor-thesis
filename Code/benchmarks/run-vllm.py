from vllm import LLM, SamplingParams

llm = LLM("facebook/opt-125m")
prompt = "Once upon a time,"
sampling_params = SamplingParams(max_tokens=50, temperature=0.8)

outputs = llm.generate([prompt], sampling_params)

for output in outputs:
    print("Prompt:", output.prompt)
    for generated_text in output.outputs:
        print("Generated Text:", generated_text.text)
        print("-" * 50)
