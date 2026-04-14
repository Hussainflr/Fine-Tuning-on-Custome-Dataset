from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="outputs/final_model",
    device_map="auto"
)

prompt = "<|im_start|>user\nExplain AI<|im_end|>\n<|im_start|>assistant\n"

result = pipe(prompt, max_new_tokens=100)

print(result[0]["generated_text"])