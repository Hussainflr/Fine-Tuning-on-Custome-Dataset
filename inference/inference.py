from transformers import pipeline

def generate(model_path, prompt):
    pipe = pipeline(
        "text-generation",
        model=model_path
    )

    return pipe(prompt, max_new_tokens=100)[0]["generated_text"]


if __name__ == "__main__":
    print(generate("outputs/final_model", "Explain AI"))