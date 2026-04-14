def format_example(example, tokenizer):
    # Safely get fields
    instruction = example.get("instruction", "").strip()
    input_text = example.get("input", "").strip()
    output = example.get("output", "").strip()

    # Combine instruction + input cleanly
    if input_text:
        user_content = f"{instruction}\n{input_text}"
    else:
        user_content = instruction

    # Build chat messages
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output}
    ]

    # Apply Qwen chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text}
