def format_example(example):
    return {
        "text": f"""<start_of_turn>user
{example['instruction']} {example['input']}
<end_of_turn>
<start_of_turn>model
{example['output']}
<end_of_turn>"""
    }