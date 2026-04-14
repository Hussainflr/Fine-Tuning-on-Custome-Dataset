from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name, use_4bit=True):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=use_4bit,
        trust_remote_code=True
    )

    return model, tokenizer