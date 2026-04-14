from peft import LoraConfig, get_peft_model

def apply_lora(model, config):
    lora_config = LoraConfig(
        r=config["r"],
        lora_alpha=config["alpha"],
        lora_dropout=config["dropout"],
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    return get_peft_model(model, lora_config)