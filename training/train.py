import yaml
from functools import partial

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from format import format_example




# Main Training Function

def main():
    # Load config
    config = yaml.safe_load(open("configs/training_config.yaml"))

    model_name = config["model_name"]

  
    # Load Tokenizer
  
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    tokenizer.pad_token = tokenizer.eos_token

  
    # Load Model (QLoRA Ready)
  
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=config["quantization"]["use_4bit"],
        trust_remote_code=True
    )

  
    # Apply LoRA
  
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, lora_config)

  
    # Load Dataset
  
    dataset = load_dataset(
        "json",
        data_files="data/raw/dataset.jsonl"
    )

    # Format dataset
    dataset = dataset.map(
        partial(format_example, tokenizer=tokenizer),
        remove_columns=dataset["train"].column_names
    )


    # Training Arguments
  
    training_args = TrainingArguments(
        output_dir="outputs/checkpoints",
        per_device_train_batch_size=config["training"]["batch_size"],
        num_train_epochs=config["training"]["epochs"],
        learning_rate=config["training"]["learning_rate"],
        logging_steps=10,
        save_steps=100,
        fp16=True,
        bf16=False,
        report_to="none"
    )

    # Trainer

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=1024
    )

 
    # Train
   
    trainer.train()


    # Save Model
  
    trainer.model.save_pretrained("outputs/final_model")
    tokenizer.save_pretrained("outputs/final_model")



if __name__ == "__main__":
    main()