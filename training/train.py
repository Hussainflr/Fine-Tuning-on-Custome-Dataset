import yaml
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

from model.load_model import load_model
from model.lora import apply_lora
from data.format import format_example


def main():
    config = yaml.safe_load(open("./configs/training_config.yaml"))

    # Load model
    model, tokenizer = load_model(
        config["model_name"],
        config["quantization"]["use_4bit"]
    )

    # Apply LoRA
    model = apply_lora(model, config["lora"])

    # Load dataset
    dataset = load_dataset("json", data_files="data/raw/dataset.json")

    dataset = dataset.map(format_example)

    # Training args
    training_args = TrainingArguments(
        output_dir="outputs/checkpoints",
        per_device_train_batch_size=config["training"]["batch_size"],
        num_train_epochs=config["training"]["epochs"],
        learning_rate=config["training"]["learning_rate"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        fp16=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text"
    )

    trainer.train()

    # Save model
    trainer.model.save_pretrained("outputs/final_model")
    tokenizer.save_pretrained("outputs/final_model")


if __name__ == "__main__":
    main()