# 🤖 Qwen Fine-Tuning + Chat UI (LoRA / QLoRA)

A complete end-to-end project to **fine-tune a small LLM** using **LoRA / QLoRA** and deploy it with a **Streamlit Chat UI**.

Built using:

* Hugging Face Transformers
* PEFT (LoRA)
* TRL (SFT Trainer)
* Streamlit

---

## 🚀 Features

* ✅ Fine-tune LLM with your own dataset
* ✅ Supports **LoRA / QLoRA (4-bit)**
* ✅ Uses **Qwen2-1.5B-Instruct (lightweight & powerful)**
* ✅ Clean modular project structure
* ✅ ChatGPT-like UI with:

  * 💬 Conversation memory
  * ⚡ Streaming responses
  * 🎛️ Adjustable parameters (temperature, top-p, max tokens)
  * 🧹 Reset chat button

---

## 📁 Project Structure

```
llm-finetuning-project/
│
├── data/
│   └── raw/dataset.jsonl
│
├── configs/
│   └── training_config.yaml
│
├
│── data/format.py
│── model/load_model.py
│── model/lora.py
│── training/train.py
│── inference/generate.py
│
├── outputs/
│   ├── checkpoints/
│   └── final_model/
│
├── app.py              # Streamlit chat UI
├── requirements.txt
├── run.sh
└── README.md
```

---

## 📊 Dataset Format

Create a dataset in JSONL format:

```json
{"instruction": "Explain gravity", "input": "", "output": "Gravity is a force..."}
{"instruction": "Translate to Urdu", "input": "Hello", "output": "سلام"}
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training

```bash
bash run.sh
```

or manually:

```bash
python -m src.training.train
```

---

## 🧠 Model Used

```
Qwen/Qwen2-1.5B-Instruct
```

* Lightweight (~1.5B parameters)
* Strong instruction-following
* Works great with LoRA / QLoRA

---

## ⚡ Inference (CLI)

```bash
python -m src.inference.generate
```

---

## 🌐 Streamlit Chat UI

Run:

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🎛️ UI Features

* 💬 Chat interface (like ChatGPT)
* ⚡ Streaming responses
* 🎚️ Adjustable:

  * Temperature
  * Top-p
  * Max tokens
* 🧹 Reset conversation

---

## 🔧 Configuration

Edit:

```
configs/training_config.yaml
```

Example:

```yaml
model_name: Qwen/Qwen2-1.5B-Instruct

training:
  batch_size: 2
  epochs: 3
  learning_rate: 2e-4

lora:
  r: 16
  alpha: 32
  dropout: 0.05

quantization:
  use_4bit: true
```

---

## ⚠️ Notes

* For Mac (M1/M2), set:

  ```yaml
  use_4bit: false
  ```
* Smaller models = faster training
* Dataset quality > dataset size

---

## 🚀 Future Improvements

* 🎤 Voice assistant (speech-to-text + TTS)
* 📄 RAG (chat with documents / PDFs)
* 🐳 Docker deployment
* 🌐 FastAPI backend
* 📊 Evaluation & benchmarking

---

## 🤝 Contributing

Feel free to fork and improve this project!

---

## ⭐ Acknowledgements

* Hugging Face
* Qwen (Alibaba)
* Open-source LLM community

---

## 📬 Contact

For questions or collaboration, reach out via GitHub.

---
