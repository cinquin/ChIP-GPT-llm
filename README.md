# ChIP-GPT-llm
Code for ChIP-GPT, a Llama-based tool for question answering on SRA records

# Background
See "ChIP-GPT: managed large language model for robust data extraction from biomedical database records" (further details to be provided)

# Runtime environment
Tested with `python v3.10.2`, `cuda v11.7.1`, `gcc v11.2.0`, `transformers 4.28.0.dev` (commit id `151425ddb29d4ad1a121e8cce62000a2ac52d3ba`), and `peft v0.3.0.dev` (commit id `64f63a7df2a02cfd144592d9aa9c871b59258c55`).

A full set of Python packages is specified in `requirements.txt`.

The code was only tested with an NVIDIA 80GB A100 GPU.

# Example usage
```python
from ChipGPT import *
model_name: str = "huggyllama/llama-30b"
model_size: str = '30B'


# Train on record summarization
bob_training_set: List[Dict] = load_training_dataset("bob_training_samples", tokenizer=tokenizer)
summarizations: AutoComputedShelfDB = AutoComputedShelfDB('summaries_' + model_size, bob_summarize0)

trainer_var: Trainer = trainer(output_dir_base_name='bob_training_' + model_size, learning_rate=1e-4, dataset=bob_training_set, validation_split=0, model=lora_model_init(lora_r=8, fp16=True), fp16=True, early_stop_patience=50)
with torch.autocast("cuda"):
    trainer_var.train(resume_from_checkpoint=False)


# Train on record QA
barb_training_set: List[List[Dict]] = barb_training_directory_to_dataset('barb_training_samples')
flattened_barb: List[Dict] = [item for sublist in barb_training_set for item in sublist]

model = load_lora_checkpoint(lora_r=8, lora_alpha=1.0, lora_bin_path='bob_training_' + model_size + '_out/checkpoint-epoch-2.0/adapter_model.bin', fp16=True, do_freeze=True)

trainer_var: Trainer = train_barb(model=model, validation_split=0, output_dir_base='barb_bob_combined_' + model_size, fp16=True, micro_batch_size=1, lora_r=8, weight_decay=0.05, lora_dropout=0.5)
with torch.autocast("cuda"):
    trainer_var.train(resume_from_checkpoint=False)


# Perform record QA
process_all_files_in_directory(model=model, input_directory_path='test', output_directory_path='out_' + model_size, high_perplexity_directory='out_hp_' + model_size)
```

# Credits
- *Hugging Face Transformers library*: [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)
- *Hugging Face PEFT library* (for model fine-tuning):  [https://github.com/huggingface/peft](https://github.com/huggingface/peft)

Portions of the code are inspired and/or copied with modifications from PEFT and from:

- *Alpaca-LoRA*: [https://github.com/tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)
- *Stanford Alpaca*: [https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)

All the above projects are released under the Apache License, Version 2.0
