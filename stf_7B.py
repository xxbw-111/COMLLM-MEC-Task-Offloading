"""Supervised fine-tuning entry point for MEC task offloading."""

# --base-model-name	基础模型路径或 Hugging Face 模型名。比如 Qwen/Qwen2.5-7B 或本地模型目录。必须传。
# --dataset-path	SFT 训练数据路径。这里的数据应该是 oracle 标注好的数据，也就是每个样本都有最优 action_name。必须传。
# --output-dir	训练输出目录，默认 ./outputs/sft。checkpoint 和最终 LoRA adapter 会保存在这里。
# --max-length	最大 token 长度，默认 2048。超过长度可能会被截断或影响训练。
# --epochs	训练轮数，默认 1。
# --batch-size	每张 GPU 上的 batch size，默认 2。显存不够可以调小。
# --gradient-accumulation-steps	梯度累积步数，默认 4。用于在显存较小时模拟更大的 batch。
# --learning-rate	学习率，默认 1e-5。
# --trust-remote-code	是否允许 Hugging Face 执行模型仓库里的自定义代码。加这个参数就是 True，不加就是 False。


import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig,prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import os
from mec_utils import format_mec_input


def parse_args():
    parser = argparse.ArgumentParser(description="Run SFT for MEC task offloading.")
    parser.add_argument("--base-model-name", required=True, help="Base Hugging Face model name or local path.")
    parser.add_argument("--dataset-path", required=True, help="Oracle-labeled SFT JSON dataset.")
    parser.add_argument("--output-dir", default="./outputs/sft", help="Directory for checkpoints and final adapter.")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device train batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable Hugging Face trust_remote_code.")
    return parser.parse_args()


def format_sft_prompt(sample, tokenizer):
    """Serialize one MEC sample into a chat-style SFT training string."""
    instruction = sample['instruction']
    input_str = format_mec_input(sample['input'])
    target_output = sample['output']['action_name']
    messages = [
        {"role": "system", "content": "你是一个MEC任务卸载专家。请严格按照用户的指示分析输入数据，输出卸载策略。不要包含任何解释、代码块标记或额外文本。"},
        {"role": "user", "content": f"{instruction}\n\n**输入数据**：\n{input_str}"},
        {"role": "assistant", "content": target_output}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def main():
    args = parse_args()

    # --- 2. 加载分词器 ---
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=args.trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- 3. 加载数据集---
    print(f"Loading and formatting dataset from {args.dataset_path}...")
    raw_dataset = load_dataset("json", data_files=args.dataset_path, split="train")
    formatted_dataset = raw_dataset.map(
        lambda sample: format_sft_prompt(sample, tokenizer),
        remove_columns=list(raw_dataset.features)
    )
    print("Dataset prepared.")

    # --- 4. 配置模型加载 ---
    # 4-bit 量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {args.base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
        # 【新增】如果你的硬件和库支持，可以尝试启用 Flash Attention 2
        # attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    # 【可选】梯度检查点默认关闭以追求速度。如果后续OOM，再取消这行注释
    model.gradient_checkpointing_enable()

    # 为 k-bit 训练准备模型，处理兼容性问题
    model = prepare_model_for_kbit_training(model)

    # --- 5. 配置 LoRA (为 4090 优化) ---
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )


    # --- 6. 配置训练参数  ---
    training_arguments = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_32bit", 
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=1,
        fp16=False,
        bf16=True,  
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        group_by_length=True,
        dataset_text_field="text",
        max_length=args.max_length,
        packing=False,
    )

    # --- 7. 创建并开始训练---
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_arguments,
    )

    print("Starting SFT training for the 7B model on RTX 4090...")
    trainer.train()

    # --- 8. 保存最终模型 ---
    print("Training complete. Saving final LoRA adapter...")
    final_model_path = os.path.join(args.output_dir, "final_lora_adapter")
    trainer.save_model(final_model_path)
    print(f"Fine-tuned LoRA adapter saved to {final_model_path}")


if __name__ == "__main__":
    main()
