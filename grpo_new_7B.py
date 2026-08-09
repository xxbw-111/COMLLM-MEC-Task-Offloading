"""GRPO training entry point without LACS future-impact reward shaping."""


# --sft-model-path	SFT 后的模型路径。GRPO 不是从原始基座模型开始，而是在 SFT 模型基础上继续训练，所以这个参数必须传。
# --dataset-path	GRPO 训练数据路径，比如 ./data/grpo_train.json。
# --output-dir	GRPO 训练结果保存目录，默认 ./outputs/grpo。最终 adapter 会保存在这个目录下的 final_grpo_adapter。
# --max-seq-length	最大序列长度，默认 2048。超过这个长度的训练样本会被过滤。
# --epochs	训练轮数，默认 1。
# --learning-rate	学习率，默认 5e-6。
# --beta	GRPO 里的 KL 惩罚系数，默认 0.005。它控制模型不要偏离原 SFT 模型太远。
# --num-generations	每个 prompt 采样几个回答，也就是 GRPO 的 group size，默认 8。
# --generation-batch-size	生成回答时的 batch size，默认 8。显存不够时可以调小。
# --trust-remote-code	是否允许 Hugging Face 加载模型仓库里的自定义代码。加了这个参数就是 True，不加就是 False。

import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
import os
from functools import partial

try:
    from cost_calculator import calculate_cost
    from mec_utils import extract_input_data_from_prompt, format_mec_input, parse_model_prediction
except ImportError as e:
    print(f"Error: {e}. Make sure all helper scripts are in the same directory.")
    exit()


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRPO training for MEC task offloading.")
    parser.add_argument("--sft-model-path", required=True, help="Merged SFT model path used as the GRPO base.")
    parser.add_argument("--dataset-path", required=True, help="GRPO JSON dataset path.")
    parser.add_argument("--output-dir", default="./outputs/grpo", help="Directory for GRPO checkpoints.")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.005, help="KL coefficient used by GRPO.")
    parser.add_argument("--num-generations", type=int, default=8, help="GRPO group size.")
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable Hugging Face trust_remote_code.")
    return parser.parse_args()


# ============================
# 1. 奖励函数 (Reward Function)
# ============================
def mec_reward_function(prompts: list, completions: list, **kwargs) -> list[float]:
    """
    融合了“任务奖励”和“格式奖励”的奖励函数。
    """
    rewards = []

    # --- 权重配置 ---
    TASK_REWARD_WEIGHT = 0.8
    FORMAT_REWARD_WEIGHT = 0.2
    # TASK_REWARD_SCALE_FACTOR = 10.0

    for i, completion_item in enumerate(completions):
        # 从 prompt 中解析出原始的 input_data
        current_prompt = prompts[i]
        input_data = extract_input_data_from_prompt(current_prompt)

        if not input_data:
            rewards.append(-10.0)  # 惩罚无法解析的情况
            continue

        completion_str = ""
        if isinstance(completion_item, str):
            completion_str = completion_item
        elif isinstance(completion_item, list) and len(completion_item) > 0 and isinstance(completion_item[0], dict):
            completion_str = completion_item[0].get('content', '')
        else:
            completion_str = str(completion_item)

        format_reward = calculate_format_reward(completion_str)

        task_reward = 0.0
        model_decision = parse_model_prediction(completion_str)

        if model_decision is None:
            task_reward = -10.0
        else:
            task_reward += 1.0
            action_id = model_decision['action_id']
            # 使用从 prompt 解析出的 input_data 计算成本
            cost = calculate_cost(action_id, input_data)
            task_reward += -cost
            if cost > 10.0:
                task_reward -= 10.0

        # scaled_task_reward = task_reward / TASK_REWARD_SCALE_FACTOR
        # final_reward = (scaled_task_reward * TASK_REWARD_WEIGHT) + (format_reward * FORMAT_REWARD_WEIGHT)
        final_reward = (task_reward * TASK_REWARD_WEIGHT) + (format_reward * FORMAT_REWARD_WEIGHT)
        rewards.append(final_reward)

    return rewards


def calculate_format_reward(completion_str: str) -> float:
    """
    计算单个模型输出字符串的格式奖励。
    """
    if not isinstance(completion_str, str):
        return -2.0

    clean_str = completion_str.strip()
    if not clean_str:
        return -2.0

    reward = 0.5

    # 惩罚词保持不变
    penalty_keywords = ["json", "{", "}", "```", "action_id", "action_name"]
    for keyword in penalty_keywords:
        if keyword in clean_str.lower():
            reward -= 0.5

    # 长度限制保持不变
    ideal_min_len, ideal_max_len = 4, 30
    if not (ideal_min_len <= len(clean_str) <= ideal_max_len):
        length_penalty = abs(len(clean_str) - (ideal_min_len + ideal_max_len) / 2) / 10.0
        reward -= length_penalty

    valid_server_nums = [str(i) for i in range(1, 21)]
    positive_keywords = ["本地", "边缘", "local", "server"] + valid_server_nums

    if any(kw in clean_str.lower() for kw in positive_keywords):
        reward += 1.0
    else:
        reward -= 2.0

    return reward


# ============================
# 2. 数据加载与预处理
# ============================
def format_for_grpo_and_analyze(sample, tokenizer, max_seq_length):
    instruction = sample['instruction']

    input_str = format_mec_input(sample['input'])

    system_prompt = (
        "你是一个MEC任务卸载专家。请严格按照用户的指示分析输入数据，输出卸载策略。"
        "不要包含任何解释、代码块标记或额外文本。"
    )
    user_prompt = f"{instruction}\n\n**输入数据**：\n{input_str}"

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    completion_text = sample['output']['action_name']

    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False).input_ids
    completion_tokens = tokenizer(completion_text, add_special_tokens=False).input_ids
    prompt_length = len(prompt_tokens)
    completion_length = len(completion_tokens)
    total_length = prompt_length + completion_length

    return {
        "prompt": prompt_text,
        "completion": completion_text,
        "original_sample": sample,
        "is_truncated": total_length > max_seq_length,
        "prompt_length": prompt_length,
        "completion_length": completion_length,
        "total_length": total_length,
    }


# ============================
# 3. 主函数
# ============================
def main():
    args = parse_args()

    # --- 加载SFT全量模型和分词器 ---
    print(f"--- Step 1: Loading Merged SFT Model from: {args.sft_model_path} ---")
    if not os.path.exists(args.sft_model_path):
        print(f"FATAL ERROR: Merged SFT model path not found: {args.sft_model_path}")
        exit()

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Merged SFT model loaded successfully.")

    print("\n--- Step 2: Adding a new LoRA adapter for GRPO training ---")
    lora_config = LoraConfig(
        r=64, lora_alpha=128, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model.config.use_cache = False
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n--- Step 3: Loading and Preparing Dataset ---")
    raw_dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    formatting_function = partial(format_for_grpo_and_analyze, tokenizer=tokenizer, max_seq_length=args.max_seq_length)
    processed_dataset = raw_dataset.map(formatting_function) # 此时保留所有列用于分析和预览

    train_dataset_full = processed_dataset.filter(lambda x: not x['is_truncated'])
    if len(train_dataset_full) == 0:
        print("FATAL ERROR: All samples were filtered out.")
        exit()
    print(f"Dataset prepared. Total valid samples before cleaning: {len(train_dataset_full)}")
    # <--- 在这里添加新的打印 --->
    print("=" * 50)
    print(f"FINAL VERIFICATION: The length of the dataset being passed to the trainer is EXACTLY: {len(train_dataset_full)}")
    print("=" * 50)
    # # ==============================================================================
    # # 添加打印逻辑
    # # ==============================================================================
    # print("\n" + "=" * 20 + " DATA SAMPLE PREVIEW " + "=" * 20)
    # num_samples_to_preview = 2  # 想看的数量
    # for i in range(min(num_samples_to_preview, len(train_dataset_full))):
    #     sample = train_dataset_full[i]
    #     print(f"\n--- Sample {i + 1} ---")

    #     prompt_preview = (sample['prompt'][:300] + '...') if len(sample['prompt']) > 300 else sample['prompt']

    #     print(f"  Prompt (Preview): {repr(prompt_preview)}")
    #     print(f"  Completion (Expert Answer): '{sample['completion']}'")
    #     print(f"  - Prompt Length:     {sample['prompt_length']} tokens")
    #     print(f"  - Completion Length: {sample['completion_length']} tokens")
    #     print(f"  - Total Length:      {sample['total_length']} tokens")
    #     print(f"  - Exceeds Max Length: {'Yes' if sample['is_truncated'] else 'No'}")
    #     # 你也可以打印 original_sample 的内容来检查
    #     print(f"  Original Input Keys: {list(sample['original_sample']['input'].keys())}")

    # print("=" * 63 + "\n")
    # # ==============================================================================
    # # 打印逻辑结束
    # # ==============================================================================

    #创建最终只包含 'prompt' 和 'completion' 的数据集用于训练
    columns_to_keep = ["prompt", "completion"]
    columns_to_remove = [col for col in train_dataset_full.column_names if col not in columns_to_keep]

    train_dataset = train_dataset_full.remove_columns(columns_to_remove)

    print("=" * 50)
    print(f"FINAL VERIFICATION: The length of the dataset being passed to the trainer is EXACTLY: {len(train_dataset)}")
    print(f"Final dataset columns for trainer: {train_dataset.column_names}")
    print("=" * 50)

    # --- 配置GRPO训练 ---
    print("\n--- Step 4: Configuring GRPOTrainer ---")
    grpo_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_generations=args.num_generations,
        generation_batch_size=args.generation_batch_size,

        max_prompt_length=args.max_seq_length - 100,
        max_completion_length=15,
        logging_steps=1,
        save_strategy="epoch",
        # report_to="tensorboard",
        remove_unused_columns=False,
        bf16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        # warmup_ratio=0.01,
        temperature=1.3,
        top_p=0.95,
        top_k=None,         # 禁用 top_k 过滤
    )

    # --- 创建并开始训练 ---
    print("\n--- Step 5: Starting GRPO Training ---")
    grpo_trainer = GRPOTrainer(
        model=model,
        args=grpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=[mec_reward_function],
    )
    grpo_trainer.train()

    # --- 保存最终模型 ---
    print("\n--- Step 6: Saving Final GRPO Adapter ---")
    final_model_path = os.path.join(args.output_dir, "final_grpo_adapter")
    model.save_pretrained(final_model_path)
    print(f"GRPO fine-tuned adapter saved to {final_model_path}")


if __name__ == "__main__":
    main()
