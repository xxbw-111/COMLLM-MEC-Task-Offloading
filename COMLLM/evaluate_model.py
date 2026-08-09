# 
# --test-data-path	测试集 JSON 路径，比如 ./data/test.json。
# --results-dir	可选	评测结果保存目录，默认是 ./results。
# --model-config	可选	多模型评测配置文件路径。里面可以一次写多个模型，比如 SFT、GRPO、COMLLM。
# --model-name	可选	单模型评测时，这个模型在结果文件里的名字，默认叫 model。
# --base-model-path	单模型时需要	单个模型的基础模型路径。
# --adapter-path	可选	LoRA adapter 路径。如果评测的是 LoRA 模型，就填；如果评测的是完整合并模型，就不用填。
# --torch-dtype	可选	加载模型时用的数据类型，只能选 float16 或 bfloat16，默认 bfloat16。
# --trust-remote-code	可选开关	是否允许 Hugging Face 执行模型仓库里的自定义代码。加了就是 True，不加就是 False。


# 有两种用法。
# 用法 1：评测单个模型
# python evaluate_model.py \
#   --test-data-path ./data/test.json \
#   --results-dir ./results/default \
#   --model-name comllm_model_7B \
#   --base-model-path ./outputs/qwen2.5-7b-sft-merged \
#   --adapter-path ./outputs/qwen2.5-7b-comllm/final_grpo_adapter \
#   --trust-remote-code
# 这会生成：
# ./results/default/comllm_model_7B_results.json
# 用法 2：一次评测多个模型
# 先写一个 model_configs.json：
# {
#   "sft_model_7B": {
#     "base_model_path": "./outputs/qwen2.5-7b-sft-merged",
#     "adapter_path": null
#   },
#   "comllm_model_7B": {
#     "base_model_path": "./outputs/qwen2.5-7b-sft-merged",
#     "adapter_path": "./outputs/qwen2.5-7b-comllm/final_grpo_adapter"
#   }
# }
# 然后运行：
# python evaluate_model.py \
#   --test-data-path ./data/test.json \
#   --results-dir ./results/default \
#   --model-config ./model_configs.json \
#   --trust-remote-code
# 如果用了 --model-config，就不需要再传 --model-name、--base-model-path、--adapter-path。



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel 
import argparse
import json
import os
from tqdm import tqdm

try:
    from cost_calculator import calculate_cost
    from mec_utils import format_mec_input, parse_model_prediction

except ImportError as e:
    print(f"Error importing helper scripts: {e}.")
    exit()

# 推理时使用的生成参数
GENERATION_CONFIG = {
    "max_new_tokens": 20,
    "do_sample": True,
    "temperature": 0.7,  # 使用一个较低的温度以获得稳定的结果
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one or more MEC offloading LLMs.")
    parser.add_argument("--test-data-path", required=True, help="JSON test dataset path.")
    parser.add_argument("--results-dir", default="./results", help="Directory used to save result JSON files.")
    parser.add_argument(
        "--model-config",
        help=(
            "Optional JSON file describing multiple models. Format: "
            "{\"model_name\": {\"base_model_path\": \"...\", \"adapter_path\": null}}"
        ),
    )
    parser.add_argument("--model-name", default="model", help="Name used when evaluating a single model.")
    parser.add_argument("--base-model-path", help="Base model path for single-model evaluation.")
    parser.add_argument("--adapter-path", default=None, help="Optional LoRA adapter path for single-model evaluation.")
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Dtype used when loading the model.",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable Hugging Face trust_remote_code.")
    return parser.parse_args()


def load_models_to_evaluate(args):
    """Load model definitions from a config file or a single-model CLI input."""
    if args.model_config:
        with open(args.model_config, "r", encoding="utf-8") as f:
            return json.load(f)
    if not args.base_model_path:
        raise ValueError("Please provide --base-model-path or --model-config.")
    return {
        args.model_name: {
            "base_model_path": args.base_model_path,
            "adapter_path": args.adapter_path,
        }
    }


# --- 2. 推理函数定义 ---

def format_simple_prompt(sample, tokenizer):
    """为所有模型构建统一的、简单的 prompt"""
    instruction = sample['instruction']
    input_str = format_mec_input(sample['input'])
    system_prompt = (
        "你是一个MEC任务卸载专家。请严格按照用户的指示分析输入数据，输出卸载策略。"
        "不要包含任何解释、代码块标记或额外文本。"
    )
    user_prompt = f"{instruction}\n\n**输入数据**：\n{input_str}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# --- 2. 推理函数定义 ---
def run_inference(model_name, model_info, test_data_path, results_dir, dtype, trust_remote_code):
    """
    加载一个模型（可能带 LoRA）并在测试集上运行完整的推理流程。
    """
    base_model_path = model_info["base_model_path"]
    adapter_path = model_info.get("adapter_path")  # 使用 .get() 更安全

    print(f"\n===== Evaluating Model: {model_name} =====")
    print(f"  - Base Model: {base_model_path}")
    if adapter_path:
        print(f"  - LoRA Adapter: {adapter_path}")

    # 统一使用 float16 加载，不再进行 4-bit 量化
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 【新增】如果提供了 adapter_path，则动态加载 LoRA 适配器
    if adapter_path:
        if not os.path.exists(adapter_path):
            print(f"警告：找不到 LoRA adapter 路径，将只评估基础模型: {adapter_path}")
        else:
            print("Applying LoRA adapter...")
            model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()  # 切换到评估模式
    # ================================================================

    test_dataset = load_dataset("json", data_files=test_data_path, split="train")
    results = []

    for sample in tqdm(test_dataset, desc=f"Inferencing with {model_name}"):
        # 推理逻辑完全统一，因为 LoRA 是透明的
        prompt_text = format_simple_prompt(sample, tokenizer)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():  # 推理时禁用梯度计算
            outputs = model.generate(**inputs, **GENERATION_CONFIG)

        model_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

        # --- 结果记录 ---
        if 'output' not in sample or 'action_name' not in sample['output']:
            print(f"警告：测试样本缺少 'output' 或 'action_name' 字段作为 ground truth，跳过此样本。")
            continue
        ground_truth_decision_name = sample['output']['action_name']

        model_decision = parse_model_prediction(model_output)
        ground_truth_decision = parse_model_prediction(ground_truth_decision_name)

        cost = calculate_cost(model_decision['action_id'], sample['input']) if model_decision else float('inf')
        optimal_cost = calculate_cost(ground_truth_decision['action_id'], sample['input']) if ground_truth_decision else float('inf')

        results.append({
            "model_name": model_name,
            "input": sample['input'],
            "model_output": model_output,
            "model_decision": model_decision['action_name'] if model_decision else "Parse Error",
            "ground_truth": ground_truth_decision_name,
            "cost": cost,
            "optimal_cost": optimal_cost,
            "is_optimal": (model_decision['action_id'] == ground_truth_decision['action_id']) if model_decision and ground_truth_decision else False
        })


    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    output_file = os.path.join(results_dir, f"{model_name}_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results for {model_name} saved to {output_file}")

    del model
    torch.cuda.empty_cache()


# --- 3. 主执行流程 ---
if __name__ == "__main__":
    cli_args = parse_args()
    torch_dtype = torch.float16 if cli_args.torch_dtype == "float16" else torch.bfloat16
    models_to_evaluate = load_models_to_evaluate(cli_args)

    for name, model_info in models_to_evaluate.items():
        # 跳过路径为空的模型
        if not model_info.get("base_model_path"):
            print(f"\n===== Skipping Model: {name} (path is empty) =====")
            continue
        run_inference(
            name,
            model_info,
            cli_args.test_data_path,
            cli_args.results_dir,
            torch_dtype,
            cli_args.trust_remote_code,
        )
    print("\nAll end-to-end evaluations complete!")

