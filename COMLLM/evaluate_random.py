import json
import os
import random
import re
import argparse
from tqdm import tqdm
from datasets import load_dataset

try:
    from cost_calculator import calculate_cost
    from mec_utils import parse_model_prediction, remove_nulls
except ImportError as e:
    print(f"错误: 无法导入 'cost_calculator'. 请确保 cost_calculator.py 文件在同一个目录下。")
    print(f"详细信息: {e}")
    exit()


# --test-data-path	测试集路径。比如 ./data/test.json。随机策略会在这个测试集上随机选择动作，然后计算成本。
# --results-dir	结果保存目录。默认是 ./results。
# --seed	随机种子。默认 42。

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a random MEC offloading policy.")
    parser.add_argument("--test-data-path", required=True, help="JSON test dataset path.")
    parser.add_argument("--results-dir", default="./results", help="Directory used to save random baseline results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible baseline evaluation.")
    return parser.parse_args()


# --- 3. 主执行函数 ---
def run_random_policy_evaluation(test_data_path, results_dir, seed):
    """在测试集上运行随机策略并保存结果（支持动态服务器数量）"""
    print("\n===== Evaluating Policy: random_policy =====")
    random.seed(seed)

    if not os.path.exists(test_data_path):
        print(f"错误: 测试文件未找到: {test_data_path}")
        exit()

    test_dataset = load_dataset("json", data_files=test_data_path, split="train")
    results = []

    for sample in tqdm(test_dataset, desc="Inferencing with random_policy"):

        # --- 核心修改：先清洗数据，移除 None ---
        # 这样 input_data.keys() 里就只会剩下真实存在的服务器 Key
        input_data = remove_nulls(sample['input'])

        # --- 1. 动态确定当前样本的服务器数量 ---
        # 使用正则找出最大的服务器 ID
        # 兼容 "边缘服务器X" 和 "serverX"
        available_actions = [0]
        for key in input_data.keys():
            # 确保只匹配键名 (排除值)
            key_str = str(key)
            match = re.search(r"(边缘服务器|server)\s*(\d+)", key_str)

            # 排除 "状态" 后缀的键，防止重复计数 (虽然 id 是一样的)
            if match and "状态" not in key_str:
                available_actions.append(int(match.group(2)))

        # 动作空间：0 (本地) + 当前样本中真实存在的服务器 ID。
        action_id = random.choice(sorted(set(available_actions)))

        if action_id == 0:
            model_decision_name = "本地执行"
        else:
            model_decision_name = f"卸载到边缘服务器{action_id}"

        # --- 3. 结果记录 ---
        if 'output' not in sample or 'action_name' not in sample['output']:
            ground_truth_decision_name = None
        else:
            ground_truth_decision_name = sample['output']['action_name']

        model_decision = {"action_id": action_id, "action_name": model_decision_name}
        ground_truth_decision = parse_model_prediction(ground_truth_decision_name)

        # 此时 input_data 已经是干净的，传给 calculate_cost 很安全
        cost = calculate_cost(model_decision['action_id'], input_data) if model_decision else float('inf')

        # 注意：如果 ground_truth 为空，这里给个 inf 或者 None
        optimal_cost = calculate_cost(ground_truth_decision['action_id'], input_data) if ground_truth_decision else float('inf')

        is_optimal = False
        if model_decision and ground_truth_decision:
            is_optimal = (model_decision['action_id'] == ground_truth_decision['action_id'])

        results.append({
            "model_name": "random_policy",
            "input": input_data,  # 保存清洗后的 input
            "model_output": f"Randomly chose: {model_decision_name}",
            "model_decision": model_decision_name,
            "ground_truth": ground_truth_decision_name,
            "cost": cost,
            "optimal_cost": optimal_cost,
            "is_optimal": is_optimal
        })

    # --- 保存结果文件 ---
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    output_file = os.path.join(results_dir, "random_policy_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nRandom policy evaluation complete.")
    print(f"Results for {len(results)} samples saved to {output_file}")


if __name__ == "__main__":
    cli_args = parse_args()
    run_random_policy_evaluation(cli_args.test_data_path, cli_args.results_dir, cli_args.seed)
