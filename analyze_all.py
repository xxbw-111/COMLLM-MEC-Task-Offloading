import pandas as pd
import json
import os
import re
import numpy as np

# --- 1. 配置 ---

#这是存放所有评测结果的统一目录
RESULTS_DIR = "设置为你的输出目录"

#所有模型/策略的名称 (可设置为你需要对比的模型）
MODEL_NAMES = [
    "sft_model_1.5B",
    "grpo_model_1.5B",
    "sft_model_7B",
    "grpo_model_7B",
    "comllm_model_7B",
    "dqn_model",
    "random_policy",
]

# 用于在最终表格中显示的美化名称
PRETTY_NAMES = {
    "sft_model_1.5B": "SFT 1.5B",
    "sft_model_7B": "SFT 7B",
    "grpo_model_7B": "GRPO 7B",
    "comllm_model_7B": "COMLLM 7B",
    "random_policy":"Random",
    "dqn_model":"DQN",
    "grpo_model_1.5B":"GRPO 1.5B",
}

# --- 2. 辅助函数 ---

def remove_nulls(obj):
    """
    递归移除字典或列表中值为 None (null) 的项。
    用于在分析前清洗数据，避免 Pandas 处理 null 时出错或数据杂乱。
    """
    if isinstance(obj, dict):
        return {
            k: remove_nulls(v)
            for k, v in obj.items()
            if v is not None
        }
    elif isinstance(obj, list):
        return [remove_nulls(v) for v in obj if v is not None]
    else:
        return obj


def parse_model_prediction(raw_output: str):
    """
    从模型的原始输出中稳定地解析出 action_id 和 action_name。
    """
    if not isinstance(raw_output, str):
        return None

    clean_output = raw_output.lower().strip()

    # 1. 优先判断本地执行
    if "本地" in clean_output or "local" in clean_output:
        return {"action_id": 0, "action_name": "本地执行"}

    # 2. 使用正则动态提取服务器编号
    match = re.search(r"(服务器|server)\s*(\d+)", clean_output)

    if match:
        try:
            server_id = int(match.group(2))
            # 加一个简单的范围校验，如1-20
            if 1 <= server_id <= 20:
                return {
                    "action_id": server_id,
                    "action_name": f"卸载到边缘服务器{server_id}"
                }
        except ValueError:
            pass

    return None


# --- 3. 主分析逻辑 ---

def main():
    summary_data = []
    theoretical_best_avg_cost = float('nan')

    print("--- Starting Comprehensive Advanced MEC Metric Analysis (With Auto-Cleaning) ---")

    # --- 步骤 1: 预先计算理论最优基准 ---
    temp_dfs = []
    for model_name in MODEL_NAMES:
        file_path = os.path.join(RESULTS_DIR, f"{model_name}_results.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)

                # 加载时立即清洗数据
                clean_data = remove_nulls(raw_data)

                if clean_data:
                    temp_dfs.append(pd.DataFrame(clean_data))
            except json.JSONDecodeError:
                print(f"警告: 无法解析 JSON 文件: {file_path}")

    if temp_dfs:
        # 合并所有数据计算全局最优平均值（假设所有模型测试的是同一批数据，去重）
        # 注意：因为 input 是字典，不能直接作为 drop_duplicates 的 subset，这里简化处理
        # 只要有一个模型的数据包含 optimal_cost 即可
        full_df = pd.concat(temp_dfs, ignore_index=True)
        if 'optimal_cost' in full_df.columns:
            # 过滤掉无效的 optimal_cost
            valid_optimal = full_df['optimal_cost'].replace([np.inf, -np.inf], np.nan).dropna()
            theoretical_best_avg_cost = valid_optimal.mean()

    # --- 步骤 2: 循环计算每个模型的指标 ---
    for model_name in MODEL_NAMES:
        file_path = os.path.join(RESULTS_DIR, f"{model_name}_results.json")
        if not os.path.exists(file_path):
            print(f"警告: 结果文件未找到，跳过 '{model_name}'")
            continue

        print(f"Analyzing results for: {model_name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 清洗数据
        data = remove_nulls(raw_data)

        if not data:
            print(f"警告：结果文件为空，跳过 '{model_name}'")
            continue

        df = pd.DataFrame(data)

        # 过滤掉成本为无穷大的样本 (通常是解析错误)
        valid_df = df[df['cost'] != float('inf')].copy()
        if valid_df.empty:
            print(f"警告：'{model_name}' 的所有样本都无法计算有效成本。")
            continue

        # --- a. 平均延迟 (ATCT) ---
        avg_delay = valid_df['cost'].mean()

        # --- c. 任务丢弃率 (Task Droppage Rate) ---
        try:
            # 这里的 input 已经是清洗过的字典
            # valid_df['max_delay'] = valid_df['input'].apply(
            #     lambda x: float(re.findall(r"[-+]?\d*\.\d+|\d+", str(x.get("最大延迟", "10")))[0])
            # )
            valid_df['max_delay'] = 12.5  #过滤的阈值
            task_droppage_rate = (valid_df['cost'] > valid_df['max_delay']).mean() * 100
        except (KeyError, IndexError, TypeError, ValueError):
            task_droppage_rate = float('nan')

        # --- d. 卸载率 (Offloading Rate) ---
        valid_df['action_id'] = valid_df['model_decision'].apply(
            lambda x: parse_model_prediction(x)['action_id'] if parse_model_prediction(x) else -1
        )
        offloading_rate = (valid_df['action_id'] > 0).mean() * 100

        # --- e. 负载均衡指数 (Jain's Fairness Index) ---
        # 核心逻辑：统计整个测试集中，各服务器被选中的次数分布

        # 1. 筛选出所有边缘卸载的决策 (action_id > 0)
        offload_actions = valid_df[valid_df['action_id'] > 0]['action_id'].tolist()

        if len(offload_actions) > 0:
            # 2. 统计每个服务器被选中的次数
            # 假设服务器 ID 范围是 1 到 20 (根据你的场景调整，或者动态获取)
            # 需要统计 server_1 到 server_MAX 的选中次数

            # 动态获取最大的服务器 ID 以确定 n
            max_server_id = max(offload_actions) if offload_actions else 0

            # 创建一个计数列表 x，x[i] 表示第 i+1 号服务器被选中的次数
            # 注意：如果某个服务器一次都没被选，它的计数就是 0
            server_counts = [0] * max_server_id
            for action in offload_actions:
                if 1 <= action <= max_server_id:
                    server_counts[action - 1] += 1

            n = len(server_counts)
            sum_x = sum(server_counts)
            sum_x_sq = sum([x ** 2 for x in server_counts])

            if n > 0 and sum_x_sq > 0:
                jain_index = (sum_x ** 2) / (n * sum_x_sq)
                load_balancing_index = jain_index * 100  # 转为百分比
            else:
                load_balancing_index = 0.0
        else:
            # 如果没有发生任何卸载，负载均衡无从谈起
            load_balancing_index = float('nan')

        # --- g. 性能达成率 ---
        if not np.isnan(theoretical_best_avg_cost) and avg_delay > 0:
            perf_achievement_rate = (theoretical_best_avg_cost / avg_delay) * 100
        else:
            perf_achievement_rate = float('nan')

        summary_data.append({
            "模型": PRETTY_NAMES.get(model_name, model_name),
            "平均延迟": avg_delay,
            "性能达成率 (%)": perf_achievement_rate,
            # "卸载正确率 (%)": avg_accuracy,
            "任务丢弃率 (%)": task_droppage_rate,
            # "卸载率 (%)": offloading_rate,
            "负载均衡指数 (%)": load_balancing_index,
        })

    if not summary_data:
        print("\n没有找到任何有效的结果文件，无法生成对比报告。")
        return

    # --- 4. 打印最终表格 ---
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(by="平均延迟", ascending=True).reset_index(drop=True)

    for col in ["性能达成率 (%)", "任务丢弃率 (%)",  "负载均衡指数 (%)"]:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].map('{:.2f}'.format)

    if "平均延迟" in summary_df.columns:
        summary_df["平均延迟"] = summary_df["平均延迟"].map('{:.4f}'.format)

    print("\n" + "=" * 120)
    print("===== 综合 MEC 性能指标对比报告 (已自动清洗无效数据) =====".center(120))
    print("=" * 120)
    print(summary_df.to_string(index=False))
    print("=" * 120)

    if not np.isnan(theoretical_best_avg_cost):
        print(f"\n* 理论最优基准 (平均延迟): {theoretical_best_avg_cost:.4f} 时间槽")


if __name__ == "__main__":
    main()