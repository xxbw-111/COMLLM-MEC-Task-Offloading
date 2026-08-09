"""Cost functions for MEC task offloading decisions.
"""

import re
import copy
import random


DEFAULT_CYCLES_PER_BIT = 297.0
DEFAULT_TIME_SLOT_DURATION = 0.1
INVALID_ACTION_COST = 5.0


def _safe_extract(pattern, text, default=0.0):
    """Extract the first numeric regex group from text and fall back safely."""
    if not text:
        return default
    match = re.search(pattern, str(text))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return default
    return default


def _find_server_by_id(servers, server_id):
    """Return the parsed server whose real id equals server_id."""
    return next((server for server in servers if server.get("id") == server_id), None)

def parse_input_string(input_data):
    """
    解析原始输入数据字典，提取数值参数。
    适配格式：服务器的静态参数、状态和历史负载均合并在一个字符串中。
    """
    parsed = {}
    if not isinstance(input_data, dict):
        return parsed

    try:
        # 1. 解析全局任务参数
        parsed['task_size'] = _safe_extract(r'([\d\.]+)', input_data.get("任务大小", "0"))
        parsed['max_delay'] = _safe_extract(r'([\d\.]+)', input_data.get("最大延迟", "0"))
        parsed['drop_penalty'] = _safe_extract(r'([\d\.]+)', input_data.get("丢弃惩罚", "100"), default=100.0)
        parsed['local_wait'] = _safe_extract(r'([\d\.]+)', input_data.get("本地CPU队列等待时间", "0"))
        parsed['net_wait'] = _safe_extract(r'([\d\.]+)', input_data.get("网络队列等待时间", "0"))
        parsed['trans_rate'] = _safe_extract(r'([\d\.]+)', input_data.get("网络上传速率", "0"))
        parsed['time_slot_duration'] = _safe_extract(
            r'([\d\.]+)',
            input_data.get("时间槽时长", str(DEFAULT_TIME_SLOT_DURATION)),
            default=DEFAULT_TIME_SLOT_DURATION,
        )

        # 2. 解析本地设备参数
        local_device_str = input_data.get("本地设备", "")
        parsed['local_cpu_freq'] = _safe_extract(r'CPU频率\s*([\d\.]+)', local_device_str)
        # 默认 297.0，如果数据里有则覆盖
        parsed['cycles_per_bit'] = _safe_extract(
            r'每比特周期数\s*([\d\.]+)',
            local_device_str,
            default=DEFAULT_CYCLES_PER_BIT,
        )

        # 3. 动态解析边缘服务器列表
        server_keys = [key for key in input_data.keys() if re.match(r'边缘服务器\d+$', key)]

        # 提取真实服务器 ID 并排序。后续按 id 查找，不依赖列表下标。
        server_ids = sorted([int(re.search(r'(\d+)', k).group(1)) for k in server_keys])

        parsed['servers'] = []

        for i in server_ids:
            full_info_str = input_data.get(f"边缘服务器{i}", "")

            if not full_info_str or "None" in full_info_str:
                continue

            # 在合并的长字符串中提取各项参数
            active_tasks = int(_safe_extract(r'活跃任务数[：:]?\s*(\d+)', full_info_str))
            queue_len = _safe_extract(r'服务器总队列长度[：:]?\s*([\d\.]+)', full_info_str)
            cpu_freq = _safe_extract(r'CPU频率[：:]?\s*([\d\.]+)', full_info_str)

            parsed['servers'].append({
                "id": i,
                "active_tasks": active_tasks,
                "queue_len": queue_len,
                "cpu_freq": cpu_freq
            })

    except Exception as e:
        print(f"[Parse Error] 解析输入数据时发生异常: {e}")
        return {}

    return parsed


def predict_multiturn_impact(action_id, input_data, lookahead_steps=3):
    """
    严谨计算：如果当前选择了 action_id，未来 lookahead_steps 个任务的平均最优延迟是多少。

    逻辑：
    1. 状态转移：根据 action_id 更新本地或特定服务器的队列状态。
    2. 蒙特卡洛模拟：生成未来任务，计算其在更新后状态下的最小延迟。
    """
    # 1. 解析参数
    params = parse_input_string(input_data)
    if not params: return 0.0

    # 物理常数
    comp_density = params.get('cycles_per_bit', DEFAULT_CYCLES_PER_BIT) / 1000.0
    time_slot_duration = params.get('time_slot_duration', DEFAULT_TIME_SLOT_DURATION)
    if comp_density <= 0: return 100.0  # 异常保护

    # 2. 状态克隆与更新 (State Transition)
    # 我们不仅要更新服务器，如果选了本地，本地队列也会变长！
    simulated_servers = copy.deepcopy(params.get('servers', []))
    simulated_local_wait = params.get('local_wait', 0.0)
    current_task_size = params.get('task_size', 0.0)

    # --- 模拟当前决策对环境的改变 ---
    if action_id == 0:
        # 决策：本地执行
        # 影响：本地 CPU 变得更忙了。
        # 计算本地处理速率用于估算新增排队时间
        local_cap = params.get('local_cpu_freq', 0) * time_slot_duration
        local_rate = local_cap / comp_density if local_cap > 0 else 0.1
        # 本地等待时间增加 = 当前任务处理时间
        simulated_local_wait += (current_task_size / local_rate)

    elif action_id > 0:
        # 决策：卸载到服务器
        target_srv = _find_server_by_id(simulated_servers, action_id)
        if target_srv is not None:
            # 影响：目标服务器队列变长，活跃任务数增加
            target_srv['queue_len'] += current_task_size
            target_srv['active_tasks'] += 1  # 占用一个并发槽位

    # 3. 前向采样 (Collaborative Simulation)
    total_future_min_cost = 0.0

    for _ in range(lookahead_steps):
        # 3.1 生成未来任务
        future_task_size = random.uniform(current_task_size * 0.5, current_task_size * 1.5)

        # 3.2 寻找未来任务的最优解 (Oracle Policy)
        # 需要遍历所有可能的去向，看哪个延迟最低

        # --- Option A: 未来任务在本地跑 ---
        local_cap = params.get('local_cpu_freq', 0) * time_slot_duration
        local_rate = local_cap / comp_density if local_cap > 0 else 0.001

        cost_if_local = 100.0
        if local_rate > 0:
            cost_if_local = simulated_local_wait + (future_task_size / local_rate)

        # --- Option B: 未来任务在各个边缘服务器跑 ---
        costs_if_edge = []

        wait_trans = params.get('net_wait', 0)
        trans_rate = params.get('trans_rate', 0)
        trans_time = (future_task_size / trans_rate) if trans_rate > 0 else 100.0

        for srv in simulated_servers:
            # 计算该服务器的共享速率
            # 未来任务加入后，活跃数还要再 +1
            predicted_active = srv['active_tasks'] + 1

            srv_cap = srv['cpu_freq'] * time_slot_duration
            shared_rate = (srv_cap / predicted_active) / comp_density if predicted_active > 0 else 0.001

            if shared_rate > 0:
                t_wait = srv['queue_len'] / shared_rate
                t_exec = future_task_size / shared_rate
                total_edge = wait_trans + trans_time + t_wait + t_exec
                costs_if_edge.append(total_edge)
            else:
                costs_if_edge.append(100.0)

        # 3.3 决策：选最小的那个
        min_edge_cost = min(costs_if_edge) if costs_if_edge else 100.0
        best_future_cost = min(cost_if_local, min_edge_cost)

        total_future_min_cost += best_future_cost

    # 4. 返回平均值
    return total_future_min_cost / lookahead_steps



def calculate_cost(action_id, input_data):
    """
    计算 MEC 卸载任务的总延迟（成本）。

    Args:
        action_id (int): 0 表示本地，1~N 表示卸载到对应服务器。
        input_data (dict): 原始输入数据字典。

    Returns:
        float: 总延迟时间（时间槽）。如果出错或动作无效，返回 float('inf')。
    """
    # 基础检查
    if action_id is None:
        return float('inf')

    # 确保 action_id 是整数
    try:
        action_id = int(action_id)
    except ValueError:
        return float('inf')

    if action_id < 0:
        return float('inf')

    try:
        # 解析数据
        params = parse_input_string(input_data)
        if not params:
            return float('inf')

        # --- 物理常数与单位修正 ---
        comp_density = params.get('cycles_per_bit', DEFAULT_CYCLES_PER_BIT) / 1000.0
        time_slot_duration = params.get('time_slot_duration', DEFAULT_TIME_SLOT_DURATION)

        # 防止除以零
        if comp_density <= 0: return float('inf')

        # === 情况 1: 本地执行 (Action 0) ===
        if action_id == 0:
            wait_time = params.get('local_wait', 0)

            # 本地计算能力 
            local_comp_power = params.get('local_cpu_freq', 0) * time_slot_duration
            local_proc_rate = local_comp_power / comp_density

            if local_proc_rate <= 0: return float('inf')

            exec_time = params.get('task_size', 0) / local_proc_rate
            return wait_time + exec_time

        # === 情况 2: 边缘卸载 (Action 1..N) ===
        else:
            servers = params.get('servers', [])

            # 按真实服务器 ID 查找，避免“服务器编号”和“列表下标”混淆。
            server = _find_server_by_id(servers, action_id)
            if server is None:
                return INVALID_ACTION_COST

            # 1. 传输延迟
            wait_time_trans = params.get('net_wait', 0)
            trans_rate = params.get('trans_rate', 0)

            if trans_rate <= 0: return float('inf')
            trans_time = params.get('task_size', 0) / trans_rate

            # 2. 服务器计算延迟
            # 预测加入新任务后的活跃数
            predicted_active_tasks = server.get('active_tasks', 0) + 1

            server_comp_power = server.get('cpu_freq', 0) * time_slot_duration

            # 共享后的处理能力
            shared_proc_rate = (server_comp_power / predicted_active_tasks) / comp_density

            if shared_proc_rate <= 0: return float('inf')

            wait_time_server = server.get('queue_len', 0) / shared_proc_rate
            exec_time_server = params.get('task_size', 0) / shared_proc_rate

            total_latency = wait_time_trans + trans_time + wait_time_server + exec_time_server
            return total_latency

    except Exception as e:
        # print(f"[Calc Error] 计算成本时出错 (Action {action_id}): {e}")
        return float('inf')
