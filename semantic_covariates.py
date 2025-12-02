# semantic_covariates.py
import torch
import numpy as np
import re
from typing import List, Dict, Deque
from collections import deque, Counter
import math


class SemanticCovariateSystem:
    """
    8维精简版语义协变量系统
    保留最核心的指令质量、语义对齐、探索状态和反馈信号
    """

    def __init__(self, max_history: int = 50):
        self.max_history = max_history

        # 历史记录
        self.performance_history = deque(maxlen=max_history)
        self.instruction_history = deque(maxlen=max_history)
        self.score_history = deque(maxlen=max_history)

        # 任务关键词库
        self.task_keywords = {
            'synonyms': ['synonym', 'similar', 'same meaning', 'equivalent', 'word'],
            'antonyms': ['antonym', 'opposite', 'contrary', 'reverse', 'word'],
            'cause_and_effect': ['cause', 'effect', 'reason', 'result', 'because', 'leads to'],
            'common_concept': ['common', 'concept', 'category', 'group', 'shared', 'theme'],
            'diff': ['difference', 'subtract', 'minus', 'gap', 'distance'],
            'sum': ['sum', 'add', 'plus', 'total', 'combine'],
            'default': ['find', 'return', 'identify', 'list', 'generate']
        }

    def compute_8d_covariates(self, instruction: str, performance: float,
                               scores: np.ndarray, step: int, total_steps: int,
                               task_name: str = None) -> torch.Tensor:
        """
        计算8维精简语义协变量
        """
        covariates = torch.zeros(8, dtype=torch.float32)

        # 更新历史记录
        self._update_history(instruction, performance, scores)

        # === 1. 基础质量维度 (2维) ===
        # [0] 长度适宜性：太短(如"t")或太长(废话)都不好
        covariates[0] = float(self._length_appropriateness(instruction))
        # [1] 指令清晰度：结构、标点、是否像句人话
        covariates[1] = float(self._instruction_clarity(instruction))

        # === 2. 语义对齐维度 (2维) ===
        # [2] 任务对齐度：是否包含任务核心词(如 subtract, antonym)
        covariates[2] = float(self._task_alignment(instruction, task_name))
        # [3] 约束适宜性：是否使用了 strong constraints (must, only)
        covariates[3] = float(self._constraint_appropriateness(instruction))

        # === 3. 探索状态维度 (2维) ===
        # [4] 指令新颖性：与历史指令的差异（局部探索）
        covariates[4] = float(self._instruction_novelty(instruction))
        # [5] 探索紧迫度：结合了"当前阶段(step)"和"性能停滞(stagnation)"的综合指标
        covariates[5] = float(self._exploration_need(step, total_steps))

        # === 4. 反馈信号维度 (2维) ===
        # [6] 输出可靠性：根据 Llama-3 打分的分布判断（过滤全0分或极差分布）
        covariates[6] = float(self._output_reliability(scores))
        # [7] 性能一致性：该指令在不同样本上的表现是否稳定
        covariates[7] = float(self._performance_consistency(scores))

        return covariates

    def _update_history(self, instruction: str, performance: float, scores: np.ndarray):
        """更新历史记录"""
        self.instruction_history.append(instruction)
        self.performance_history.append(performance)
        self.score_history.append(scores)

    # === 以下是具体的计算方法 (保持大部分逻辑不变，删除了不用的) ===

    def _length_appropriateness(self, text):
        if not text: return 0.0
        l = len(text.split())
        # 理想长度在 5-30 词之间
        if 5 <= l <= 30: return 1.0
        return max(0.0, 1.0 - abs(l - 17)/20.0)

    def _instruction_clarity(self, instruction: str) -> float:
        """指令清晰度"""
        if not instruction: return 0.0
        score = 0.0
        length = len(instruction)
        if 20 <= length <= 100: score += 0.4
        elif 10 <= length < 20 or 100 < length <= 150: score += 0.2
        else: score += 0.1

        sentences = re.split(r'[.!?]+', instruction)
        valid_sentences = [s.strip() for s in sentences if s.strip()]
        if 1 <= len(valid_sentences) <= 3: score += 0.3
        elif len(valid_sentences) == 0: score += 0.1
        else: score += 0.2

        if instruction.strip().endswith(('.', '?')): score += 0.3
        elif instruction.strip().endswith(','): score += 0.1
        return min(1.0, score)

    def _task_alignment(self, instruction: str, task_name: str) -> float:
        """任务对齐度"""
        if not instruction: return 0.0
        if not task_name: return 0.5
        
        keywords = self.task_keywords.get(task_name, self.task_keywords['default'])
        instruction_lower = instruction.lower()
        
        # 关键词匹配
        matches = sum(1 for keyword in keywords if keyword in instruction_lower)
        match_ratio = matches / max(1, len(keywords))
        
        # 动作动词检测
        action_verbs = ['return', 'find', 'identify', 'list', 'generate', 'create', 'write', 'output']
        has_action = any(verb in instruction_lower for verb in action_verbs)
        
        alignment = match_ratio * 0.7 + (0.3 if has_action else 0.1)
        return min(1.0, alignment)

    def _constraint_appropriateness(self, instruction: str) -> float:
        if not instruction: return 0.0
        strong = ['must', 'only', 'never', 'always', 'exactly', 'directly']
        moderate = ['should', 'avoid', 'prefer', 'try to', 'please']
        
        s_count = sum(1 for c in strong if c in instruction.lower())
        m_count = sum(1 for c in moderate if c in instruction.lower())
        
        val = s_count * 0.5 + m_count * 0.3
        return min(1.0, val * 0.7)

    def _instruction_novelty(self, instruction: str) -> float:
        """与历史指令的差异度 (1 - max_similarity)"""
        if not instruction: return 0.0
        if len(self.instruction_history) < 2: return 1.0
        
        recent = list(self.instruction_history)[-10:] # 看最近10个
        max_sim = 0.0
        for prev in recent:
            if prev == instruction: continue
            sim = self._cosine_similarity(instruction, prev)
            max_sim = max(max_sim, sim)
            
        return max(0.1, 1.0 - max_sim)

    def _exploration_need(self, step: int, total_steps: int) -> float:
        """探索需求：前期高，后期低；若性能停滞则升高"""
        base_need = 1.0 - (step / max(1, total_steps))
        
        # 检查性能停滞
        if len(self.performance_history) >= 5:
            recent = list(self.performance_history)[-5:]
            if max(recent) - min(recent) < 0.01: # 停滞
                base_need += 0.3
        
        return min(1.0, max(0.0, base_need))

    def _output_reliability(self, scores: np.ndarray) -> float:
        """输出可靠性：分数分布是否合理"""
        if len(scores) == 0: return 0.0
        if len(scores) < 2: return 0.5
        
        # 如果全是 0 或全是 1，可能不可靠（或者任务太简单/太难）
        mean_score = np.mean(scores)
        if mean_score < 0.01: return 0.1 # 全错，可能是指令太烂
        
        # 标准差适中比较好
        std = np.std(scores)
        return max(0.1, 1.0 - abs(std - 0.2)) # 假设 0.2 左右的区分度是健康的

    def _performance_consistency(self, scores: np.ndarray) -> float:
        """性能一致性"""
        if len(scores) < 2: return 0.5
        std = np.std(scores)
        # 越低越一致
        consistency = 1.0 - min(1.0, std * 2.0)
        return max(0.1, consistency)

    def _cosine_similarity(self, a: str, b: str) -> float:
        if not a or not b: return 0.0
        def get_bow(text):
            words = re.findall(r'\w+', text.lower())
            return Counter(words)
        vec_a, vec_b = get_bow(a), get_bow(b)
        intersection = set(vec_a.keys()) & set(vec_b.keys())
        dot = sum(vec_a[w] * vec_b[w] for w in intersection)
        norm_a = math.sqrt(sum(c**2 for c in vec_a.values()))
        norm_b = math.sqrt(sum(c**2 for c in vec_b.values()))
        return 0.0 if norm_a == 0 or norm_b == 0 else dot / (norm_a * norm_b)
