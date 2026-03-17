# Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
from collections import Counter
import torch
import numpy as np
import uuid
from .reward_score.ttrl_math import extract_answer, simplify_expression_string, grade
import re

def adjust_batch(batch,n_gpus_per_node):
    remainder=len(batch)%n_gpus_per_node
    if remainder!=0:
        batch=batch[:-remainder]
    else:
        batch=batch
    return batch


def grade_answer_ttrl(model_answer, gt_answer, fast=False):
    """
    Grade answer with multi-step logic to handle different ground truth types.
    """
    if model_answer is None:
        return False

    is_correct = False
    if isinstance(gt_answer, float) or isinstance(gt_answer, int):
        gt_answer = str(gt_answer)
    if isinstance(gt_answer, str):
        is_correct = grade(model_answer, gt_answer, fast)
    elif isinstance(gt_answer, list):
        is_correct = False
        for gt in gt_answer:
            is_correct |= grade(model_answer, gt, fast)
    return is_correct


def batch_majority_vote(model_outputs: List[str], n: int) -> tuple[List[str], List[float], List[dict[str, dict[str, any]]]]:
    """
    Used to generate the ground truth for TTRL.
    Input:
        model_outputs: list of str
        n: int
    Output:
        majority_gt_list: list of str
        majority_ratio_list: list of float
        answer_ratios_list: list of dict[str, dict[str, any]] where each dict maps answer to a nested dict with 'ratio' and 'answer_anchor_uid'
    """
    majority_gt_list = []
    majority_ratio_list = []
    model_answers_list = []
    answer_ratios_list = []
    assert len(model_outputs) % n == 0
    n_prompts = len(model_outputs) // n
    for i in range(n_prompts):
        prompt_outputs = model_outputs[i * n:(i + 1) * n]
        prompt_majority_gt, prompt_majority_ratio, model_answers, answer_ratios = majority_vote(prompt_outputs)
        majority_gt_list.append(prompt_majority_gt)
        majority_ratio_list.append(prompt_majority_ratio)
        answer_ratios_list.append(answer_ratios)
        model_answers_list.append(model_answers)

    return majority_gt_list, majority_ratio_list, model_answers_list, answer_ratios_list


def majority_vote(model_outputs: List[str]) -> tuple[str, float, List[str], dict[str, dict[str, any]]]:
    assert len(model_outputs) > 0
    model_answers = [extract_answer(generated_text) for generated_text in model_outputs]
    model_answers = [simplify_expression_string(answer) for answer in model_answers]
    model_answers_not_none = [answer for answer in model_answers if answer is not None]
    if len(model_answers_not_none) == 0:
        return "None", 0.0, [None] * len(model_outputs), {}

    counter = Counter(model_answers_not_none)

    majority_answer, majority_count = counter.most_common(1)[0]
    majority_ratio = majority_count / len(model_answers_not_none) if len(model_answers_not_none) > 0 else 0.0

    # Create answer ratio dictionary with nested structure
    answer_ratios = {
        answer: {
            "ratio": count / len(model_answers_not_none) if len(model_answers_not_none) > 0 else 0.0,
            "count": count,
            "answer_anchor_uid": str(uuid.uuid4())
        }
        for answer, count in counter.items()
    }

    # Sort the dictionary by ratio in descending order
    answer_ratios = dict(sorted(answer_ratios.items(), key=lambda x: x[1]["ratio"], reverse=True))

    return majority_answer, majority_ratio, model_answers,answer_ratios


# === Metrics Computation ===
def compute_generator_metrics(batch):
    """Compute generator metrics including reward_accuracy, label_accuracy, format_correct_rate, used_to_update_policy_rate, majority_voting_reward and ground_truth_reward.

    Compute metrics for level=1 (first generation) samples.

    reward_accuracy is true when: is_correct_with_majority_gt and is_correct_with_original_gt are both true or both false.
    label_accuracy: ratio of majority_gt matching original_gt (using all samples).
    format_correct_rate: ratio of format_correct being true (using all samples).
    used_to_update_policy_rate: ratio of used_to_update_policy being true (using all samples).
    majority_voting_reward: ratio of extracted_generator_answer matching majority_gt (using all samples).
    ground_truth_reward: ratio of extracted_generator_answer matching generator_gt (using all samples).
    Only samples with used_to_update_policy=True participate in reward_accuracy calculation.

    Args:
        batch: DataProto batch containing generator nodes data (level=1 only)

    Returns:
        dict: Contains all generator metrics
    """
    
    is_correct_with_majority_gt = batch.non_tensor_batch.get("is_generator_correct_with_majority_gt", [])
    is_correct_with_original_gt = batch.non_tensor_batch.get("is_generator_correct_with_original_gt", [])
    used_to_update_policy = batch.non_tensor_batch.get("used_to_update_policy", [])
    format_correct = batch.non_tensor_batch.get("format_correct", [])
    
    total_samples = len(is_correct_with_majority_gt)
    majority_gt = batch.non_tensor_batch.get("majority_gt", [])
    generator_gt = batch.non_tensor_batch.get("generator_gt", [])

    majority_ratio_list = batch.non_tensor_batch.get("majority_ratio", [])

    filtered_local_indices = [i for i, used in enumerate(used_to_update_policy) if used]
    total_filtered = len(filtered_local_indices)

    majority_voting_reward_count = sum(1 for is_correct in is_correct_with_majority_gt if is_correct)
    majority_voting_reward = majority_voting_reward_count / total_samples if total_samples > 0 else 0.0

    ground_truth_reward_count = sum(1 for is_correct in is_correct_with_original_gt if is_correct)
    ground_truth_reward = ground_truth_reward_count / total_samples if total_samples > 0 else 0.0

    correct_count = sum(1 for i in filtered_local_indices
                             if is_correct_with_majority_gt[i] == is_correct_with_original_gt[i])
    
    reward_accuracy = correct_count / total_filtered if total_filtered > 0 else 0.0

    label_accuracy_count = sum(1 for i in filtered_local_indices if grade_answer_ttrl(majority_gt[i], generator_gt[i]))
    label_accuracy = label_accuracy_count / total_filtered if total_filtered > 0 else 0.0

    format_correct_count = sum(1 for fc in format_correct if fc)
    format_correct_rate = format_correct_count / total_samples if total_samples > 0 else 0.0

    used_to_update_policy_count = sum(1 for used in used_to_update_policy if used)
    used_to_update_policy_rate = used_to_update_policy_count / total_samples if total_samples > 0 else 0.0

    majority_ratio = sum(majority_ratio_list) / total_samples if total_samples > 0 else 0.0

    # Compute majority_ratio for samples with correct labels
    correct_label_majority_ratios = []
    incorrect_label_majority_ratios = []

    for i in range(total_samples):
        if grade_answer_ttrl(majority_gt[i], generator_gt[i]):
            correct_label_majority_ratios.append(majority_ratio_list[i])
        else:
            incorrect_label_majority_ratios.append(majority_ratio_list[i])

    correct_label_majority_ratio = sum(correct_label_majority_ratios) / len(correct_label_majority_ratios) if len(correct_label_majority_ratios) > 0 else 0.0
    incorrect_label_majority_ratio = sum(incorrect_label_majority_ratios) / len(incorrect_label_majority_ratios) if len(incorrect_label_majority_ratios) > 0 else 0.0

    return {
        "label_accuracy": label_accuracy,
        "reward_accuracy": reward_accuracy,
        "majority_ratio": majority_ratio,
        "correct_label_majority_ratio": correct_label_majority_ratio,
        "incorrect_label_majority_ratio": incorrect_label_majority_ratio,
        "majority_voting_reward": majority_voting_reward,
        "ground_truth_reward": ground_truth_reward,
        "format_correct_rate": format_correct_rate,
        "used_to_update_policy_rate": used_to_update_policy_rate,
        "count": total_samples
    }


def compute_verifier_metrics(batch):
    """Compute verifier metrics: TP, FP, TN, FN, recall, precision, accuracy, format_correct_rate, used_to_update_policy_rate, reward_accuracy

    Args:
        batch: DataProto batch containing verifier nodes data
    """
    tp = fp = tn = fn = 0
    # breakpoint()
    assert 'verifier_indicator' in batch.non_tensor_batch, "verifier_indicator must be in batch.non_tensor_batch"
    verifier_indicators = batch.non_tensor_batch['verifier_indicator']
    format_correct = batch.non_tensor_batch.get('format_correct', [])
    used_to_update_policy = batch.non_tensor_batch.get('used_to_update_policy', [])
    is_verifier_correct_with_original_gt=batch.non_tensor_batch.get('is_verifier_correct_with_original_gt', [])
    is_verifier_correct_with_majority_gt=batch.non_tensor_batch.get('is_verifier_correct_with_majority_gt', [])

    for indicator in verifier_indicators:
        if indicator == "TP":
            tp += 1
        elif indicator == "FP":
            fp += 1
        elif indicator == "TN":
            tn += 1
        elif indicator == "FN":
            fn += 1

    acc_count = sum(1 for is_correct in is_verifier_correct_with_original_gt if is_correct)

    # Compute metrics
    total = len(verifier_indicators)
    assert acc_count==(tp + tn)
    accuracy = acc_count / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    wrong_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Compute format_correct_rate
    format_correct_count = sum(1 for fc in format_correct if fc)
    verifier_format_correct_rate = format_correct_count / len(format_correct) if len(format_correct) > 0 else 0.0

    # Compute verifier_used_to_update_policy_rate
    used_to_update_policy_count = sum(1 for used in used_to_update_policy if used)
    verifier_used_to_update_policy_rate = used_to_update_policy_count / len(used_to_update_policy) if len(used_to_update_policy) > 0 else 0.0

    # Compute reward_accuracy using samples with used_to_update_policy=True
    # Condition: is_verifier_correct_with_majority_gt and is_verifier_correct_with_original_gt are both true or both false
    filtered_local_indices = [i for i, used in enumerate(used_to_update_policy) if used]
    total_filtered = len(filtered_local_indices)
    reward_accuracy_count = sum(1 for i in filtered_local_indices
                                     if is_verifier_correct_with_majority_gt[i] == is_verifier_correct_with_original_gt[i])
    reward_accuracy = reward_accuracy_count / total_filtered if total_filtered > 0 else 0.0

    metrics = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "wrong_recall": wrong_recall,
        "format_correct_rate": verifier_format_correct_rate,
        "used_to_update_policy_rate": verifier_used_to_update_policy_rate,
        "reward_accuracy": reward_accuracy,
        "count": total
    }

    return metrics


def compute_re_generator_metrics(batch):
    """Compute re_generator metrics including ground_truth_reward, label_accuracy, reward_accuracy, re_generator_format_correct_rate, re_generator_used_to_update_policy_rate

    Args:
        batch: DataProto batch containing re_generator nodes data (level=3 only)

    Returns:
        dict: Contains all re_generator metrics
    """
    # Use pre-computed correctness flags
    is_correct_with_majority_gt = batch.non_tensor_batch.get("is_generator_correct_with_majority_gt", [])
    is_correct_with_original_gt = batch.non_tensor_batch.get("is_generator_correct_with_original_gt", [])
    used_to_update_policy = batch.non_tensor_batch.get("used_to_update_policy", [])
    format_correct = batch.non_tensor_batch.get("format_correct", [])

    total_samples = len(is_correct_with_majority_gt)

    if total_samples == 0:
        return {
            "ground_truth_reward": 0.0,
            "label_accuracy": 0.0,
            "reward_accuracy": 0.0,
            "format_correct_rate": 0.0,
            "used_to_update_policy_rate": 0.0,
            "count": 0
        }

    # Get ground truth information
    majority_gt = batch.non_tensor_batch.get("majority_gt", [])
    generator_gt = batch.non_tensor_batch.get("generator_gt", [])

    # Compute ground_truth_reward: ratio of extracted_generator_answer matching generator_gt
    ground_truth_reward_count = sum(1 for is_correct in is_correct_with_original_gt if is_correct)
    ground_truth_reward = ground_truth_reward_count / total_samples if total_samples > 0 else 0.0

    # Compute label_accuracy: ratio of majority_gt matching generator_gt (only computing samples with used_to_update_policy=True)
    filtered_local_indices = [i for i, used in enumerate(used_to_update_policy) if used]
    total_filtered = len(filtered_local_indices)
    label_accuracy_count = sum(1 for i in filtered_local_indices if grade_answer_ttrl(majority_gt[i], generator_gt[i]))
    label_accuracy = label_accuracy_count / total_filtered if total_filtered > 0 else 0.0

    # Compute reward_accuracy: only for samples with used_to_update_policy=True
    # Condition: is_correct_with_majority_gt and is_correct_with_original_gt are both true or both false
    filtered_local_indices = [i for i, used in enumerate(used_to_update_policy) if used]
    total_filtered = len(filtered_local_indices)

    reward_accuracy_count = sum(1 for i in filtered_local_indices
                                     if is_correct_with_majority_gt[i] == is_correct_with_original_gt[i])
    reward_accuracy = reward_accuracy_count / total_filtered if total_filtered > 0 else 0.0

    # Compute re_generator_format_correct_rate: ratio of format_correct being true
    format_correct_count = sum(1 for fc in format_correct if fc)
    re_generator_format_correct_rate = format_correct_count / total_samples if total_samples > 0 else 0.0

    # Compute re_generator_used_to_update_policy_rate: ratio of used_to_update_policy being true
    used_to_update_policy_count = sum(1 for used in used_to_update_policy if used)
    re_generator_used_to_update_policy_rate = used_to_update_policy_count / total_samples if total_samples > 0 else 0.0

    return {
        "ground_truth_reward": ground_truth_reward,
        "label_accuracy": label_accuracy,
        "reward_accuracy": reward_accuracy,
        "format_correct_rate": re_generator_format_correct_rate,
        "used_to_update_policy_rate": re_generator_used_to_update_policy_rate,
        "count": total_samples
    }

def compute_all_reward_accuracy_metrics(batch):
    """
    Compute unified reward_accuracy for all samples (generate, verify, regenerate).

    reward_accuracy condition: For each sample type, the corresponding is_correct_with_majority_gt and is_correct_with_original_gt
    are both true or both false. Only samples with used_to_update_policy=True participate in the calculation.

    Args:
        batch: DataProto batch containing all types of samples (generator, verifier, re_generator)

    Returns:
        dict: Contains unified reward_accuracy metrics
    """
    levels = batch.non_tensor_batch.get('level', [])
    used_to_update_policy = batch.non_tensor_batch.get('used_to_update_policy', [])

    # Collect correctness information for all samples participating in calculation
    correct_with_majority_gt = []
    correct_with_original_gt = []

    for i, level in enumerate(levels):
        if not used_to_update_policy[i]:
            continue  # Only compute for samples with used_to_update_policy=True

        if level == 1 or level == 3:  # generator or re_generator
            is_correct_with_majority_gt = batch.non_tensor_batch.get("is_generator_correct_with_majority_gt", [])
            is_correct_with_original_gt = batch.non_tensor_batch.get("is_generator_correct_with_original_gt", [])
        elif level == 2:  # verifier
            is_correct_with_majority_gt = batch.non_tensor_batch.get("is_verifier_correct_with_majority_gt", [])
            is_correct_with_original_gt = batch.non_tensor_batch.get("is_verifier_correct_with_original_gt", [])
        else:
            continue

        if i < len(is_correct_with_majority_gt) and i < len(is_correct_with_original_gt):
            correct_with_majority_gt.append(is_correct_with_majority_gt[i])
            correct_with_original_gt.append(is_correct_with_original_gt[i])

    # Compute reward_accuracy: correct_with_majority_gt and correct_with_original_gt are both true or both false
    total_filtered = len(correct_with_majority_gt)
    if total_filtered == 0:
        reward_accuracy = 0.0
    else:
        correct_count = sum(1 for maj_correct, orig_correct in zip(correct_with_majority_gt, correct_with_original_gt)
                           if maj_correct == orig_correct)
        reward_accuracy = correct_count / total_filtered

    return {
        "reward_accuracy": reward_accuracy,
        "count": total_filtered
    }


def compute_cover_rl_metrics(batch, num_turns=2):
    """Compute all cover_rl metrics including generator metrics, verifier metrics and re_generator metrics

    Args:
        batch: DataProto batch containing rollout data
        num_turns: number of rounds (1 for generator only, 2 for generator+verifier)
        metrics: dict to update with metrics. If None, returns a new dict

    Returns:
        dict: Contains all cover_rl metrics (if metrics is None) or None (if metrics is provided)
    """
    cover_rl_metrics = {}
    # Compute generator metrics
    levels = batch.non_tensor_batch.get('level', [])
    generator_indices = [i for i, level in enumerate(levels) if level == 1]
    generator_batch = batch[generator_indices]

    # Compute generator metrics
    generator_metrics = compute_generator_metrics(generator_batch)
    for key, value in generator_metrics.items():
        cover_rl_metrics.update({f"train_generator/{key}": value})

    # Compute verifier metrics (if there are verifier nodes, num_turns=2)
    if num_turns == 2:
        verifier_indices = [i for i, level in enumerate(levels) if level == 2]
        if len(verifier_indices) > 0:
            verifier_batch = batch[verifier_indices]
            verifier_metrics = compute_verifier_metrics(verifier_batch)
            for key, value in verifier_metrics.items():
                cover_rl_metrics.update({f"train_verifier/{key}": value})

        # Compute re_generator metrics (if there are re_generator nodes, level=3)
        re_generator_indices = [i for i, level in enumerate(levels) if level == 3]
        if len(re_generator_indices) > 0:
            re_generator_batch = batch[re_generator_indices]
            re_generator_metrics = compute_re_generator_metrics(re_generator_batch)
            for key, value in re_generator_metrics.items():
                cover_rl_metrics.update({f"train_regenerator/{key}": value})

        # Compute overall reward_accuracy across all samples
        all_reward_accuracy_metrics = compute_all_reward_accuracy_metrics(batch)
        for key, value in all_reward_accuracy_metrics.items():
            cover_rl_metrics.update({f"train_overall/{key}": value})

    return cover_rl_metrics
