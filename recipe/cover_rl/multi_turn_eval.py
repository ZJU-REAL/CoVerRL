import json
import uuid
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np
import torch
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from .utils.prompt import Initial_Prompt, Generator_Prompt, Verifier_Prompt, GPQA_Initial_Prompt, GPQA_Generator_Prompt
from verl.utils.reward_score.math_verify import compute_score
from .utils.reward_score.ttrl_math import extract_answer
from .utils.gpqa import extract_gpqa_answer

@dataclass
class EvalNode:
    """
    Base Node class for evaluation, containing common attributes and methods for all nodes.
    """
    uid_to_node: Dict[str, 'EvalNode'] = field(default_factory=dict)

    def __init__(self, uid: str, raw_question: str, ground_truth: str, data_source: str, level: int, parent_uid: Optional[str] = None,is_mask: bool = False):
        self.uid = uid
        self.data_source = data_source
        self.raw_question = raw_question
        self.ground_truth = ground_truth
        self.level = level
        self.parent_uid = parent_uid
        self.type = "raw_question"
        self.response = None
        self.conversation_history = None
        self.format_correct = None
        self.is_finished = False
        self.is_mask=is_mask
        
        self.save_dict = dict()

    def construct_conversation_history(self):
        if self.level == 0:
            # Select different initial prompts based on data source
            if self.data_source == "GPQA-TTT":
                prompt = GPQA_Initial_Prompt
            else:
                prompt = Initial_Prompt
            conversation_history = [
                {"content": self.raw_question+"\n"+prompt, "role": "user"}
            ]
        else:
            parent_node = EvalNode.uid_to_node.get(self.parent_uid)
            conversation_history = parent_node.conversation_history.copy()
            if self.type == "generator":
                prompt = Verifier_Prompt
            else:
                if self.data_source == "GPQA-TTT":
                    prompt = GPQA_Generator_Prompt
                else:
                    prompt = Generator_Prompt
            conversation_history.extend([
                {"content": self.response, "role": "assistant"},
                {"content": prompt, "role": "user"}
            ])

        self.conversation_history = conversation_history
        return conversation_history

    def save_DataProto_as_dict(self, batch, index):
        """Save data from DataProto to node's save_dict"""
        tensors = batch.batch
        non_tensors = batch.non_tensor_batch
        for key, val in tensors.items():
            self.save_dict[key] = val[index]
        for key, val in non_tensors.items():
            self.save_dict[key] = val[index]


class EvalGenerator(EvalNode):
    """
    Generator Node class for evaluation, inherits from EvalNode.
    """

    def __init__(self, uid: str, raw_question: str, ground_truth: str, data_source: str, level: int, parent_uid: Optional[str] = None,is_mask: bool = False):
        super().__init__(uid, raw_question, ground_truth, data_source, level, parent_uid,is_mask)

        self.type = "generator"

        self.extracted_generator_answer = None
        self.is_generator_correct = None

    def add_generator_info(self, response: str, extracted_answer: str):
        """Add information to generator node"""
        self.response = response
        self.extracted_generator_answer = extracted_answer
        if self.data_source == "GPQA-TTT":
            if self.extracted_generator_answer is None:
                self.format_correct = False
                self.is_generator_correct=False
            else:
                self.format_correct = True
                self.is_generator_correct=(self.extracted_generator_answer==self.ground_truth)
        else:
            self.is_generator_correct=compute_score(self.response,self.ground_truth)
            if self.extracted_generator_answer is None:
                self.format_correct = False
            else:
                self.format_correct = True


class EvalVerifier(EvalNode):
    """
    Verifier Node class for evaluation, inherits from EvalNode.
    """

    def __init__(self, uid: str, raw_question: str, ground_truth: str, data_source: str, level: int, parent_uid: Optional[str] = None,is_mask: bool = False):
        super().__init__(uid, raw_question, ground_truth, data_source, level, parent_uid,is_mask)

        self.type = "verifier"

        self.extracted_verifier_answer = None
        self.verifier_indicator = None

    def add_verifier_info(self, response: str, extracted_answer: str, format_correct: bool = True):
        """Add information to verifier node"""
        self.response = response
        self.extracted_verifier_answer = extracted_answer
        self.format_correct = format_correct

        parent_node = EvalNode.uid_to_node.get(self.parent_uid)
        if parent_node.is_generator_correct:
            self.verifier_gt = "correct"
        else:
            self.verifier_gt = "wrong"

        # Compute confusion matrix indicators
        if self.extracted_verifier_answer == self.verifier_gt:
            if self.verifier_gt == "correct":
                self.verifier_indicator = "TP"
            else:
                self.verifier_indicator = "TN"
        else:
            if self.verifier_gt == "correct":
                self.verifier_indicator = "FN"
            else:
                self.verifier_indicator = "FP"


class MultiTurnEvaluator:
    """
    Multi-turn evaluator for executing multi-round generate-verify loop evaluation
    """

    def __init__(self, config: Dict[str, Any], tokenizer, actor_rollout_wg):
        self.config = config
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg

        cover_rl_config = config.get("cover_rl", {})
        self.eval_max_turns = cover_rl_config.get("eval_max_turns", 1)

        self.eval_n = config.get("actor_rollout_ref", {}).get("rollout", {}).get("val_kwargs", {}).get("n", 1)

        self.enable_thinking = cover_rl_config.get("enable_thinking", False)
        
        EvalNode.uid_to_node = {}

    def _generate_responses(self, nodes: List[EvalNode], rollout_n: int) -> Tuple[DataProto, List[str]]:
        batch = self.preprocess_batch(nodes)
        batch=batch.repeat(repeat_times=rollout_n, interleave=True)
        batch_input = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"],
                            non_tensor_batch_keys=["raw_prompt_ids"])
        batch_input.meta_info["validate"] = True         
            
        batch_input.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": True,
            "validate": True,
        }

        batch_output = self.actor_rollout_wg.generate_sequences(batch_input)
        batch = batch.union(batch_output)                        

        responses = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

        return batch, responses

    def _create_child_nodes(self, parent_nodes: List[EvalNode], batch: DataProto, responses: List[str],
                           node_type: str, rollout_n: int = 1) -> List[EvalNode]:
        child_nodes = []

        for i, parent_node in enumerate(parent_nodes):
            for j in range(rollout_n):
                uid = str(uuid.uuid4())
                batch_idx = i * rollout_n + j

                if node_type == "generator":
                    child_node = EvalGenerator(
                        uid=uid,
                        raw_question=parent_node.raw_question,
                        ground_truth=parent_node.ground_truth,
                        data_source=parent_node.data_source,
                        level=parent_node.level + 1,
                        parent_uid=parent_node.uid,
                        is_mask=parent_node.is_mask
                    )
                    # Add generator info
                    extracted_answer = self.extract_generator_answer(responses[batch_idx], parent_node.data_source)
                    child_node.add_generator_info(responses[batch_idx], extracted_answer)

                elif node_type == "verifier":
                    child_node = EvalVerifier(
                        uid=uid,
                        raw_question=parent_node.raw_question,
                        ground_truth=parent_node.ground_truth,
                        data_source=parent_node.data_source,
                        level=parent_node.level + 1,
                        parent_uid=parent_node.uid,
                        is_mask=parent_node.is_mask
                    )
                    # Add verifier info
                    extracted_answer, format_correct = self.extract_verifier_answer_and_format(responses[batch_idx])
                    child_node.add_verifier_info(responses[batch_idx], extracted_answer, format_correct)

                child_node.save_DataProto_as_dict(batch, batch_idx)
                EvalNode.uid_to_node[uid] = child_node
                child_nodes.append(child_node)

        return child_nodes

    def _create_root_nodes(self, batch: DataProto) -> List[EvalNode]:
        """Create root nodes"""
        root_nodes = []
        for i in range(len(batch.batch['input_ids'])):
            uid = str(uuid.uuid4())
            root_node = EvalNode(
                uid=uid,
                data_source=batch.non_tensor_batch['data_source'][i],
                raw_question=batch.non_tensor_batch['extra_info'][i]['question'],
                ground_truth=batch.non_tensor_batch['reward_model'][i]['ground_truth'],
                level=0,
                parent_uid=None,
                is_mask=False
            )
            root_nodes.append(root_node)
            EvalNode.uid_to_node[uid] = root_node
        return root_nodes

    def extract_generator_answer(self, response: str, data_source: str = None):
        """Extract answer from generator response"""
        if data_source == "GPQA-TTT":
            extracted_answer = extract_gpqa_answer(response)
        else:
            extracted_answer = extract_answer(response)

        return extracted_answer


    def _collect_eval_data(self, nodes: List[EvalNode]) -> DataProto:
        """Collect evaluation data"""
        eval_data = []
        for node in nodes:
            eval_data.append(node.save_dict)
        return DataProto.from_single_dict(data=collate_fn(eval_data))

    def perform_multi_turn_evaluation(self, batch: DataProto) -> Dict[str, Any]:

        # Create root nodes
        all_nodes=[]
        root_nodes = self._create_root_nodes(batch)
        all_nodes.extend(root_nodes)

        # Store generator and verifier nodes for each turn
        all_generator_nodes = []
        all_verifier_nodes = []

        # Round 1: Generate initial responses
        print(f"Starting evaluation turn 1/{self.eval_max_turns}")
        gen_batch, responses = self._generate_responses(root_nodes, self.eval_n)
        generator_nodes = self._create_child_nodes(root_nodes, gen_batch, responses, "generator", self.eval_n)
        all_generator_nodes.append(generator_nodes)
        all_nodes.extend(generator_nodes)


        for turn in range(1, self.eval_max_turns):
            print(f"Starting evaluation turn {turn + 1}/{self.eval_max_turns}")

            # 1. Verify current nodes
            ver_batch, ver_responses = self._generate_responses(generator_nodes, 1)
            verifier_nodes = self._create_child_nodes(generator_nodes, ver_batch, ver_responses, "verifier",1)
            all_verifier_nodes.append(verifier_nodes)
            all_nodes.extend(verifier_nodes)

            # 2. Find nodes that failed verification (need to regenerate)
            failed_uids = [v_node.uid for v_node in verifier_nodes
                                if v_node.extracted_verifier_answer == "wrong"]

            if failed_uids:
                # Generate responses for all parent nodes (including successful ones)
                gen_batch, gen_responses = self._generate_responses(verifier_nodes, 1)
                generator_nodes = self._create_child_nodes(verifier_nodes, gen_batch, gen_responses, "generator",1)

                # Mark which nodes are truly need to regenerate (failed verification)
                for _, gen_node in enumerate(generator_nodes):
                    if gen_node.parent_uid not in failed_uids:
                        gen_node.is_mask = True  # Not participate in metric calculation

                all_generator_nodes.append(generator_nodes)
                all_nodes.extend(generator_nodes)
            else:
                # If all nodes passed verification, end evaluation
                break

        # Collect all unique data_sources
        all_data_sources = set(node.data_source for node in all_nodes)

        # Compute evaluation metrics by data_source
        eval_metrics = {}
        for data_source in all_data_sources:
            source_metrics = self._compute_evaluation_metrics(all_generator_nodes, all_verifier_nodes, data_source)
            eval_metrics[data_source] = source_metrics

        return eval_metrics

    def _compute_generator_metrics(self, all_generator_nodes: List[List[EvalGenerator]],
                                  data_source: Optional[str] = None) -> Dict[str, Any]:

        metrics = {}

        # If data_source is specified, filter nodes
        if data_source is not None:
            filtered_generator_nodes = []
            for turn_nodes in all_generator_nodes:
                filtered_turn = [node for node in turn_nodes if node.data_source == data_source]
                filtered_generator_nodes.append(filtered_turn)
            all_generator_nodes = filtered_generator_nodes

        # Compute accuracy up to each round
        # For each round t (from 1 to eval_max_turns), find unmasked nodes starting from round t
        num_turns = len(all_generator_nodes)
        num_samples = len(all_generator_nodes[0]) if all_generator_nodes else 0
        for t in range(1, num_turns + 1):
            # Up to round t, find from round t (index t-1) going backwards
            up_to_turn_nodes = [None] * num_samples

            # Traverse from round t backwards (including round t)
            for turn_idx in range(t - 1, -1, -1):
                turn_nodes = all_generator_nodes[turn_idx]

                # For each sample in this round, if no node is found yet and verification passed, record it
                for sample_idx in range(len(turn_nodes)):
                    if up_to_turn_nodes[sample_idx] is None and not getattr(turn_nodes[sample_idx], 'is_mask', False):
                        up_to_turn_nodes[sample_idx] = turn_nodes[sample_idx]

            # Compute accuracy up to round t
            up_to_turn_correct_count = sum(1 for node in up_to_turn_nodes if node and node.is_generator_correct)
            up_to_turn_total_count = sum(1 for node in up_to_turn_nodes if node is not None)
            up_to_turn_accuracy = up_to_turn_correct_count / up_to_turn_total_count if up_to_turn_total_count > 0 else 0.0

            metrics[f'generator/accuracy_up_to_turn_{t}'] = up_to_turn_accuracy
            metrics[f'generator/correct_count_up_to_turn_{t}'] = up_to_turn_correct_count
            metrics[f'generator/total_count_up_to_turn_{t}'] = up_to_turn_total_count

        return metrics

    def _compute_regenerator_metrics(self, all_generator_nodes: List[List[EvalGenerator]],
                                    data_source: Optional[str] = None) -> Dict[str, Any]:

        metrics = {}

        # If data_source is specified, filter nodes
        if data_source is not None:
            filtered_generator_nodes = []
            for turn_nodes in all_generator_nodes:
                filtered_turn = [node for node in turn_nodes if node.data_source == data_source]
                filtered_generator_nodes.append(filtered_turn)
            all_generator_nodes = filtered_generator_nodes

        # Prepare data: find first round and final round nodes for each sample
        if not all_generator_nodes:
            return metrics
            
        first_turn_nodes = all_generator_nodes[0]
        num_samples = len(first_turn_nodes)
        
        # Find final node and corresponding round for each sample
        final_nodes = [None] * num_samples
        final_turn_idx = [None] * num_samples
        
        # Traverse each round from back to front
        for turn_idx in range(len(all_generator_nodes) - 1, -1, -1):
            turn_nodes = all_generator_nodes[turn_idx]
            
            for sample_idx in range(len(turn_nodes)):
                if final_nodes[sample_idx] is None and not getattr(turn_nodes[sample_idx], 'is_mask', False):
                    final_nodes[sample_idx] = turn_nodes[sample_idx]
                    final_turn_idx[sample_idx] = turn_idx
        
        # Find samples that have regenerated at least once (final turn != first turn)
        regenerate_samples = []
        for sample_idx in range(num_samples):
            if final_turn_idx[sample_idx] is not None and final_turn_idx[sample_idx] > 0:
                first_node = first_turn_nodes[sample_idx]
                final_node = final_nodes[sample_idx]
                regenerate_samples.append({
                    'sample_idx': sample_idx,
                    'first_node': first_node,
                    'final_node': final_node,
                    'first_correct': first_node.is_generator_correct,
                    'final_correct': final_node.is_generator_correct,
                    'first_answer': first_node.extracted_generator_answer,
                    'final_answer': final_node.extracted_generator_answer
                })
        
        # First group of metrics: answer correctness changes
        correct_to_wrong = 0  # correct -> wrong
        wrong_to_correct = 0  # wrong -> correct
        correct_to_correct = 0  # correct -> correct
        wrong_to_wrong = 0  # wrong -> wrong
        
        for sample in regenerate_samples:
            if sample['first_correct'] and not sample['final_correct']:
                correct_to_wrong += 1
            elif not sample['first_correct'] and sample['final_correct']:
                wrong_to_correct += 1
            elif sample['first_correct'] and sample['final_correct']:
                correct_to_correct += 1
            elif not sample['first_correct'] and not sample['final_correct']:
                wrong_to_wrong += 1
        
        total_regenerate = len(regenerate_samples)
        
        metrics['regenerator/total'] = total_regenerate
        metrics['regenerator/correct_to_wrong'] = correct_to_wrong
        metrics['regenerator/wrong_to_correct'] = wrong_to_correct
        metrics['regenerator/correct_to_correct'] = correct_to_correct
        metrics['regenerator/wrong_to_wrong'] = wrong_to_wrong
        
        if total_regenerate > 0:
            metrics['regenerator/correct_to_wrong_ratio'] = correct_to_wrong / total_regenerate
            metrics['regenerator/wrong_to_correct_ratio'] = wrong_to_correct / total_regenerate
            metrics['regenerator/correct_to_correct_ratio'] = correct_to_correct / total_regenerate
            metrics['regenerator/wrong_to_wrong_ratio'] = wrong_to_wrong / total_regenerate
        else:
            metrics['regenerator/correct_to_wrong_ratio'] = 0.0
            metrics['regenerator/wrong_to_correct_ratio'] = 0.0
            metrics['regenerator/correct_to_correct_ratio'] = 0.0
            metrics['regenerator/wrong_to_wrong_ratio'] = 0.0
        
        # Second group of metrics: for wrong->wrong cases, track whether the answer changed
        wrong_to_different_wrong = 0  # wrong -> another wrong
        wrong_to_same_wrong = 0  # wrong -> same wrong
        
        for sample in regenerate_samples:
            if not sample['first_correct'] and not sample['final_correct']:
                # Both answers are wrong, compare if they are the same
                first_ans = sample['first_answer']
                final_ans = sample['final_answer']
                
                # Handle None case
                if first_ans is None and final_ans is None:
                    wrong_to_same_wrong += 1
                elif first_ans != final_ans:
                    wrong_to_different_wrong += 1
                else:
                    wrong_to_same_wrong += 1
        
        metrics['regenerator/wrong_to_wrong_total'] = wrong_to_wrong
        metrics['regenerator/wrong_to_different_wrong'] = wrong_to_different_wrong
        metrics['regenerator/wrong_to_same_wrong'] = wrong_to_same_wrong
        
        if wrong_to_wrong > 0:
            metrics['regenerator/wrong_to_different_wrong_ratio'] = wrong_to_different_wrong / wrong_to_wrong
            metrics['regenerator/wrong_to_same_wrong_ratio'] = wrong_to_same_wrong / wrong_to_wrong
        else:
            metrics['regenerator/wrong_to_different_wrong_ratio'] = 0.0
            metrics['regenerator/wrong_to_same_wrong_ratio'] = 0.0
        
        return metrics

    def _compute_verifier_metrics(self, all_verifier_nodes: List[List[EvalVerifier]],
                                  data_source: Optional[str] = None) -> Dict[str, Any]:

        metrics = {}

        # If data_source is specified, filter nodes
        if data_source is not None:
            filtered_verifier_nodes = []
            for turn_nodes in all_verifier_nodes:
                filtered_turn = [node for node in turn_nodes if node.data_source == data_source]
                filtered_verifier_nodes.append(filtered_turn)
            all_verifier_nodes = filtered_verifier_nodes

        # Compute Verifier metrics, only for nodes without mask
        all_verifier_nodes_flat = [node for turn_nodes in all_verifier_nodes for node in turn_nodes
                                   if not node.is_mask]

        # Compute confusion matrix
        tp = sum(1 for node in all_verifier_nodes_flat if node.verifier_indicator == "TP")
        fp = sum(1 for node in all_verifier_nodes_flat if node.verifier_indicator == "FP")
        tn = sum(1 for node in all_verifier_nodes_flat if node.verifier_indicator == "TN")
        fn = sum(1 for node in all_verifier_nodes_flat if node.verifier_indicator == "FN")

        # Compute various metrics
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        wrong_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics.update({
            'verifier/tp': tp,
            'verifier/fp': fp,
            'verifier/tn': tn,
            'verifier/fn': fn,
            'verifier/accuracy': accuracy,
            'verifier/precision': precision,
            'verifier/recall': recall,
            'verifier/wrong_recall': wrong_recall,
            'verifier/total_samples': total
        })

        return metrics

    def _compute_evaluation_metrics(self, all_generator_nodes: List[List[EvalGenerator]],
                                  all_verifier_nodes: List[List[EvalVerifier]],
                                  data_source: Optional[str] = None) -> Dict[str, Any]:

        metrics = {}

        # Compute generator metrics
        generator_metrics = self._compute_generator_metrics(all_generator_nodes, data_source)
        metrics.update(generator_metrics)

        # Compute regenerator sample metrics
        regenerator_metrics = self._compute_regenerator_metrics(all_generator_nodes, data_source)
        metrics.update(regenerator_metrics)

        # Compute verifier metrics
        verifier_metrics = self._compute_verifier_metrics(all_verifier_nodes, data_source)
        metrics.update(verifier_metrics)

        return metrics

    def preprocess_single_sample(self, node: EvalNode):
        """Preprocess a single node"""
        conversation = node.construct_conversation_history()
        chat = np.array(conversation)

        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=self.enable_thinking
        )

        row_dict = {}
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.config.data.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.config.data.truncation
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': self.tokenizer.encode(prompt_with_chat_template, add_special_tokens=False),
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()

        return row_dict

    def preprocess_batch(self, cur_nodes: List[EvalNode]) -> DataProto:
        """Preprocess a batch of nodes"""
        processed_samples = []
        for node in cur_nodes:
            processed = self.preprocess_single_sample(node)
            processed_samples.append(processed)

        batch = collate_fn(processed_samples)
        new_batch = DataProto.from_single_dict(data=batch)
        return new_batch

    def extract_verifier_answer(self,response: str):
        pattern = r"The answer is (correct|wrong)\.$"
        matches = []
        try:
            matches = [(match.group(1).lower(), match.group(0), match.start())
                    for match in re.finditer(pattern, response)]
        except:
            pass

        if not matches:
            extracted_verifier_answer=None
        else:
            extracted_verifier_answer,_,_ = matches[-1]
        return extracted_verifier_answer    

    def extract_verifier_answer_and_format(self, response: str):
        """
        Extract verifier answer with format indication.
        First tries exact pattern matching, then falls back to fuzzy matching.

        Returns:
            extracted_verifier_answer: 'correct', 'wrong', or None
            verifier_answer_format: True if exact pattern matched, False if fuzzy matching used
        """
        exacted_verifier_answer = self.extract_verifier_answer(response)
        if exacted_verifier_answer is not None:
            return exacted_verifier_answer, True
        # else:
        #     return 'wrong',False
        response_lower = response.lower()
        if any(keyword in response_lower for keyword in ['wrong', 'incorrect', 'mistake']):
            return 'wrong', False
        else:
            return 'correct', False


def perform_multi_turn_eval(batch: DataProto, actor_rollout_wg, tokenizer, config) -> Dict[str, Any]:

    evaluator = MultiTurnEvaluator(config, tokenizer, actor_rollout_wg)
    return evaluator.perform_multi_turn_evaluation(batch)
