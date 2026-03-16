from ast import Pass
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
from .utils.prompt import Initial_Prompt, Generator_Prompt, Verifier_Prompt
from .utils.cover_rl_utils import batch_majority_vote, grade_answer_ttrl
from verl.utils.debug import marked_timer

@dataclass
class Node:

    uid_to_node: Dict[str, 'Node'] = field(default_factory=dict)

    def __init__(self, uid: str, raw_question: str, ground_truth: str, data_source: str, level: int, parent_uid: Optional[str] = None):
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
        self.used_to_update_policy = True
        
        self.save_dict = dict()
        
    def construct_conversation_history(self):
        if self.level == 0:
            conversation_history = [
                {"content": self.raw_question+"\n"+Initial_Prompt, "role": "user"}
            ]
        else:
            parent_node = Node.uid_to_node.get(self.parent_uid)
            conversation_history = parent_node.conversation_history.copy()
            prompt = Verifier_Prompt if self.type == "generator" else Generator_Prompt
            conversation_history.extend([
                {"content": self.response, "role": "assistant"},
                {"content": prompt, "role": "user"}
            ])

        self.conversation_history = conversation_history
        return conversation_history

    def save_DataProto_as_dict(self, batch, index):
        tensors = batch.batch
        non_tensors = batch.non_tensor_batch
        for key, val in tensors.items():
            self.save_dict[key] = val[index]
        for key, val in non_tensors.items():
            self.save_dict[key] = val[index]

        self.reward_tensor = torch.zeros_like(batch.batch["responses"][index], dtype=torch.float32)
        self.save_dict['rm_scores'] = self.reward_tensor

    def calculate_reward(self):
        reward_value = 0.0
        if self.format_correct:
            reward_value += 1.0
        if self.level == 1:
            if self.is_generator_correct_with_majority_gt:
                reward_value += 1.0
        elif self.level==2:
            if self.is_verifier_correct_with_majority_gt:
                reward_value += 1.0
        elif self.level==3:
            if self.is_generator_correct_with_majority_gt:
                reward_value += 1.0

        self.reward_tensor[-1] = reward_value
        self.save_dict['rm_scores'] = self.reward_tensor

    def save_info_as_dict(self):
        """Save node information to dictionary"""
        # List of all possible fields
        all_attrs = [
            # Basic attributes
            'uid', 'parent_uid', 'level', 'data_source', 'format_correct', 'type', 'ground_truth',
            'used_to_update_policy', 'question_uid',
            # Generator related fields
            'extracted_generator_answer', 'answer_anchor_uid', 'majority_gt', 'majority_ratio',
            'is_generator_correct_with_majority_gt', 'is_generator_correct_with_original_gt', 'generator_gt',
            'answer_ratio', 'answer_count',
            # Verifier related fields
            'is_verifier_correct_with_majority_gt', 'is_verifier_correct_with_original_gt',
            'verifier_indicator', 'extracted_verifier_answer'
        ]

        for attr in all_attrs:
            self.save_dict[attr] = getattr(self, attr, None)


class Generator(Node):
    """
    Generator node class, inherits from Node.
    """

    def __init__(self, uid: str, question_uid: str, raw_question: str, ground_truth: str, data_source: str, level: int, parent_uid: Optional[str] = None):
        super().__init__(uid, raw_question, ground_truth, data_source, level, parent_uid)

        # Set node type
        self.type = "generator"

        # Generator-specific attributes
        self.extracted_generator_answer = None
        self.generator_gt = ground_truth
        self.majority_gt = None
        self.majority_ratio = None

        # Correctness flags used for metrics calculation
        self.is_generator_correct_with_majority_gt = None
        self.is_generator_correct_with_original_gt = None

        self.question_uid=question_uid
        # Anchor related: nodes with the same answer share the same anchor_uid
        self.answer_count=None
        self.answer_ratio=None
        self.answer_anchor_uid = None

    def add_generator_info(self, response: str, extracted_answer: str, answer_ratio: float = None, answer_count: int = None, answer_anchor_uid: str = None):
        """Add information to generator node"""
        self.response = response
        self.extracted_generator_answer = extracted_answer
        if self.extracted_generator_answer is None:
            self.format_correct = False
        else:
            self.format_correct = True
        self.answer_ratio = answer_ratio
        self.answer_count = answer_count
        self.answer_anchor_uid = answer_anchor_uid

    def add_generator_pseudo_label(self, majority_gt, majority_ratio):
        self.majority_gt = majority_gt
        self.majority_ratio = majority_ratio
        # Set correctness flag with majority
        self.is_generator_correct_with_majority_gt = grade_answer_ttrl(self.extracted_generator_answer, self.majority_gt) if self.extracted_generator_answer is not None else False
        # Set correctness flag with original ground truth
        self.is_generator_correct_with_original_gt = grade_answer_ttrl(self.extracted_generator_answer, self.generator_gt) if self.extracted_generator_answer is not None else False

class Verifier(Node):
    """
    Verifier node class, inherits from Node.
    """

    def __init__(self, uid: str, question_uid: str, raw_question: str, ground_truth: str, data_source: str, level: int, parent_uid: Optional[str] = None):
        super().__init__(uid, raw_question, ground_truth, data_source, level, parent_uid)

        # Set node type
        self.type = "verifier"
        self.question_uid = question_uid

        # Verifier-specific attributes
        self.extracted_verifier_answer = None
        self.verifier_gt = None  # Will be computed in add_verifier_info
        self.is_verifier_correct_with_majority_gt = None
        self.is_verifier_correct_with_original_gt = None
        self.verifier_indicator = None

        # Inherited anchor_uid from parent generator node
        self.answer_anchor_uid = None

    def add_verifier_confusion_matrix(self):
        """Compute verifier confusion matrix indicators"""
        if self.is_verifier_correct_with_original_gt:
            self.verifier_indicator = "TP" if self.extracted_verifier_answer == "correct" else "TN"
        else:
            self.verifier_indicator = "FN" if self.extracted_verifier_answer == "wrong" else "FP"        

    def add_verifier_info(self, response: str, extracted_answer: str, format_correct: bool = True):
        """Add information to verifier node"""
        self.response = response
        self.extracted_verifier_answer = extracted_answer
        self.format_correct = format_correct
        
        parent_node = Node.uid_to_node.get(self.parent_uid)
        if parent_node.is_generator_correct_with_original_gt:
            self.verifier_gt = "correct"
        else:
            self.verifier_gt = "wrong"

        if self.extracted_verifier_answer==self.verifier_gt:
            self.is_verifier_correct_with_original_gt = True
        else:
            self.is_verifier_correct_with_original_gt = False

        self.answer_anchor_uid = parent_node.answer_anchor_uid
        self.add_verifier_confusion_matrix()

    def add_verifier_pseudo_label(self):
        """Add pseudo_label to verifier node"""
        parent_node = Node.uid_to_node.get(self.parent_uid)
        if parent_node.is_generator_correct_with_majority_gt:
            if self.extracted_verifier_answer == "correct":
                self.is_verifier_correct_with_majority_gt=True
            else:
                self.is_verifier_correct_with_majority_gt=False
        else:
            if self.extracted_verifier_answer == "wrong":
                self.is_verifier_correct_with_majority_gt=True
            else:
                self.is_verifier_correct_with_majority_gt=False


class CoVerRLMultiTurnRolloutLoop:

    def __init__(self, config: Dict[str, Any], tokenizer, actor_rollout_wg):
        self.config = config
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg

        cover_rl_config = config.get("cover_rl", {})
        self.cover_rl_config=cover_rl_config
        self.num_turns = cover_rl_config.get("num_turns", 2)
        self.initial_rollout_n = cover_rl_config.get("initial_rollout_n", 64)
        self.initial_rollout_sample_n = cover_rl_config.get("initial_rollout_sample_n", 32)
        self.use_anchor_for_verifier = cover_rl_config.get("use_anchor_for_verifier", False)
        self.seed = cover_rl_config.get("seed", 42)
        self.double_check = cover_rl_config.get("double_check", False)

        # Configuration for different majority ratio scenarios
        self.verifier_rollout_max_n = cover_rl_config.get("verify_regenerate", {}).get("verifier_rollout_max_n", 8)
        self.regenerator_rollout_n = cover_rl_config.get("verify_regenerate", {}).get("regenerator_rollout_n", 4)
        self.choose_wrong_node_strategy = cover_rl_config.get("verify_regenerate", {}).get("choose_wrong_node_strategy", "least_ratio")
        self.enable_thinking = cover_rl_config.get("enable_thinking", False)

        Node.uid_to_node = {}

    def _generate_responses(self, nodes: List[Node], rollout_n: int) -> Tuple[DataProto, List[str]]:
        batch = self.preprocess_batch(nodes)
        batch=batch.repeat(repeat_times=rollout_n, interleave=True)
        batch_input = batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"],
                            non_tensor_batch_keys=["raw_prompt_ids"])  
        
        batch_output = self.actor_rollout_wg.generate_sequences(batch_input)
        batch = batch.union(batch_output)                        

        responses = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

        return batch, responses

    def _create_child_nodes(self, parent_nodes: List[Node], batch: DataProto, responses: List[str],
                           node_type: str, rollout_n: int, **kwargs) -> List[Node]:
        child_nodes = []
        num_parents = len(parent_nodes)

        for i, parent_node in enumerate(parent_nodes):
            for j in range(rollout_n):
                uid = str(uuid.uuid4())
                batch_idx = i * rollout_n + j

                if node_type == "generator" :
                    child_node = Generator(
                        uid=uid,
                        question_uid=kwargs.get('question_uid', parent_node.question_uid),
                        raw_question=parent_node.raw_question,
                        ground_truth=parent_node.ground_truth,
                        data_source=parent_node.data_source,
                        level=parent_node.level + 1,
                        parent_uid=parent_node.uid
                    )
                    # Add generator info
                    extracted_answer = kwargs['model_answers_list'][i][j]
                    answer_ratio = kwargs.get('answer_ratios_list', [{}]*num_parents)[i].get(extracted_answer, {}).get("ratio")
                    answer_count = kwargs.get('answer_ratios_list', [{}]*num_parents)[i].get(extracted_answer, {}).get("count")
                    answer_anchor_uid = kwargs.get('answer_ratios_list', [{}]*num_parents)[i].get(extracted_answer, {}).get("answer_anchor_uid")

                    child_node.add_generator_info(responses[batch_idx], extracted_answer, answer_ratio, answer_count, answer_anchor_uid)

                    # Add pseudo label logic
                    if kwargs.get('is_re_generator', False):
                        grandparent_node = Node.uid_to_node.get(parent_node.parent_uid)
                        child_node.add_generator_pseudo_label(
                            grandparent_node.majority_gt,
                            grandparent_node.majority_ratio
                        )
                    else:
                        # For initial generator nodes: directly use majority vote result
                        if kwargs.get('majority_gt_list') and kwargs.get('majority_ratio_list'):
                            child_node.add_generator_pseudo_label(
                                kwargs['majority_gt_list'][i],
                                kwargs['majority_ratio_list'][i]
                            )

                elif node_type == "verifier":
                    child_node = Verifier(
                        uid=uid,
                        question_uid=parent_node.question_uid,
                        raw_question=parent_node.raw_question,
                        ground_truth=parent_node.ground_truth,
                        data_source=parent_node.data_source,
                        level=parent_node.level + 1,
                        parent_uid=parent_node.uid
                    )
                    # Add verifier info
                    extracted_answer, format_correct = kwargs['extracted_answers'][batch_idx]
                    child_node.add_verifier_info(responses[batch_idx], extracted_answer, format_correct)
                    child_node.add_verifier_pseudo_label()
                child_node.save_DataProto_as_dict(batch, batch_idx)
                Node.uid_to_node[uid] = child_node
                child_nodes.append(child_node)

        return child_nodes

    def _create_root_nodes(self, batch: DataProto) -> List[Node]:
        root_nodes = []
        for i in range(len(batch.batch['input_ids'])):
            uid = str(uuid.uuid4())
            root_node = Node(
                uid=uid,
                data_source=batch.non_tensor_batch['data_source'][i],
                raw_question=batch.non_tensor_batch['extra_info'][i]['question'],
                ground_truth=batch.non_tensor_batch['reward_model'][i]['ground_truth'],
                level=0,
                parent_uid=None,
            )
            root_node.question_uid = uid
            root_nodes.append(root_node)
            Node.uid_to_node[uid] = root_node
        return root_nodes

    def _perform_majority_vote_and_create_generator_nodes(self, root_nodes: List[Node], batch: DataProto,
                                                    responses: List[str]) -> List[Generator]:
        # Majority vote
        majority_gt_list, majority_ratio_list, model_answers_list, answer_ratios_list = batch_majority_vote(
            responses, self.initial_rollout_n)

        # Create generator child nodes
        generator_nodes = self._create_child_nodes(
            root_nodes, batch, responses, "generator", self.initial_rollout_n,
            model_answers_list=model_answers_list,
            answer_ratios_list=answer_ratios_list,
            majority_gt_list=majority_gt_list,
            majority_ratio_list=majority_ratio_list
        )

        return generator_nodes

    def preprocess_single_sample(self, node: Node):
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

    def preprocess_batch(self, cur_nodes: List[Node]) -> DataProto:
        """Preprocess a batch of nodes"""
        processed_samples = []
        for node in cur_nodes:
            processed = self.preprocess_single_sample(node)
            processed_samples.append(processed)

        batch = collate_fn(processed_samples)
        new_batch = DataProto.from_single_dict(data=batch)
        return new_batch

    def _align_verify_nodes_to_dp(self, question_node_groups: List[Tuple[List[Node], List[Node]]]) -> List[Node]:
        """
        Align verify stage nodes to ensure total count is divisible by dp_size.
        The number of positive and negative samples for each question must be equal.

        Reduce positive and negative sample counts (keeping them equal) from back to front to satisfy divisibility.

        Args:
            question_node_groups: List of (positive_samples_list, negative_samples_list) tuples for each question,
                                  where positive_count == negative_count

        Returns:
            Flattened node list after alignment
        """
        n_gpus_per_node = self.config.trainer.n_gpus_per_node
        tensor_model_parallel_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        dp_size = n_gpus_per_node// tensor_model_parallel_size

        # Calculate positive sample count for each question (== negative sample count)
        counts = [len(pos) for pos, neg in question_node_groups]
        total = sum(c * 2 for c in counts)  # positive + negative

        if total % dp_size == 0:
            # Already aligned, return flattened result directly
            result = []
            for pos, neg in question_node_groups:
                result.extend(pos)
                result.extend(neg)
            return result

        # Find maximum target <= total such that target % dp_size == 0 and target is even
        # (because positive and negative samples must be reduced in pairs, reducing by 2 each time)
        # lcm(dp_size, 2) is the least common multiple of dp_size and 2
        step = dp_size if dp_size % 2 == 0 else dp_size * 2
        target = (total // step) * step

        to_remove = total - target
        pairs_to_remove = to_remove // 2  # Number of positive-negative pairs to remove

        # Start reducing from the last question
        for i in range(len(counts) - 1, -1, -1):
            if pairs_to_remove <= 0:
                break
            can_remove = counts[i]  # Maximum can reduce to 0
            remove_now = min(pairs_to_remove, can_remove)
            counts[i] -= remove_now
            pairs_to_remove -= remove_now

        # Build result based on reduced counts
        result = []
        for idx, (pos, neg) in enumerate(question_node_groups):
            c = counts[idx]
            if c > 0:
                result.extend(pos[:c])
                result.extend(neg[:c])

        return result

    def _select_top_k(self, generator_nodes: List[Generator], rollout_n: int, sample_n: int) -> List[Generator]:
        """Select top-k nodes for training"""
        num_prompts = len(generator_nodes) // rollout_n
        selected_indices = []
        for i in range(num_prompts):
            start = i * rollout_n
            selected_indices.extend(range(start, start + sample_n))
        return [generator_nodes[i] for i in selected_indices]

    def _collect_rollout_data(self, nodes: List[Node]) -> DataProto:
        """Collect rollout data"""
        rollout_data = []
        for node in nodes:
            node.save_info_as_dict()
            rollout_data.append(node.save_dict)
        return DataProto.from_single_dict(data=collate_fn(rollout_data))

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
        
        response_lower = response.lower()
        if any(keyword in response_lower for keyword in ['wrong', 'incorrect', 'mistake']):
            return 'wrong', False
        else:
            return 'correct', False

    def _select_wrong_nodes(self, wrong_nodes: List[Generator]) -> List[Generator]:
        """
        Select wrong nodes based on strategy (fixed selection of 1)

        Args:
            wrong_nodes: List of all wrong nodes

        Returns:
            List of selected wrong nodes (length 1)
        """

        if self.choose_wrong_node_strategy == "random":
            # Randomly select 1
            import random
            random.seed(self.seed)
            return [random.choice(wrong_nodes)]

        elif self.choose_wrong_node_strategy == "least_ratio":
            # Sort by answer_ratio from smallest to largest, select the first
            sorted_wrong_nodes = sorted(wrong_nodes, key=lambda x: x.answer_ratio if x.answer_ratio is not None else float('inf'))
            return [sorted_wrong_nodes[0]]
        else:
            raise ValueError(f"choose_wrong_node_strategy must be 'random' or 'least_ratio', got {self.choose_wrong_node_strategy}")

    def _verify(self, generator_nodes: List[Generator]) -> List[Verifier]:
        """Execute verification step"""
        import random
        num_prompts = len(generator_nodes) // self.initial_rollout_n

        # Collect (positive_samples_list, negative_samples_list) for each question, for later dp alignment
        question_node_groups = []

        for i in range(num_prompts):
            start = i * self.initial_rollout_n
            end = start + self.initial_rollout_n
            question_generator_nodes = generator_nodes[start:end]

            majority_correct_nodes = [node for node in question_generator_nodes if node.is_generator_correct_with_majority_gt]
            majority_wrong_nodes = [node for node in question_generator_nodes if not node.is_generator_correct_with_majority_gt]

            if len(majority_wrong_nodes) < 1:
                continue

            if self.use_anchor_for_verifier:
                # use_anchor_for_verifier = True logic
                if len(majority_correct_nodes) >= self.verifier_rollout_max_n:
                    # Correct nodes count >= verifier_rollout_max_n
                    selected_correct_nodes = majority_correct_nodes[:self.verifier_rollout_max_n]
                    verify_count_per_wrong = self.verifier_rollout_max_n
                else:
                    # Correct nodes count < verifier_rollout_max_n, select maximum even number
                    x = len(majority_correct_nodes) if len(majority_correct_nodes) % 2 == 0 else len(majority_correct_nodes) - 1
                    if x == 0:
                        continue
                    selected_correct_nodes = majority_correct_nodes[:x]
                    verify_count_per_wrong = x

                selected_wrong_nodes = self._select_wrong_nodes(majority_wrong_nodes)

                # Build positive and negative sample lists
                pos_nodes = list(selected_correct_nodes)
                neg_nodes = []
                for node in selected_wrong_nodes:
                    neg_nodes.extend([node] * verify_count_per_wrong)
                question_node_groups.append((pos_nodes, neg_nodes))

            else:
                import random
                random.seed(self.seed)
                selected_correct_node = random.choice(majority_correct_nodes)
                selected_wrong_nodes = self._select_wrong_nodes(majority_wrong_nodes)

                # use_anchor_for_verifier = False logic
                # Use the same verification count logic as if branch, but only select 1 correct node
                if len(majority_correct_nodes) >= self.verifier_rollout_max_n:
                    # Correct nodes count >= verifier_rollout_max_n
                    verify_count_per_correct = self.verifier_rollout_max_n
                    verify_count_per_wrong = self.verifier_rollout_max_n
                else:
                    # Correct nodes count < verifier_rollout_max_n, select maximum even number
                    x = len(majority_correct_nodes) if len(majority_correct_nodes) % 2 == 0 else len(majority_correct_nodes) - 1
                    if x == 0:
                        continue
                    verify_count_per_wrong = x
                    verify_count_per_correct=x

                # Build positive and negative sample lists
                pos_nodes = [selected_correct_node] * verify_count_per_correct
                neg_nodes = []
                for node in selected_wrong_nodes:
                    neg_nodes.extend([node] * verify_count_per_wrong)
                question_node_groups.append((pos_nodes, neg_nodes))

        # Align to integer multiple of dp_size, keeping positive and negative sample counts equal for each question
        expanded_verifier_nodes = self._align_verify_nodes_to_dp(question_node_groups)

        # Batch verification: reuse _generate_responses function, each node generates 1 verifier
        verifier_nodes = []
        if expanded_verifier_nodes:
            verify_batch, verify_responses = self._generate_responses(expanded_verifier_nodes, rollout_n=1)
            extracted_verifier_answers = [self.extract_verifier_answer_and_format(response) for response in verify_responses]

            # Create verifiers for all nodes at once, each node generates 1 verifier
            verifier_nodes = self._create_child_nodes(expanded_verifier_nodes,verify_batch,
                verify_responses,"verifier",1,extracted_answers=extracted_verifier_answers
            )

        return verifier_nodes

    def _regenerate(self, verifier_nodes: List[Verifier]) -> List[Generator]:
        """
        Unified function for regenerating nodes that failed verification

        Args:
            verifier_wrong_nodes: List of verifier nodes marked as wrong

        Returns:
            Newly generated generator nodes list
        """
        # Select nodes marked as wrong by verifier for regeneration
        filtered_verifier_wrong_nodes = [node for node in verifier_nodes if node.extracted_verifier_answer == "wrong"]

        n_gpus_per_node = self.config.trainer.n_gpus_per_node
        tensor_model_parallel_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        dp_size = (n_gpus_per_node) // tensor_model_parallel_size

        if dp_size <= self.regenerator_rollout_n:
            self.regenerator_rollout_n = self.regenerator_rollout_n - (self.regenerator_rollout_n % self.config.trainer.n_gpus_per_node) 
        else :
            remainder = len(filtered_verifier_wrong_nodes) % dp_size
            if remainder != 0:
                filtered_verifier_wrong_nodes=filtered_verifier_wrong_nodes[:-remainder]
          
        if not filtered_verifier_wrong_nodes:
            return []
        
        re_generator_batch, re_generator_responses = self._generate_responses(filtered_verifier_wrong_nodes, self.regenerator_rollout_n)

        # Perform majority vote to get model_answers_list and answer_ratios_list
        _, _, model_answers_list, answer_ratios_list = batch_majority_vote(re_generator_responses, self.regenerator_rollout_n)

        # Create child nodes, will automatically get pseudo label from parent.parent (original generator)
        re_generator_nodes = self._create_child_nodes(filtered_verifier_wrong_nodes, re_generator_batch, re_generator_responses, "generator", self.regenerator_rollout_n,
                                      model_answers_list=model_answers_list, answer_ratios_list=answer_ratios_list,
                                      is_re_generator=True)
        return re_generator_nodes

    def _double_majority(self, generator_nodes: List[Generator], verifier_nodes: List[Verifier], global_steps: int = 0) -> Tuple[List[Generator], List[Verifier]]:
        """
        Double majority vote: For nodes that were majority in the first round generation,
        if more verifications are false than true, set the corresponding generator node's used_to_update_policy to false
        """
        if self.double_check == "non" :
            return generator_nodes, verifier_nodes
        elif self.double_check == "all":
            double_check_nodes=[]
            for node in verifier_nodes:
                parent_node=Node.uid_to_node.get(node.parent_uid)
                if parent_node.is_generator_correct_with_majority_gt:
                    double_check_nodes.append(node)
        else:
            raise ValueError(f"double_check must be 'non' or 'all', got {self.double_check}")

        # Group verification results by question_uid
        question_verification_stats = defaultdict(lambda: {"correct_count": 0, "wrong_count": 0})

        for double_check_node in double_check_nodes:
            question_uid = double_check_node.question_uid
            if double_check_node.extracted_verifier_answer == "correct":
                question_verification_stats[question_uid]["correct_count"] += 1
            else:
                question_verification_stats[question_uid]["wrong_count"] += 1

        filtered_generator_nodes=[]
        filtered_verifier_nodes=[]
        filtered_out_generator_nodes = []
        # For each question_uid, if wrong > correct, mark corresponding generator and all verifier nodes as not updating policy
        for verifier_node in verifier_nodes:
            question_uid = verifier_node.question_uid
            if question_uid in question_verification_stats:
                if question_verification_stats[question_uid]["wrong_count"]< question_verification_stats[question_uid]["correct_count"]:
                    filtered_verifier_nodes.append(verifier_node)
            else:
                filtered_verifier_nodes.append(verifier_node)

        for generator_node in generator_nodes:
            question_uid = generator_node.question_uid
            if question_uid in question_verification_stats:
                if question_verification_stats[question_uid]["wrong_count"]< question_verification_stats[question_uid]["correct_count"]:
                    filtered_generator_nodes.append(generator_node)
                else:
                    # Filtered out generator nodes (wrong_count >= correct_count)
                    filtered_out_generator_nodes.append(generator_node)
            else:
                filtered_generator_nodes.append(generator_node)

        # Record filtered out generator nodes to CSV (deduplicated by question_uid)
        if filtered_out_generator_nodes:
            # First separate generator nodes into correct and wrong categories
            correct_generator_nodes = [node for node in filtered_out_generator_nodes if node.is_generator_correct_with_original_gt]
            wrong_generator_nodes = [node for node in filtered_out_generator_nodes if not node.is_generator_correct_with_original_gt]

            # Extract unique question_uids from each category and deduplicate (keep only one node per question_uid)
            correct_question_uids = set(node.question_uid for node in correct_generator_nodes)
            wrong_question_uids = set(node.question_uid for node in wrong_generator_nodes)

            # Select a representative node for each question_uid
            filtered_correct_nodes = []
            for question_uid in correct_question_uids:
                # Select the first node under this question_uid as representative
                node = next(node for node in correct_generator_nodes if node.question_uid == question_uid)
                filtered_correct_nodes.append(node)

            filtered_wrong_nodes = []
            for question_uid in wrong_question_uids:
                # Select the first node under this question_uid as representative
                node = next(node for node in wrong_generator_nodes if node.question_uid == question_uid)
                filtered_wrong_nodes.append(node)

            # Collect all filtered question_uids
            filtered_question_uids = set()
            for node in filtered_correct_nodes + filtered_wrong_nodes:
                filtered_question_uids.add(node.question_uid)

            # Get all related generator nodes (for computing statistics)
            all_generator_nodes = generator_nodes

        return filtered_generator_nodes, filtered_verifier_nodes


    def filter_by_majority_ratio(self, generator_nodes: List[Generator], majority_ratio_low, step: int = 0) -> List[Generator]:
        """
        Filter out nodes with majority_ratio less than majority_ratio_low

        Args:
            generator_nodes: Generator node list to filter
            majority_ratio_low: Lower threshold for majority_ratio
            step: Current step, used for logging

        Returns:
            Filtered generator node list
        """
        if self.num_turns != 1:
            return generator_nodes

        filtered_nodes = []
        filtered_out_correct_question_uids = set()  # Filtered out correct question_uid set
        filtered_out_wrong_question_uids = set()   # Filtered out wrong question_uid set

        for node in generator_nodes:
            if node.majority_ratio is not None and node.majority_ratio >= majority_ratio_low:
                filtered_nodes.append(node)
            else:
                # Collect filtered question_uids by correctness category
                if node.is_generator_correct_with_original_gt:
                    filtered_out_correct_question_uids.add(node.question_uid)
                else:
                    filtered_out_wrong_question_uids.add(node.question_uid)

        # Record filtered out generator nodes to CSV (deduplicated by question_uid)
        filtered_correct_nodes = []
        for question_uid in filtered_out_correct_question_uids:
            # Select the first filtered node under this question_uid as representative
            node = next(node for node in generator_nodes if node.question_uid == question_uid and node.is_generator_correct_with_original_gt)
            filtered_correct_nodes.append(node)

        filtered_wrong_nodes = []
        for question_uid in filtered_out_wrong_question_uids:
            # Select the first filtered node under this question_uid as representative
            node = next(node for node in generator_nodes if node.question_uid == question_uid and not node.is_generator_correct_with_original_gt)
            filtered_wrong_nodes.append(node)

        if filtered_correct_nodes or filtered_wrong_nodes:
            # Collect all filtered question_uids
            filtered_question_uids = set()
            for node in filtered_correct_nodes + filtered_wrong_nodes:
                filtered_question_uids.add(node.question_uid)

        return filtered_nodes


    def execute_multi_turn_rollout(self, batch: DataProto, timing_raw=None, global_steps: int = 0) -> DataProto:
        # Step 1: Create root nodes
        root_nodes = self._create_root_nodes(batch)

        # Round 1: Generator - Generate initial_rollout_n responses from each root node
        with marked_timer("first_generation", timing_raw):
            gen_batch, responses = self._generate_responses(root_nodes, self.initial_rollout_n)
            generator_nodes = self._perform_majority_vote_and_create_generator_nodes(
                root_nodes, gen_batch, responses)

        # Determine whether to proceed with subsequent processing based on num_turns parameter
        if self.num_turns == 1:

            if "majority_ratio_low" in self.cover_rl_config:
                generator_nodes=self.filter_by_majority_ratio(generator_nodes, self.cover_rl_config["majority_ratio_low"], step=global_steps)

            # Calculate rewards for level-1 generator nodes
            for node in generator_nodes:
                node.calculate_reward()

            rollout_nodes = self._select_top_k(generator_nodes, self.initial_rollout_n, self.initial_rollout_sample_n)
            return self._collect_rollout_data(rollout_nodes)

        elif self.num_turns == 2:
            if "majority_ratio_low" in self.cover_rl_config:
                generator_nodes=self.filter_by_majority_ratio(generator_nodes, self.cover_rl_config["majority_ratio_low"], step=global_steps)
            num_prompts = len(generator_nodes) // self.initial_rollout_n

            # Compute average majority_ratio for all questions in the batch
            batch_majority_ratios = []
            for i in range(num_prompts):
                start = i * self.initial_rollout_n
                batch_majority_ratios.append(generator_nodes[start].majority_ratio)

            with marked_timer("second_verification", timing_raw):
                verifier_nodes = self._verify(generator_nodes)

            filtered_generator_nodes,filtered_verifier_nodes=self._double_majority(generator_nodes,verifier_nodes, global_steps)

            # Only select format-correct nodes
            selected_verifier_nodes=[verifier_node for verifier_node in filtered_verifier_nodes if verifier_node.format_correct]
            with marked_timer("second_regeneration", timing_raw):
                re_generator_nodes = self._regenerate(selected_verifier_nodes)

            # Calculate rewards for all nodes
            # level-1 generator
            for node in filtered_generator_nodes:
                node.calculate_reward()
            # level-2 verifier and level-3 re-generator
            for node in filtered_verifier_nodes + re_generator_nodes:
                node.calculate_reward()

            selected_generator_nodes = self._select_top_k(filtered_generator_nodes, self.initial_rollout_n, self.initial_rollout_sample_n)
            rollout_nodes = selected_generator_nodes + filtered_verifier_nodes + re_generator_nodes
            # Collect data: return all generated and verified nodes
            return self._collect_rollout_data(rollout_nodes)

        else:
            raise ValueError(f"num_turns must be 1 or 2, got {self.num_turns}")
