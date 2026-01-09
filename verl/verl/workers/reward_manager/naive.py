# # Copyright 2024 Bytedance Ltd. and/or its affiliates
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# from collections import defaultdict
# from typing import Any
# from concurrent.futures import ThreadPoolExecutor, as_completed

# import torch

# from verl import DataProto
# from verl.utils.reward_score import default_compute_score
# from verl.workers.reward_manager import register
# from verl.workers.reward_manager.abstract import AbstractRewardManager

# import logging
# import re

# def extract_code(text_input):
#     """Extracts the last Lean 4 code block from the model's output."""
#     try:
#         matches = re.findall(r'```lean4\n(.*?)\n```', text_input, re.DOTALL)
#         return matches[-1].strip() if matches else "No Lean 4 code block found."
#     except Exception:
#         return "Error during code extraction."

# @register("naive")
# class NaiveRewardManager(AbstractRewardManager):
#     """The reward manager."""

#     def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
#         """
#         Initialize the NaiveRewardManager instance.

#         Args:
#             tokenizer: The tokenizer used to decode token IDs into text.
#             num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
#             compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
#             reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
#                 "data_source".
#         """
#         self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
#         self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
#         self.compute_score = compute_score or default_compute_score
#         self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

#     # def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
#     #     """We will expand this function gradually based on the available datasets"""

#     #     # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
#     #     if "rm_scores" in data.batch.keys():
#     #         if return_dict:
#     #             reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
#     #             reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
#     #             return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
#     #         else:
#     #             return data.batch["rm_scores"]

#     #     reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
#     #     reward_extra_info = defaultdict(list)

#     #     already_print_data_sources = {}

#     #     for i in range(len(data)):
#     #         data_item = data[i]  # DataProtoItem

#     #         prompt_ids = data_item.batch["prompts"]

#     #         prompt_length = prompt_ids.shape[-1]

#     #         valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
#     #         valid_prompt_ids = prompt_ids[-valid_prompt_length:]

#     #         response_ids = data_item.batch["responses"]
#     #         valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
#     #         valid_response_ids = response_ids[:valid_response_length]

#     #         # decode
#     #         prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
#     #         response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

#     #         # logging.warning("testtesttesttest")
#     #         # logging.warning("testtesttesttest")
#     #         # logging.warning(f"{prompt_str}")
#     #         # with open("test.json",'w') as f:
#     #         #     import json
#     #         #     json.dump({"prompt":prompt_str,"response":response_str},f,indent=4)
#     #         ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
#     #         data_source = data_item.non_tensor_batch[self.reward_fn_key]
#     #         extra_info = data_item.non_tensor_batch.get("extra_info", {})
#     #         num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
#     #         rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
#     #         extra_info["num_turns"] = num_turns
#     #         extra_info["rollout_reward_scores"] = rollout_reward_scores

#     #         response_str = extract_code(prompt_str + response_str)
#     #         #logging.warning(f"{response_str}")
#     #         score = self.compute_score(
#     #             data_source=data_source,
#     #             solution_str=response_str,
#     #             ground_truth=ground_truth,
#     #             extra_info=extra_info,
#     #         )

#     #         if isinstance(score, dict):
#     #             reward = score["score"]
#     #             # Store the information including original reward
#     #             for key, value in score.items():
#     #                 reward_extra_info[key].append(value)
#     #         else:
#     #             reward = score

#     #         reward_tensor[i, valid_response_length - 1] = reward

#     #         if data_source not in already_print_data_sources:
#     #             already_print_data_sources[data_source] = 0

#     #         if already_print_data_sources[data_source] < self.num_examine:
#     #             already_print_data_sources[data_source] += 1
#     #             print("[prompt]", prompt_str)
#     #             print("[response]", response_str)
#     #             print("[ground_truth]", ground_truth)
#     #             if isinstance(score, dict):
#     #                 for key, value in score.items():
#     #                     print(f"[{key}]", value)
#     #             else:
#     #                 print("[score]", score)

#     #     if return_dict:
#     #         return {
#     #             "reward_tensor": reward_tensor,
#     #             "reward_extra_info": reward_extra_info,
#     #         }
#     #     else:
#     #         return reward_tensor
#     def _compute_single_reward(self, data_item, index):
#         """Compute reward for a single data item. Returns (index, result_dict)."""
#         prompt_ids = data_item.batch["prompts"]

#         prompt_length = prompt_ids.shape[-1]

#         valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
#         valid_prompt_ids = prompt_ids[-valid_prompt_length:]

#         response_ids = data_item.batch["responses"]
#         valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
#         valid_response_ids = response_ids[:valid_response_length]

#         # decode
#         prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
#         response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

#         ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
#         data_source = data_item.non_tensor_batch[self.reward_fn_key]
#         extra_info = data_item.non_tensor_batch.get("extra_info", {})
#         num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
#         rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
#         extra_info["num_turns"] = num_turns
#         extra_info["rollout_reward_scores"] = rollout_reward_scores

#         response_str = extract_code(prompt_str + response_str)
        
#         score = self.compute_score(
#             data_source=data_source,
#             solution_str=response_str,
#             ground_truth=ground_truth,
#             extra_info=extra_info,
#         )

#         return {
#             "index": index,
#             "score": score,
#             "valid_response_length": int(valid_response_length),
#             "prompt_str": prompt_str,
#             "response_str": response_str,
#             "ground_truth": ground_truth,
#             "data_source": data_source,
#         }

#     def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
#         """We will expand this function gradually based on the available datasets"""

#         # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
#         if "rm_scores" in data.batch.keys():
#             if return_dict:
#                 reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
#                 reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
#                 return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
#             else:
#                 return data.batch["rm_scores"]

#         reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
#         reward_extra_info = defaultdict(list)

#         already_print_data_sources = {}

#         # Use ThreadPoolExecutor for parallel reward computation
#         max_workers = 32
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             # Submit all tasks
#             futures = {
#                 executor.submit(self._compute_single_reward, data[i], i): i 
#                 for i in range(len(data))
#             }
            
#             # Collect results as they complete
#             results = []
#             for future in as_completed(futures):
#                 results.append(future.result())
        
#         # Sort results by index to maintain order for printing
#         results.sort(key=lambda x: x["index"])
        
#         # Process results
#         for result in results:
#             i = result["index"]
#             score = result["score"]
#             valid_response_length = result["valid_response_length"]
#             prompt_str = result["prompt_str"]
#             response_str = result["response_str"]
#             ground_truth = result["ground_truth"]
#             data_source = result["data_source"]

#             if isinstance(score, dict):
#                 reward = score["score"]
#                 # Store the information including original reward
#                 for key, value in score.items():
#                     reward_extra_info[key].append(value)
#             else:
#                 reward = score

#             reward_tensor[i, valid_response_length - 1] = reward

#             if data_source not in already_print_data_sources:
#                 already_print_data_sources[data_source] = 0

#             if already_print_data_sources[data_source] < self.num_examine:
#                 already_print_data_sources[data_source] += 1
#                 print("[prompt]", prompt_str)
#                 print("[response]", response_str)
#                 print("[ground_truth]", ground_truth)
#                 if isinstance(score, dict):
#                     for key, value in score.items():
#                         print(f"[{key}]", value)
#                 else:
#                     print("[score]", score)

#         if return_dict:
#             return {
#                 "reward_tensor": reward_tensor,
#                 "reward_extra_info": reward_extra_info,
#             }
#         else:
#             return reward_tensor


# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from collections import defaultdict
from typing import Any
from multiprocessing import Pool, TimeoutError as MPTimeoutError
import multiprocessing

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

import logging
import re

def extract_code(text_input):
    """Extracts the last Lean 4 code block from the model's output."""
    try:
        matches = re.findall(r'```lean4\n(.*?)\n```', text_input, re.DOTALL)
        return matches[-1].strip() if matches else "No Lean 4 code block found."
    except Exception:
        return "Error during code extraction."


# Global variable to hold compute_score function for multiprocessing
_compute_score_fn = None

def _init_worker(compute_score_fn):
    """Initialize worker process with the compute_score function."""
    global _compute_score_fn
    _compute_score_fn = compute_score_fn

def _compute_single_reward_worker(args):
    """Worker function for multiprocessing. Must be at module level."""
    global _compute_score_fn
    (index, prompt_str, response_str, ground_truth, data_source, 
     extra_info, valid_response_length) = args
    
    try:
        response_str = extract_code(prompt_str + response_str)
        
        score = _compute_score_fn(
            data_source=data_source,
            solution_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )

        return {
            "index": index,
            "score": score,
            "valid_response_length": valid_response_length,
            "prompt_str": prompt_str,
            "response_str": response_str,
            "ground_truth": ground_truth,
            "data_source": data_source,
        }
    except Exception as e:
        logging.warning(f"Error computing reward for sample {index}: {e}")
        return {
            "index": index,
            "score": 0,
            "valid_response_length": valid_response_length,
            "prompt_str": prompt_str,
            "response_str": response_str,
            "ground_truth": ground_truth,
            "data_source": data_source,
            "error": str(e),
        }


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", timeout=60, max_workers=192) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
            timeout: Maximum time in seconds to wait for each reward computation. Defaults to 60.
            max_workers: Maximum number of parallel workers. Defaults to 32.
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.timeout = timeout  # Timeout for each reward computation
        self.max_workers = max_workers

    def _prepare_task_args(self, data_item, index):
        """Prepare arguments for a single task (serializable for multiprocessing)."""
        prompt_ids = data_item.batch["prompts"]

        prompt_length = prompt_ids.shape[-1]

        valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        response_ids = data_item.batch["responses"]
        valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        # decode
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_source = data_item.non_tensor_batch[self.reward_fn_key]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        return (index, prompt_str, response_str, ground_truth, data_source, 
                extra_info, int(valid_response_length))

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # Prepare all task arguments
        all_task_args = [self._prepare_task_args(data[i], i) for i in range(len(data))]
        
        results = []
        
        # Use multiprocessing Pool with spawn context for better isolation
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=self.max_workers, initializer=_init_worker, initargs=(self.compute_score,)) as pool:
            # Submit all tasks asynchronously
            async_results = {
                pool.apply_async(_compute_single_reward_worker, (args,)): args[0]  # args[0] is index
                for args in all_task_args
            }
            
            # Collect results with timeout
            for async_result, index in async_results.items():
                try:
                    result = async_result.get(timeout=self.timeout)
                    results.append(result)
                except MPTimeoutError:
                    logging.warning(f"Timeout for sample {index} after {self.timeout} seconds. Assigning reward 0.")
                    # Find the original args to get necessary info
                    original_args = all_task_args[index]
                    results.append({
                        "index": index,
                        "score": 0,
                        "valid_response_length": original_args[6],  # valid_response_length
                        "prompt_str": "[TIMEOUT]",
                        "response_str": "[TIMEOUT]",
                        "ground_truth": original_args[3],  # ground_truth
                        "data_source": original_args[4],  # data_source
                        "timeout": True,
                    })
                except Exception as e:
                    logging.warning(f"Error for sample {index}: {e}. Assigning reward 0.")
                    original_args = all_task_args[index]
                    results.append({
                        "index": index,
                        "score": 0,
                        "valid_response_length": original_args[6],
                        "prompt_str": "[ERROR]",
                        "response_str": "[ERROR]",
                        "ground_truth": original_args[3],
                        "data_source": original_args[4],
                        "error": str(e),
                    })
            
            # Terminate any remaining processes
            pool.terminate()
            pool.join()
        
        # Sort results by index to maintain order for printing
        results.sort(key=lambda x: x["index"])
        
        # Process results
        for result in results:
            i = result["index"]
            score = result["score"]
            valid_response_length = result["valid_response_length"]
            prompt_str = result["prompt_str"]
            response_str = result["response_str"]
            ground_truth = result["ground_truth"]
            data_source = result["data_source"]

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor