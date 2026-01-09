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

import re
import nest_asyncio
from verl.utils.reward_score.Lean4Client import Lean4Client
from typing import List, Dict, Any, Optional
import re
import hashlib
from tqdm.auto import tqdm
import argparse
import copy
import os
import time

class LeanChecker:
    def __init__(
            self, 
            lean4_project_path: str,
            # tmp_folder: str
    ):
        # self.Lean4_client = Lean4Client(lean4_project_path=lean4_project_path, tmp_folder=tmp_folder)
        self.Lean4_client = Lean4Client(lean4_project_path=lean4_project_path)

    def _submit_job(
            self, 
            job_list: List[Dict], 
            timeout: int = 120,
            tmp_folder: str = "test_tmp_folder"
    ) -> Dict[str, Any]:
        """
        Submit the list of jobs to the Lean server and return the original response.
        :param job_list: List of jobs to be submitted, the inner dict should follows the format of Kimina-Lean-Server as follows:
            - "proof": The Lean4 proof/statement to be checked.
            - "custom_id": The unique ID for the job to identify the which is which.
        :return: The response from the Lean server, the format dict should be as follows:
            - "results": List of dict of verified results, the inner dict each dict contains:
                - "custom_id": The unique ID for the job.
                - "error": What error does the Lean server return, if no error, this should be None.
                - "response": The list of dict of the evaluated results
                - "time": The time it takes to verify
        """
        responses = self.Lean4_client.verify(job_list, timeout=timeout, max_concurrent_queries=192, asyc_verify=False, delete_temp_files=True, tmp_folder=tmp_folder)
        return responses
    
    def verify_proof(
            self, 
            proofs_to_verify: str, 
            id_list: Optional[List[str]] = None,
            timeout: int = 120, 
            original_statement: Optional[List[str]] = None
    ):
        """
        This function is used to verify the list of Lean4 proofs, it will firstly add the headers to the proof and then upload it to the server for verification.
        :param proofs_to_verify: List of Lean4 proofs to be verified, we assume that some of the proofs may be incorrect in format.
        :param id_list: Optional list of custom IDs for the proofs, if not provided, it will be generated from the proof content.
        :param timeout: The timeout for the verification, default is 120 seconds.
        :return: The list of dict of the verified results, the inner dict should contains:
            - "custom_id": The unique ID for the job.
            - "proof": The original Lean4 proof.
            - "pass": Whether there is an error in the code or verification.
            - "verify_result": The result of the verification, which is a dict containing:
                - "verified_code": The code that sent to the server for verification, including all the headers and imports.
                - "error": What error does the Lean server return, if no error, this should be None.
                - "response": The list of dict of the evaluated results
                - "time": The time it takes to verify
        """

        if original_statement is not None:
            assert len(original_statement) == len(proofs_to_verify), "The length of original_statement should be the same as proofs_to_verify"

        processed_proofs = []
        curr_proof = proofs_to_verify
        # for i, curr_proof in enumerate(proofs_to_verify):
        original_proof = copy.deepcopy(curr_proof)
        if type(curr_proof) is not str:
            curr_proof = "None"

        if "sorry" in curr_proof or "admit" in curr_proof or "apply?" in curr_proof:
            curr_proof = "None"
        if not curr_proof:
            curr_proof = "None"
        if curr_proof == "None":
            curr_id = f"wrong_proof"

        curr_time = str(time.time_ns())
        curr_id = f"{curr_proof}_{curr_time}"
        curr_id = (hashlib.sha256(curr_id.encode('utf-8'))).hexdigest()
        #curr_id = (hashlib.sha256(curr_proof.encode('utf-8'))).hexdigest()
        processed_proofs.append({
            "proof": curr_proof,
            "custom_id": curr_id,
            "original_proof": original_proof
        })
        verified_results = self._submit_job(processed_proofs, timeout=timeout, tmp_folder=curr_id)["results"]
        processed_results = []
        verified_results_dict = {
            result["custom_id"]: result for result in verified_results
        }

        for curr_proof_dict in processed_proofs:
            curr_id = curr_proof_dict["custom_id"]
            curr_verified_result = verified_results_dict[curr_id]
            pass_verification = True
            if curr_verified_result["error"] is not None or curr_verified_result["response"] is None:
                pass_verification = False
                processed_results.append({
                    "custom_id": curr_id, 
                    "proof": curr_proof_dict["original_proof"],
                    "pass": pass_verification,
                    "verify_result": curr_verified_result["response"]
                })
                continue
            try:
                for curr_msg in curr_verified_result["response"]["messages"]:
                    if curr_msg["severity"] == "error":
                        pass_verification = False
                        break
            except Exception as e:
                UserWarning(f"Error in processing the response: {e}")
                print(f"current verified result is: {curr_verified_result}")
                pass_verification = False
                processed_results.append({
                    "custom_id": curr_id, 
                    "proof": curr_proof_dict["original_proof"],
                    "pass": pass_verification,
                    "verify_result": curr_verified_result["response"]
                })
                continue
            
            processed_results.append({
                "custom_id": curr_id, 
                "proof": curr_proof_dict["original_proof"],
                "pass": pass_verification,
                "verify_result": curr_verified_result
            })
        return processed_results

    
# Lean_checker = LeanChecker(lean4_project_path="/u/hanning5/verl/verl/utils/reward_score/PhysLean", tmp_folder=".temp_verify_folder_conjecture")
Lean_checker = LeanChecker(lean4_project_path="/u/hanning5/verl/verl/utils/reward_score/PhysLean")
# Lean_checker.verify_proof("")

# def extract_solution(solution_str, method="strict"):
#     assert method in ["strict", "flexible"]

#     # Optimization: Regular expression matching on very long strings can be slow.
#     # For math problems, the final answer is usually at the end.
#     # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
#     if len(solution_str) > _SOLUTION_CLIP_CHARS:
#         solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

#     if method == "strict":
#         # this also tests the formatting of the model
#         solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
#         if len(solutions) == 0:
#             final_answer = None
#         else:
#             # take the last solution
#             final_answer = solutions[-1].replace(",", "").replace("$", "")
#     elif method == "flexible":
#         answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
#         final_answer = None
#         if len(answer) == 0:
#             # no reward is there is no answer
#             pass
#         else:
#             invalid_str = ["", "."]
#             # find the last number that is not '.'
#             for final_answer in reversed(answer):
#                 if final_answer not in invalid_str:
#                     break
#     return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    #print(solution_str)
    results = Lean_checker.verify_proof(solution_str)
    print(results[0]['pass'])
    if results[0]['pass']:
        return 1.0
    else:
        return 0.0
    # answer = extract_solution(solution_str=solution_str, method=method)
    # if answer is None:
    #     return 0
    # else:
    #     if answer == ground_truth:
    #         return score
    #     else:
    #         return format_score
