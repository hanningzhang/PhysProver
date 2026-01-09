import nest_asyncio
from Lean4Client import Lean4Client
from utils import utils
from typing import List, Dict, Any, Optional
import re
import hashlib
from tqdm.auto import tqdm
import argparse
import copy
import os

def rstrip_space_newline_and_by(s: str) -> str:
    return re.sub(r'(?:[ \n]+|[ \n]*\bby\b[ \n]*)+$', '', s)



Lean4_headers = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

"""

"""
This class is based on remote query of Lean-server to automatically 
check whether the Lean4 statement may cause compilation errors or is the Lean4 theorem proof is correct. 
This class highly rely on inter-server communication and it needs the Lean-server to be started remotely.
"""
class LeanChecker:
    def __init__(
            self, 
            lean4_project_path: str,
            tmp_folder: str
    ):
        self.Lean4_client = Lean4Client(lean4_project_path=lean4_project_path, tmp_folder=tmp_folder)

    def _submit_job(
            self, 
            job_list: List[Dict], 
            timeout: int = 120
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
        responses = self.Lean4_client.verify(job_list, timeout=timeout, max_concurrent_queries=48, asyc_verify=True, delete_temp_files=True)
        return responses
    
    def verify_proof(
            self, 
            proofs_to_verify: List[str], 
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
        for i, curr_proof in enumerate(proofs_to_verify):
            original_proof = copy.deepcopy(curr_proof)
            if type(curr_proof) is not str:
                curr_proof = "None"

            # Make sure the proof does not change the statement
            if original_statement is not None:
                processed_statement = rstrip_space_newline_and_by(original_statement[i])
                if processed_statement not in curr_proof:
                    curr_proof = "None"
            if "sorry" in curr_proof or "admit" in curr_proof or "apply?" in curr_proof:
                curr_proof = "None"
            if not curr_proof:
                curr_proof = "None"
            if curr_proof == "None":
                curr_id = f"wrong_proof_{i}"
            else:
                if id_list is not None and i < len(id_list):
                    curr_id = id_list[i]
                else:
                    # generate a unique ID based on the proof content
                    curr_id = (hashlib.sha256(curr_proof.encode('utf-8'))).hexdigest()

            processed_proofs.append({
                "proof": curr_proof,
                "custom_id": curr_id,
                "original_proof": original_proof
            })

        verified_results = self._submit_job(processed_proofs, timeout=timeout)["results"]
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

    def verify_proof_dataset(
            self, 
            proof_dataset: List[Dict],
            timeout: int = 120,
            ckpt_path: str = "./ProofCheck_ckpt"
    ) -> List[Dict]:
        """
        This function verify whether the generated proofs can pass the Lean verifier. It will also save the log for each proof.
        :param proof_dataset: List of dict of the proof dataset, the inner dict should at least contains:
            - "Name": The name of the theorem
            - "Statement": The Lean4 statement of the theorem.
            - "Natural_language_statement": The informal natural language statement of the theorem to proof.
            - "Generated_proof": The list of generated Lean4 proof of the theorem to be verified.
            - "Proof_generation_log": The log of the proof generation, which is a list of dict where the inner dict should contains:
                - "generation_idx": The unique ID for the generation.
                - "generated_content": The inpiut-output pair of the generation.
                - "generated_proof": The generated Lean4 proof. For easy verification, we should use this section as our verified proof.
        :param batch_size: The batch size for the verification, it depend on the power of CPU.
        :param timeout: The timeout for the verification, default is 120 seconds.
        :param ckpt_path: The path to save the checkpoint of the verified proofs, default is "./ProofCheck_ckpt".
        :return: The list of dict of the verified results, the inner dict contains:
            - "Name": The name of the theorem
            - "Statement": The original Lean4 statement of the theorem.
            - "Natural_language_statement": The informal natural language statement of the theorem to proof.
            - "pass_rate": How many proofs can pass the Lean verifier.
            - "Generated_proof": The list of generated Lean4 proof of the theorem to be verified.
            - "Proof_verification_log": The log of the proof verification, which is a list of dict containing:
                - "custom_id": The unique ID for the job.
                - "proof": The original Lean4 proof.
                - "pass": Whether there is an error in the code or verification.
                - "verify_result": The result of the verification, which is a dict containing:
                    - "verified_code": The code that sent to the server for verification, including all the headers and imports.
                    - "error": What error does the Lean server return, if no error, this should be None.
                    - "response": The list of dict of the evaluated results
                    - "time": The time it takes to verify
        """

        utils.check_folder_exit(f"{ckpt_path}/pass_proofs")
        utils.check_folder_exit(f"{ckpt_path}/failed_proofs")

        reference_proof_dataset = copy.deepcopy(proof_dataset)

        for i in tqdm(range(len(reference_proof_dataset))):
            # if i < 63:
            #     continue
            print(f"Verifying proof {i + 1}/{len(reference_proof_dataset)}: {reference_proof_dataset[i]['Name']}")
            curr_proof_ls = [curr_record["generated_proof"] for curr_record in reference_proof_dataset[i]["Proof_generation_log"]]
            # get the original statement
            original_statement = reference_proof_dataset[i].get("Statement", None)
            if original_statement is not None:
                original_statement = [original_statement] * len(curr_proof_ls)
            else:
                original_statement = None
            id_ls = [curr_record["generation_idx"] for curr_record in reference_proof_dataset[i]["Proof_generation_log"]]
            varification_results = self.verify_proof(curr_proof_ls, timeout=timeout, original_statement=original_statement, id_list=id_ls)
            reference_proof_dataset[i]["Proof_verification_log"] = varification_results
            reference_proof_dataset[i]["pass_rate"] = sum(
                [1 if curr_result["pass"] else 0 for curr_result in varification_results]
            ) / len(varification_results)
            if reference_proof_dataset[i]["pass_rate"] == 0:
                print(f"Warning: The proof {reference_proof_dataset[i]['Name']} has no valid proofs.")
                utils.write_to_json(
                    f"{ckpt_path}/failed_proofs/{reference_proof_dataset[i]['Name']}.json", 
                    reference_proof_dataset[i]
                )
            else:
                print(f"Info: The proof {reference_proof_dataset[i]['Name']} has valid proofs with pass rate: {reference_proof_dataset[i]['pass_rate'] * 100:.2f}%")
                utils.write_to_json(
                    f"{ckpt_path}/pass_proofs/{reference_proof_dataset[i]['Name']}.json", 
                    reference_proof_dataset[i]
                )

        # summarize the results
        print(f"Total proofs: {len(reference_proof_dataset)}")
        total_passed_proofs = sum(
            [1 if curr_record["pass_rate"] > 0 else 0 for curr_record in reference_proof_dataset]
        )
        print(f"Total passed proofs: {total_passed_proofs} / {len(reference_proof_dataset)}")
        print(f"Pass rate: {total_passed_proofs / len(reference_proof_dataset) * 100:.2f}%")
        return reference_proof_dataset


    
    def verify_statement(
            self, 
            statements_to_verify: List[str], 
            timeout: int = 120
    ) -> List[Dict]:
        """
        This function is used to verify the list of Lean4 statements, 
        it will firstly extract the last Lean4 statements in the string, then uopload it to the server for verification.
        :param statements_to_verify: List of Lean4 statements to be verified.
        :param timeout: The timeout for the verification, default is 120 seconds.
        :return: The list of dict of the verified results, the inner dict should contains:
            - "custom_id": The unique ID for the job.
            - "Statement": The original Lean4 statement.
            - "pass": Whether there is an error in the code or verification.
            - "verify_result": The result of the verification, which is a dict containing:
                - "verified_code": The code that sent to the server for verification, including all the headers and imports.
                - "error": What error does the Lean server return, if no error, this should be None.
                - "response": The list of dict of the evaluated results
                - "time": The time it takes to verify
        """

        processed_statements = []
        for i, statement in enumerate(statements_to_verify):
            if type(statement) is not str:
                last_theorem_block = "None"
            else:
                # 匹配以 'theorem' 开头，直到 ':= by' 之间的内容（非贪婪匹配）
                matches = re.findall(r'(theorem.*?:= by)', statement, re.DOTALL)
                # 获取最后一个 match
                last_theorem_block = matches[-1] if matches else "None"
                last_theorem_block += " sorry"
            
            processed_theorem = Lean4_headers + last_theorem_block
            
            if last_theorem_block == "None":
                curr_id = f"wrong_statement_{i}"
            else:
                curr_id = (hashlib.sha256(processed_theorem.encode('utf-8'))).hexdigest()
            processed_statements.append({
                "proof": processed_theorem,
                "custom_id": curr_id, 
                "original_statement": statement
            })

        verified_results = self._submit_job(processed_statements, timeout=timeout)["results"]
        processed_results = []

        verified_results_dict = {
            result["custom_id"]: result for result in verified_results
        }

        for curr_statement_dict in processed_statements:
            curr_id = curr_statement_dict["custom_id"]
            curr_verified_result = verified_results_dict[curr_id]
            pass_verification = True
            if curr_verified_result["error"] is not None:
                pass_verification = False
            try:
                for curr_msg in curr_verified_result["response"]["messages"]:
                    if curr_msg["severity"]  == "error":
                        pass_verification = False
                        break
            
                processed_results.append({
                    "custom_id": curr_id, 
                    "Statement": curr_statement_dict["original_statement"],
                    "pass": pass_verification,
                    "verify_result": curr_verified_result["response"]["messages"]
                })
            except Exception as e:
                print(f"Have some unknown error: {e}, skipping this statement")

        return processed_results
    
    def verify_statement_dataset(
            self, 
            statement_dataset: List[Dict], 
            batch_size: int = 128,
            timeout: int = 120, 
            ckpt_path: str = "./StatementCheck_ckpt"
    ) -> List[Dict]:
        """
        This function verify whether the generated statement can pass Lean complier.
        :param statement_dataset: List of dict of the statement dataset, the inner dict should at least contains:
            - "Name": The name of the statement
            - "Statement": The Lean4 statement to be verified.
            - "batch_size": The batch size for the verification,it depend on the power of CPU.
            - "Generation_log": List of dict of the generation log, it should contains "Statement_fusion_log" and "Autoformalization_log"
        :param timeout: The timeout for the verification, default is 120 seconds.
        :return: The list of dict of the verified results, the inner dict should contains:
            - "Name": The name of the statement
            - "Statement": The original Lean4 statement
            - "Statement_pass": Whether the statement can pass the Lean complier
            - "Generation_log": The generation log of the statement, which is a dict containing:
                - "Statement_fusion_log": The log of the statement fusion
                - "Autoformalization_log": The log of the autoformalization
                - "Statement_verification_log": The log of the statement verification, which is a list of dict containing:
                    - "custom_id": The unique ID for the job.
                    - "Statement": The original Lean4 statement.
                    - "pass": Whether there is an error in the code or verification.
                    - "verify_result": The result of the verification, which is a dict containing:
                        - "verified_code": The code that sent to the server for verification, including all the headers and imports.
                        - "error": What error does the Lean server return, if no error, this should be None.
                        - "response": The list of dict of the evaluated results
                        - "time": The time it takes to verify
        """
        utils.check_folder_exit(f"{ckpt_path}/pass_statements")
        utils.check_folder_exit(f"{ckpt_path}/fail_statements")
        batched_statement_dataset = utils.split_batch(statement_dataset, batch_size=batch_size)
        all_verified_dataset = []

        for curr_batch in tqdm(batched_statement_dataset):
            curr_statement_ls = [curr_statement_record["Statement"] for curr_statement_record in curr_batch]
            curr_verified_result = self.verify_statement(curr_statement_ls, timeout=timeout)
            for i, curr_statement_record in enumerate(curr_batch):
                curr_processed_record = {}
                curr_verified_record = curr_verified_result[i]
                for k, v in curr_statement_record.items():
                    if k != "Generation_log":
                        curr_processed_record[k] = v
                curr_processed_record["Statement_pass"] = curr_verified_record["pass"]
                curr_processed_record["Generation_log"] = curr_statement_record["Generation_log"]
                curr_processed_record["Generation_log"]["Statement_verification_log"] = curr_verified_record
                all_verified_dataset.append(curr_processed_record)

                if curr_processed_record["Statement_pass"] == False:
                    utils.write_to_json(
                        f"{ckpt_path}/fail_statements/{curr_processed_record['Name']}.json", 
                        curr_processed_record
                    )
                else:
                    utils.write_to_json(
                        f"{ckpt_path}/pass_statements/{curr_processed_record['Name']}.json", 
                        curr_processed_record
                    )
        return all_verified_dataset


def main():
    parser = argparse.ArgumentParser(description="Statement checker initiate program")
    parser.add_argument("--verify_type", type=str, choices=["proof", "statement"], default="statement", help="The type of verification to be done, it can be 'proof' or 'statement'.")
    parser.add_argument("--dataset_path", type=str, help="The path to json file whch stores required dataset or it could be a folder containing all the json files, it can be organized as a list that has similar effects as the json file.")
    parser.add_argument("--Lean4_server_url", type=str, help="The url for Lean4 server, it currently can only be a local server and unable to query remote server (even if you have done port forward)")
    parser.add_argument("--save_path", type=str, default="./StatementCheck_ckpt/verified_results.json", help="The json file's path to save the verified results.")
    parser.add_argument("--ckpt_path", type=str,default="./StatementCheck_ckpt", help="Where the statements should be stored")
    parser.add_argument("--batch_size", type=int, default=5, help="The batch size for the verification, it depend on the power of CPU.")
    parser.add_argument("--timeout", type=int, default=120, help="The timeout for the verification, default is 120 seconds.")
    parser.add_argument("--tmp_folder", type=str, default=".temp_verify_folder")
    args = parser.parse_args()

     # check whether the dataset_path is a folder or a json file
    if os.path.isdir(args.dataset_path):
        dataset_to_verify = utils.read_json_in_folder(args.dataset_path)
    else:
        dataset_to_verify = utils.read_from_json(args.dataset_path)
    Lean_checker = LeanChecker(lean4_project_path=args.Lean4_server_url, tmp_folder=args.tmp_folder)
    dataset_to_verify = dataset_to_verify[:]
    if args.verify_type == "statement":
        utils.check_folder_exit(args.ckpt_path)
        verified_results = Lean_checker.verify_statement_dataset(
            dataset_to_verify, 
            ckpt_path=args.ckpt_path,
            batch_size=args.batch_size,
            timeout=args.timeout
        )
    elif args.verify_type == "proof":
        utils.check_folder_exit(args.ckpt_path)
        verified_results = Lean_checker.verify_proof_dataset(
            dataset_to_verify, 
            ckpt_path=args.ckpt_path,
            timeout=args.timeout
        )
        print(f"finish evaluation on dataset {str(args.dataset_path)}")
    utils.write_to_json(args.save_path, verified_results)

# Testing main
if __name__ == "__main__":
    main()
