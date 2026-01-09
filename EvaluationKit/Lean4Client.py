from utils import utils
from typing import List, Tuple, Dict
import os
import subprocess
import re
import shutil

class Lean4Client:

    def __init__(
            self, 
            lean4_project_path: str,
            tmp_folder = ".temp_verify_folder"
        ):
        """
        The naive Lean4 client that directly runs Lean4 commands via terminal and returns the outputs.
        """

        self.lean4_project_path = lean4_project_path
        self.tmp_folder = tmp_folder
        # utils.check_folder_exit(os.path.join(self.lean4_project_path, ".temp_verify_folder"))
        utils.check_folder_exit(os.path.join(self.lean4_project_path, self.tmp_folder))

    def _crearte_temp_file(
            self,
            codes: List[Dict]
    ) -> List[Dict]:
        """
        Create temporary files for the given Lean4 codes.
        :param codes: List of dict for Lean4 codes to be verified, each dict contains the following keys:
            - "proof": The Lean4 code to be verified
            - "custom_id": The custom id for the code, it will be used as the name for the temp file if we check it is indeed unique in the codes list. Otherwise, we will give it a random uuid as the file name.
        :return: Updated codes with an additional key "temp_file_path" for each code, which is the path to the temporary file created.
        """

        #temp_folder = os.path.join(self.lean4_project_path, ".temp_verify_folder")
        temp_folder = os.path.join(self.lean4_project_path, self.tmp_folder)
        utils.check_folder_exit(temp_folder, print_info=False)

        for curr in codes:
            curr_temp_file_path = os.path.join(temp_folder, f"{curr['custom_id']}.lean")
            utils.write_to_file(curr_temp_file_path, curr["proof"])
            curr["temp_file_path"] = curr_temp_file_path

        return codes
    
    def _formulate_bash_command(
            self, 
            codes: List[Dict], 
            timeout: int,
            max_memory_per_query: int = 1024
    ) -> List[Dict]:
        """
        Formulate the bash commands for verifying the given Lean4 codes.
        :param codes: List of dict for Lean4 codes to be verified, each dict contains the following keys:
            - "proof": The Lean4 code to be verified
            - "custom_id": The custom id for the code, it will be used as the name for the temp file if we check it is indeed unique in the codes list. Otherwise, we will give it a random uuid as the file name.
            - "temp_file_path": The path to the temporary file created for the code.
        :return: Updated codes with an additional key "bash_command" for each code, which is the bash command to be run for verification.
        """

        for curr in codes:
            curr_bash_command = f"cd {self.lean4_project_path} && lake env lean -Dweak.timeout={timeout} -Dweak.max_memory={max_memory_per_query} {curr['temp_file_path']}"
            curr["bash_command"] = curr_bash_command

        return codes
    
    def _run_bash_command(
            self, 
            bash_command: str
    ) -> Tuple[str, str]:
        """
        Run the given bash command and return the output and error messages.
        :param bash_command: The bash command to be run.
        :return: A tuple containing the output and error messages.
        """

        process = subprocess.Popen(bash_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        return output.decode("utf-8"), error.decode("utf-8")
    
    def _batched_run_commands(
        self, 
        codes_to_run: List[str],
        asyc_verify: bool = True,
        max_concurrent_queries: int = 32
    ) -> List[Tuple[str, str]]:
        
        if not asyc_verify:
            results = []
            for cmd in codes_to_run:
                result = self._run_bash_command(cmd)
                results.append(result)
            return results

        # 异步执行
        import asyncio
        from asyncio.subprocess import PIPE, create_subprocess_shell

        # async def run_command(cmd):
        #     proc = await create_subprocess_shell(cmd, stdout=PIPE, stderr=PIPE)
        #     stdout, stderr = await proc.communicate()
        #     return stdout.decode(), stderr.decode()
        async def run_command(cmd):
            proc = await create_subprocess_shell(cmd, stdout=PIPE, stderr=PIPE)
            try:
                # Add timeout (e.g., 5 minutes per command)
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                return stdout.decode(), stderr.decode()
            except asyncio.TimeoutError:
                proc.kill()  # Kill the hung process
                await proc.wait()  # Clean up
                return "ERROR", "ERROR: Command timed out"

        async def run_commands_in_batches(cmds, batch_size):
            semaphore = asyncio.Semaphore(batch_size)

            async def sem_run(cmd):
                async with semaphore:
                    return await run_command(cmd)

            tasks = [sem_run(cmd) for cmd in cmds]
            return await asyncio.gather(*tasks)

        # 智能处理事件循环
        try:
            # 尝试获取当前运行的循环（Jupyter环境）
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 没有运行的循环，使用 asyncio.run
            return asyncio.run(run_commands_in_batches(codes_to_run, max_concurrent_queries))
        
        # 在 Jupyter 中，使用 nest_asyncio
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(run_commands_in_batches(codes_to_run, max_concurrent_queries))
        except ImportError:
            # 没有 nest_asyncio，回退到同步执行
            print("Warning: nest_asyncio not installed. Falling back to synchronous execution.")
            print("Install it with: pip install nest_asyncio")
            results = []
            for cmd in codes_to_run:
                result = self._run_bash_command(cmd)
                results.append(result)
            return results


    def parse_lean4_output(
            self, 
            std_out: str, 
            std_err: str
    ) -> Dict:
        """
        Parse the output and error messages from Lean4 verifier.
        :param std_out: The standard output from the Lean4 verifier.
        :param std_err: The standard error from the Lean4 verifier.
        :return: A dict containing the parsed messages and full log.
            - "messages": List of dicts representing the messages from the Lean4 verifier:
                - "severity": The severity of the message, can be "info", "warning", "error"
                - "line": The line number of the message
                - "column": The column number of the message
                - "data": The content of the message
            - "full_log": The full log of the verification process as a string
        """

        # check std_err to see whether the programs exists with non-zero code


        messages = []
        pattern = r'^(.*?):(\d+):(\d+):\s*(error|warning|info):\s*(.*)$'

        lines = std_out.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            match = re.match(pattern, line)
            if match:
                file_path = match.group(1)
                line_num = int(match.group(2))
                col_num = int(match.group(3))
                severity = match.group(4)
                message_start = match.group(5)

                # collect information of the msg
                message_lines = [message_start]
                context_lines = []
                i += 1

                # Judge whether there are any context information
                in_context = False
                while i < len(lines):
                    next_line = lines[i]

                    # stop if we reach another message
                    if re.match(pattern, next_line):
                        break

                    # check the context
                    if next_line.strip() == '' and i + 1 < len(lines):
                        in_context = True
                        i += 1
                        continue
                    
                    if in_context:
                        context_lines.append(next_line)
                    else:
                        message_lines.append(next_line)
                    
                    i += 1
                full_message = '\n'.join(message_lines).strip()
                context = '\n'.join(context_lines).strip() if context_lines else None
                messages.append({
                    "severity": severity,
                    "line": line_num,
                    "column": col_num,
                    "data": full_message,
                    "context": context
                })
            else:
                i += 1
        return {
            "messages": messages,
            "full_log": std_out.strip() + '\n' + std_err.strip()
        }



    def verify(
            self, 
            codes: List[Dict], 
            timeout: int, 
            max_memory_per_query: int = 1024, 
            asyc_verify: bool = True, 
            max_concurrent_queries: int = 32, 
            delete_temp_files: bool = True
    ) -> Dict[str, List[Dict]]:
        """
        Verify the given Lean4 codes via terminal commands. It will firstly create temp files for each code, then cd to the root of the lean4 project path 
        and run the Lean4 verification command via terminal, the command will be `lake env lean

        :param codes: List of dict for Lean4 codes to be verified, each dict contains the following keys:
            - "proof": The Lean4 code to be verified
            - "custom_id": The custom id for the code, it will be used as the name for the temp file if we check it is indeed unique in the codes list. Otherwise, we will give it a random uuid as the file name.
        :param timeout: Timeout for each verification query
        :param max_memory_per_query: Maximum memory (in MB) allowed for each verification query
        :param asyc_verify: Whether to verify the codes asynchronously, if True, the code will verify multiple codes concurrently under the limit of max_concurrent_queries
        :param max_concurrent_queries: Maximum number of concurrent verification queries when asyc_verify is True.
        :param delete_temp_files: Whether to delete the temporary files after verification.
        :return: Dict with a single key "results", the value is a list of dicts for each code verification result, each dict contains the following keys:
            - "custom_id": The custom id for the code
            - "error": The error message if the verification failed, otherwise None
            - "verified_proof": The verified Lean4 code that sent to verification.
            - "response": The response message from the Lean4 verifier, it should be a dict contains the following keys:
                - "messages": List of dicts representing the messages from the Lean4 verifier:
                    - "severity": The severity of the message, can be "info", "warning", "error"
                    - "line": The line number of the message
                    - "column": The column number of the message
                    - "data": The content of the message
                - "full_log": The full log of the verification process as a string
        """

        # see whether the custom ids are unique
        custom_ids = [code["custom_id"] for code in codes]
        if len(custom_ids) != len(set(custom_ids)):
            print("Warning: The custom ids are not unique, we will use random uuid as the file names for the temp files.")
            for curr in codes:
                curr["custom_id"] = utils.hash_dict(curr)

        # create temp files
        codes_with_temp_files = self._crearte_temp_file(codes)

        # formulate bash commands
        codes_with_bash_commands = self._formulate_bash_command(codes_with_temp_files, timeout, max_memory_per_query)

        # run bash commands in batches
        bash_commands = [code["bash_command"] for code in codes_with_bash_commands]
        bash_results = self._batched_run_commands(bash_commands, asyc_verify, max_concurrent_queries)

        # parse results
        results = []
        for idx, (std_out, std_err) in enumerate(bash_results):
            parsed_response = self.parse_lean4_output(std_out, std_err)
            error_msg = []
            for msg in parsed_response["messages"]:
                if msg["severity"] == "error":
                    error_msg.append(msg["data"])
            if len(error_msg) == 0:
                error_msg = None
            results.append({
                "custom_id": codes_with_bash_commands[idx]["custom_id"],
                "error": error_msg,
                "verified_proof": codes_with_bash_commands[idx]["proof"],
                "response": parsed_response
            })
        # delete temp files
        if delete_temp_files:
            # temp_folder = os.path.join(self.lean4_project_path, ".temp_verify_folder")
            temp_folder = os.path.join(self.lean4_project_path, self.tmp_folder)
            for curr in codes_with_bash_commands:
                temp_file_path = curr["temp_file_path"]
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            # remove the temp folder itself
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder,ignore_errors=True)
        return {"results": results}