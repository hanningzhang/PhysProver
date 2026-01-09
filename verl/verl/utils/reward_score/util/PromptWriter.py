"""
Set of helper functions for writing prompts
"""
from utils import utils
from transformers import PreTrainedTokenizer
from typing import List, Dict

Lean4_HEADER = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat
"""

KIMINA_PROVER_SYS_PROMPT = """You are an expert in mathematics and Lean 4."""

OpenR1_SYS_PROMPT = """You are an expert in Lean4 theorem proving with exceptional strategic reasoning abilities. When solving problems, strictly follow this process:
1. First create a natural language proof draft explaining key insights and logical steps
2. Then analyze required Lean4 tactics, specifying exact syntax and their logical purpose"""


def formulate_prompt_autoformalization_Goedel(
        name: str,
        problem_to_autoformalize: str,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = None, 
):
    """
    This is the helper function for writing the prompt to autoformalize the problem into Lean4 code using Goedel-Autoformalizer model.
    """

    user_prompt = f"""Please autoformalize the following natural language problem statement in Lean 4. Use the following theorem name: {name}
The natural language statement is: {problem_to_autoformalize}"""

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    return text


def write_regenerate_prompt(
        failed_response: str, 
        original_statement: str, 
        NL_statement: str = None
):
    """
    The function write the regeneration prompt for the failed response (majorly because the original statement is modified during the last generation)
    :param failed_response: The failed response from last round of generation, we assume th original prompt as well as all the formatting is included in the failed response
    :param original_statement: The original statement that should be proved.
    :param NL_statement: The natural language statement of the theorem, if None, we will not include it in the prompt.
    """

    original_statement = utils.rstrip_space_newline_and_by(original_statement)
    # The helper function that removes all the content after the last code block using ```lean4
    def _remove_last_generation(text: str):
        return text[:text.rfind("```lean4")]
    
    cleaned_response = _remove_last_generation(failed_response)
    cleaned_response += f"""
```lean4
"""
    cleaned_response += f"""{utils.preprocess_theorem_statement(original_statement)}"""
    return cleaned_response

def formulate_prompt_proof_writing_Goedel_V2_verification_mode(
        NL_statement: str,
        Lean4_statement: str,
        tokenizer: PreTrainedTokenizer
):
    """
    This function formulates the prompt for proof writing using Goedel V2 in verification mode. Different from the 
    normal mode, in verification mode, we will add an extra instruction to prevent the model from modifying the statement.
    :param NL_statement: The natural language statement of the theorem.
    :param Lean4_statement: The Lean4 statement of the theorem.
    :param tokenizer: The tokenizer to be used for formatting the prompt.
    """
    prompt = f"""Complete the following Lean 4 code:

```lean4
{Lean4_HEADER}

{utils.preprocess_theorem_statement(Lean4_statement)}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def formulate_prompt_proof_writing_Goedel_V2(
        NL_statement: str, 
        Lean4_statement: str, 
        tokenizer: PreTrainedTokenizer
):
    """
    This function formulates the prompt for proof writing using Goedel V2.
    :param NL_statement: The natural language statement of the theorem.
    :param Lean4_statement: The Lean4 statement of the theorem.
    :param tokenizer: The tokenizer to be used for formatting the prompt.
    :return: The formatted prompt string.
    """
    prompt = f"""Complete the following Lean 4 code:

```lean4
{Lean4_HEADER}


/-- {NL_statement} -/
{utils.preprocess_theorem_statement(Lean4_statement)}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def formulate_prompt_autoformalization_Kimina(
        name: str,
        problem_to_autoformalize: str,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = "You are an expert in mathematics and Lean 4.", 
    ) -> str:
    """
    This is the function for writing prompt to autoformalize the problem into Lean4 code.
    """

    prompt = f"""Please combine the following theorems into a more advanced theorem. Use the following theorem names: {name}

{problem_to_autoformalize}"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    return text


def formulate_prompt_fuse_statements(
        base_statement_ls: List[str],
        tokenizer: PreTrainedTokenizer, 
        skip_thinking: bool = False
    ) -> str:
    """
    The function to formulate the prompt for fusing two statements.
    :param base_statement_ls: A list of two base statements to be fused, it should contain at least two strings
    :param tokenizer: The tokenizer to be used for formatting the prompt
    :param skip_thinking: Whether to skip the thinking process in the prompt
    :return: the prompt for fusing two statements
    """
    assert len(base_statement_ls) >= 2, "At least two base statements are required to fuse."
    problem = f"""You are an expert mathematics educator skilled in problem design. Your task is to combine miultiple given problems into a SINGLE, more challenging problem by combining their key elements. Follow these steps: Please firstly do the following steps as your analysis process enclosed within <analysis></analysis>.
1. Analysis the points of knowledge that needed to be used when solving the proof problem and identify overlapping or complementary aspects (e.g., shared topic areas or constrasting difficulty levels)
2. Draft the new problems that integrates at least 2 key components from each original problems and make sure the new problem requires multi-step reasoning (e.g., combining algebraic manipulation with probabilistic analysis). Also, your combined problem should have non-trivial extension.
3. Additionally, you should make sure that the new problem is solvable.

After your analysis, you should put the new problem into a md code block. The new problem should be a SINGLE proof problem and you should not give the solution to the problem.

Here are the problems you need to fuse:"""
    for i, base_statement in enumerate(base_statement_ls):
        problem += f"\n\nProblem {i+1}:\n```md\n{base_statement}\n```"
    messages = [
        {"role": "system", "content": "You are an expert in mathematics and are good at problem design and reformulation."},
        {"role": "user", "content": problem}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    if skip_thinking:
        text += "\n<think>\n\n</think>\n"
    return text
    

def formulate_postLongCoT_prompt(
        post_LongCoT_content: str,
        NL_statement: str,
        FL_statement: str,
        output_begin_sign: str = "",
        pre_Lean4_codeblock_content: str = "",
        pre_statement_content: str = Lean4_HEADER,
) -> str:
    """
    The function to formulate post Long CoT prompt to perform Long CoT control generation. It's major usage is to boost the performance of Lean4 inference
    :param post_LongCoT_content: the previous sequence before the output of Long CoT content, it should contain the Long CoT stop sign
    :param NL_statement: Natural language statement for the theorem
    :param FL_statement: Lean4 statement for the theorem
    :param output_begin_sign: begin sign for output, for DeepSeek-R1, it should be none
    :param pre_Lean4_codeblock_content: The content before lean4 statement code-block
    :param pre_statement_content: the content before the starting of Lean4 statement that we need to prove
    :return: formatted post Long CoT prompt to query the model
    """
    return f"""{post_LongCoT_content}
{output_begin_sign}
{pre_Lean4_codeblock_content}
```lean4
{pre_statement_content}

/--{NL_statement}-/
{utils.preprocess_theorem_statement(FL_statement)}
"""

def formulate_prompt_LoTSolver(
        NL: str,
        thm_name: str,
        FL_statement: str,
        system_prompt: str=OpenR1_SYS_PROMPT
) -> str:
    """
    This function formulate the prompt for the LoT-Solver
    :param NL: NL statement of theorem
    :param thm_name: Name of theorem
    :param FL_statement: FL statement
    :return: formulated prompt
    """
    prompt =  f"""<｜begin▁of▁sentence｜>{system_prompt}
### Instruction: Please solve this Lean4 problem by completing both sections below:

NL Proof Draft (Natural Language)
Explain the proof strategy using mathematical reasoning and high-level steps. Consider:
- Key lemmas/theorems to apply
- Structural decomposition approaches
- Critical logical dependencies
- Potential proof patterns/methods

Lean4 Tactics Analysis (Technical Specification)
Identify concrete tactics needed to implement the proof, including:
1. Tactic name and syntax template
2. Purpose within proof context 
3. Expected goal state before/after application
4. Alternative tactics considered
"""
    prompt += f"""@ Natural language theorem statement:
{thm_name}:
{NL}

@ Lean4 theorem statement:
```lean4
{utils.preprocess_theorem_statement(FL_statement)}
```&


@ Lean4 theorem statement and proof with explanatory comments preceding each line of code:
### Response:
<think>
Okay, I should do the following:

  1. Provide the natural language analysis for the theorem based on the Natural language theorem statement.

  2. Draft the Lean4 tactics I should use to solve the problem

The user also asks that I should avoid using the keyword `sorry` or `admit` to give up the proof, so I will not write it in my Lean4 code.

The `{thm_name}` can be proofed by"""
    return prompt

def formulate_prompt_Kimina_Prover(
        NL: str,
        FL_statement: str,
        system_prompt: str = KIMINA_PROVER_SYS_PROMPT
):
    """
    The helper function for formulate the prompt for the Kimina-Prover.

    Args:
        NL: the natural language statement
        FL_statement: the formal language statement
        use_few_shot: whether to use few-shot learning
        system_prompt: the system prompt for the model

    Returns:
        The prompt for the model in string format
    """
    if FL_statement.endswith("sorry"):
        FL_statement = FL_statement[:-len("sorry")]
    prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
Think about and solve the following problem step by step in Lean 4.

```lean4
{Lean4_HEADER}

/-- {NL}-/
{utils.preprocess_theorem_statement(FL_statement)}
```
<|im_end|>
<|im_start|>assistant
"""
    return prompt

def formulate_prompt_gemini_basic(input_dict: Dict):
    """
    The basic function that only demostrates how to formulate, it uses the same prompt as Goedel-Prover-V2. 
    :param input_dict: The input dict that contains the min elements:
        - "Natural_language_statement" or "Informal_statement": The NL statement of the theorem.
        - "Statement": The Lean4 statement of the theorem.
        - "Name": Unique identifier for the theorem.
    """

    curr_NL_statement = input_dict.get("Natural_language_statement", "")
    curr_FL_statement = input_dict["Statement"]

    return f"""Complete the following Lean 4 code:

```lean4
{Lean4_HEADER}


/-- {curr_NL_statement} -/
{utils.preprocess_theorem_statement(curr_FL_statement)}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""
