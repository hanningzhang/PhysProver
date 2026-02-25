import json
from tqdm import tqdm
from transformers import AutoTokenizer
from openai import OpenAI
from pydantic import BaseModel
from anthropic import Anthropic, transform_schema
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=99999999)
parser.add_argument("--dataset_path", type=str, default="seed_data_prompt.json")
parser.add_argument("--output_file", type=str, default="claude_formal_new_prompt.json")
args = parser.parse_args()

class Conjecture(BaseModel):
    conjecture: str

class NaturalLanguageStatement(BaseModel):
    natural_language: list[Conjecture]

client = OpenAI(
    base_url='https://api.nuwaapi.com/v1',
    api_key=''
)

def call_gpt(messages):

    response = None
    try:
        response = client.chat.completions.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=16384,
            temperature=1.0,
            messages = messages,
        )
    except Exception as e:
        print(e)
    # )

    return response

user_prompt = """You are an expert in mathematics, physics and Lean 4.
You are provided a context, a lemma, and a proof. Your task is to generate a list of 10 related physics conjecture in formal language based on the context and the seed language statements.

The conjectures should be:
1. A meaningful variant of the original theorem: modify hypotheses, generalize structures, or extend scope while keeping the core mathematical insight.
2. Must differ significantly in mathematical content (changed assumptions, stronger/weaker conclusions, or different algebraic structures) but remain recognizably related.
3. The new conjecture should be in formal language.
4. Do not include the proof.

When generating the conjectures, preserve all specific Lean identifiers exactly as they appear in the formal statement. You can also refer to the original formal statement.

Context:
{context}

Natural Language Statement:
{nq}

Original Formal Statement:
{theorem}

Return the final conjectures in JSON format as a dictionary where:
- The key is "conjectures"
- The value is a list of dictionaries
- Each dictionary in the list has a key "statement" whose value is a string containing one conjecture

Please generate a list of conjectures.
"""

with open(args.dataset_path,'r') as f:
    prompt_list = json.load(f)[args.start:args.end]

collected_data = []
for prompt in tqdm(prompt_list):
    response = call_gpt(prompt['prompt'])
    if not response:
        continue

    response_text = response.choices[0].message.content.strip()
    json_pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, response_text)
    if match:
        try:
            json_string = match.group(1)
            json_object = json.loads(json_string)
            prompt['formal_language'] = json_object
            collected_data.append(prompt)
        except:
            continue
    else:
        try:
            json_object = json.loads(response_text)
            prompt['formal_language'] = json_object
            collected_data.append(prompt)
        except:
            continue

    with open(args.output_file,'w') as f:
        json.dump(collected_data,f,indent=4)