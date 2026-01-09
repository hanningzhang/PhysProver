import argparse
import json
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import re

def extract_code(text_input):
    """Extracts the last Lean 4 code block from the model's output."""
    try:
        matches = re.findall(r'```lean4\n(.*?)\n```', text_input, re.DOTALL)
        return matches[-1].strip() if matches else "No Lean 4 code block found."
    except Exception:
        return "Error during code extraction."
    
def main():
    parser = argparse.ArgumentParser(description="Run inference with vLLM on MATH")
    parser.add_argument("--model", type=str, default="AI-MO/Kimina-Prover-Distill-8B", help="Model name or path")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="tensor_parallel_size")
    parser.add_argument("--n", type=int, default=16, help="number of trajectory per sample")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")
    parser.add_argument("--dataset", type=str, default="physicslean_test.json", help="Dataset Path")
    parser.add_argument("--start", type=int, default=0, help="start of the dataset")
    parser.add_argument("--end", type=int, default=9999, help="end of the dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum new tokens to generate")
    parser.add_argument("--output_file", type=str, default="physicslean_kimina_8b_n16.json", help="Where to save results")
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset,'r') as f:
        data = json.load(f)[args.start:args.end]

    # Load tokenizer and LLM
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model,tensor_parallel_size=args.tensor_parallel_size, dtype = "float16",enforce_eager=True, gpu_memory_utilization=0.8,swap_space=32,trust_remote_code=True, max_model_len=4096*2)
    llm.llm_engine.tokenizer.tokenizer.add_bos_token = False
    # llm.llm_engine.tokenizer._tokenizer.add_bos_token = False     please try this if the above line does not work
    # Greedy decoding
    sampling_params = SamplingParams(n=args.n, temperature=args.temperature, max_tokens=args.max_tokens, top_p=0.95, seed=args.seed)

    results = []
    prompt_list = []
    gt_list = []
    for sample in data:
        prompt_list.append(sample['input'])
        gt_list.append(sample['ground_truth'])

    outputs = llm.generate(prompt_list, sampling_params)
    for i,output in enumerate(outputs):
        prompt = output.prompt
        generated_text = [output.outputs[i].text for i in range(len(output.outputs))]
        answers = generated_text
        outputs = []
        for j in range(len(answers)):
            outputs.append({"generation_idx":j+1,"generated_proof":extract_code(prompt + answers[j])})
        results.append({"Name":f"physlean_{i}", "Proof_generation_log": outputs})
        
    with open(args.output_file,'w') as f:
        json.dump(results,f,indent=4,ensure_ascii=False)


if __name__ == "__main__":
    main()
