# import json

# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Prover-V2-7B")

# with open("physicslean_test.json",'r')  as f:
#     test_data = json.load(f)

# data = []
# for sample in test_data:
#     length = len(tokenizer.encode(sample['input']))
#     if length < 4000:
#         tmp = {
#             "messages": [
#                 {"role": "user", "content": sample['input']},
#                 {"role": "assistant", "content": sample['ground_truth']+"\n```<｜end▁of▁sentence｜>"},
#             ]
#         }
#         data.append(tmp)

# with open("physicslean/test.json",'w')  as f:
#     json.dump(data[:50], f, indent=4)


# import json
# with open("physicslean_newsplit/train.json",'r')  as f:
#     train_data = json.load(f)

# sorted_list = sorted(train_data, key=lambda sample: len(sample['messages'][1]['content']))

# with open("physicslean_newsplit/sorted_train.json",'w')  as f:
#     json.dump(sorted_list, f, indent=4, ensure_ascii=False)



import json

with open("physicslean_kimina/test.json",'r') as f:
    data = json.load(f)

new_data = []
for sample in data:
    tmp = {
        "input" : sample['messages'][0]['content'],
        "ground_truth" : sample['messages'][1]['content']
    }
    new_data.append(tmp)
with open("physicslean_kimina_new_split_test.json",'w')  as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)