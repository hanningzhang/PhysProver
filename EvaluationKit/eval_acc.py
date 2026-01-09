import json
from utils import utils
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save_folder", default="physicslean_kimina_train_gen_from_claude_deepseek_max4_1e-5_bs32_2ep_ep2")
args = parser.parse_args()

data = utils.read_json_in_folder(f"{args.save_folder}/pass_proofs")
failed_data = utils.read_json_in_folder(f"{args.save_folder}/failed_proofs")
pass_count = 0
for i,sample in enumerate(data):
    log = sample['Proof_verification_log']
    for l in log:
        if l['pass']and l['proof'].strip() != "":
            pass_count += 1
            break

print("================================")
print(f"The accuracy of {args.save_folder}: {pass_count}/{len(data) + len(failed_data)}")


# import json
# from utils import utils
# import os
# import shutil

# shutil.rmtree('PhysLean/tmp')
# os.makedirs("PhysLean/tmp")
# data = utils.read_json_in_folder("claude45_conjecture_deepseek_gen_0_6000_full_negation/pass_proofs")
# for i,sample in enumerate(data):
#     log = sample['Proof_verification_log']
#     store = []
#     for l in log:
#         if l['pass']and l['proof'].strip() != "":
#             store.append(l['proof'])

#     if "¬(" in store[0]:
#         print(store[0])
#         break
    # if len(store) > 0:
    #     utils.write_to_file(f"PhysLean/tmp/physlean_{i}.lean",store[0])