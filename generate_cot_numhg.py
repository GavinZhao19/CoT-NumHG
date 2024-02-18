import json
from tqdm import tqdm

# Path to the original JSON file
input_json_file = 'data/NumHG_CoT_Steps/Train_Numerical_Reasoning_with_CoT.json'
# Path to the new JSON file
output_json_file = 'data/CoT-NumHG/Train_Numerical_Reasoning_with_CoT.json'
file_path = 'data/Prompt/prompt.json'

# Load the original JSON data
with open(input_json_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

with open(file_path, 'r', encoding="utf-8") as file:
    instruction = json.load(file)
instruction = instruction[2]["train_numerical_reasoning"] 

# Modify the data structure and create a new list of dictionaries
new_data = []
for item in tqdm(data):
    input_sentence = "#Article\n" + item["news"] + "\n\n#Question\nFill in the blank: " + item["masked headline"]
    output_sentence = "To fill in the blank in the question sentence: " + item["masked headline"] + "\n\n" + item["output"] + "\n\nSummary: math methods: " + item["calculation"] + "Answer: " + str(item["ans"])

    # Create a new dictionary
    new_dict = {
        "instruction": instruction,
        "input": input_sentence,
        "output": output_sentence
    }
    new_data.append(new_dict)

# Write the modified data to a new JSON file
with open(output_json_file, 'w', encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=4)
