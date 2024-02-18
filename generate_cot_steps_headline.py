import openai
import json
from tqdm import tqdm

# Set the OpenAI API key
openai.api_key = ''

def get_response(messages):
    """
    Sends a request to the OpenAI API to generate a response based on the provided messages.
    """
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            temperature=0.3,
            max_tokens=1000,
            request_timeout=60  # Direct argument for readability
        )
        response = completion.choices[0].message.content.strip()
        return response
    except Exception as e:
        return "Error occurred: " + str(e)

def generate_message(instruction, new_article, new_question, new_calculation, new_ans):
    """
    Prepares the messages payload for the OpenAI API, including the instruction, article, question, chosen mathematical methods, and answer.
    """
    messages = [{
      "role": "system",
      "content": instruction
    }]

    new_content = "#Article:\n" + str(new_article) + "\n\n#Headline Sentence:\n" + str(new_headline) + "\n"

    messages.append({
      "role": "user",
      "content": new_content
    })

    return messages

def load_instructions(file_path):
    """
    Loads instructions from a given JSON file path.
    """
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            data = json.load(file)
            return data[1]["cot_steps_headline_generation"]  
    except (IOError, IndexError, json.JSONDecodeError) as e:
        print(f"Failed to load instructions: {e}")
        return ""

def process_data(input_file_path, output_file_path, instruction):
    """
    Processes the data by generating messages and fetching responses from the OpenAI API.
    """
    with open(input_file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)

    new_data = []
    failed_indices = []

    for index, item in enumerate(tqdm(data)):
        messages = generate_messaage(instruction, new_article=item['news'],new_headline=item['headline'])
        response = get_response(messages)

        if response.startswith("Error occurred:"):
            failed_indices.append(index)
        else:
            item["output"] = response

        new_data.append(item)

    if failed_indices:
        with open('failed_indices.txt', 'w') as f:
            for index in failed_indices:
                f.write(f"{index}\n")

    with open(output_file_path, 'w', encoding="utf-8") as file:
        json.dump(new_data, file, ensure_ascii=False, indent=4)

instruction_file_path = "data/Prompt/prompt.json"
input_file_path = "data/NumHG/Train_Headline_Generation.json"
output_file_path = "data/CoT-NumHG/Train_Headline_Generation_with_CoT.json"

instruction = load_instructions(instruction_file_path)
if instruction:
    process_data(input_file_path, output_file_path, instruction)
else:
    print("Processing aborted due to missing instructions.")