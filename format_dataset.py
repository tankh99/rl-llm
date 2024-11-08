from datasets import load_dataset
import json
from constants import jailbreak_prompt_template

def format_jailbreak_instruction(examples, EOS_TOKEN):
    harmful_questions, jailbreak_prompts, responses = examples["original_harm_behavior"], examples["nested_prompt"], examples["claude2_output"]
    texts = []
    for harmful_question, jailbreak_prompt in zip(harmful_questions, jailbreak_prompts):
        text = jailbreak_prompt_template.format(harmful_question, jailbreak_prompt) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

def load_jailbreak_dataset(EOS_TOKEN):
    dataset = load_dataset("json", data_files="./data/sample_jailbreaks.json", split="train")
    dataset.map(lambda x: format_jailbreak_instruction(x, EOS_TOKEN), batched=True)
    return dataset
