from unsloth import is_bfloat16_supported
import torch
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, PeftModel, TaskType
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from format_dataset import load_jailbreak_dataset

max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model_name = "cognitivecomputations/Wizard-Vicuna-7B-Uncensored"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Handles multi-GPU or CPU deployment
    trust_remote_code=True,  # Required for some models
    revision="main",  # Specify model revision/branch
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.2,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    bias="none"
)

model = PeftModel(model, lora_config)
model.print_trainable_parameters()
# model.gradient_checkpointing_enable()
# for param in model.parameters():
#     param = param.float()
#     param.requires_grad = True

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ",
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
#     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
# )

# model = FastLanguageModel.get_peft_model(
#     model,
#     r=16,
#     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
#                       "gate_proj", "up_proj", "down_proj",],
#     lora_alpha = 16,
#     lora_dropout = 0, # Supports any, but = 0 is optimized
#     bias = "none",    # Supports any, but = "none" is optimized
#     # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
#     use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
#     random_state = 3407,
#     use_rslora = False,  # We support rank stabilized LoRA
#     loftq_config = None, # And LoftQ
# )

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

dataset = load_jailbreak_dataset(EOS_TOKEN)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 4,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
# used_percentage = round(used_memory         /max_memory*100, 3)
# lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
# print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
# print(f"Peak reserved memory % of max memory = {used_percentage} %.")
# print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

model.save_pretrained("lora_model") # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving