# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from fastchat.train.llama2_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

from fastchat.train import train

from fastchat.train.train import transformers, Dict, rank0_print, Dataset, torch

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # Apply prompt templates, you might want to modify this 
    conversations = []
    for i, source in enumerate(sources):
        conversations.append(source["input"] + "::: " + source["output"])

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        #from extract_data_from_train import load_data_from_json
        #self.raw_data = load_data_from_json(self.raw_data, True)
        #state_after_list = ["state after: "+ ex["state_after"] for ex in raw_datasets]
        #state_before_tactic = ["state before: "+ ex["state_before"] + ", tactic: "+ ex["tactic"] for ex in raw_datasets]
        #self.raw_data = [x1 + "::: " + x2 for x1, x2 in zip(state_after_list, state_before_tactic)]
        

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret
    
train.preprocess = preprocess
train.LazySupervisedDataset = LazySupervisedDataset

if __name__ == "__main__":
    train.train()
