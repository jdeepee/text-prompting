import time
import torch
import argparse
import bittensor as bt
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList, pipeline, StoppingCriteria
from prompting.baseminer.miner import Miner
from prompting.protocol import Prompting

class StopOnTokens(StoppingCriteria):
    """
    Custom stopping criteria for the Misteral model.

    This class defines a stopping criterion based on specific tokens. The model stops generating
    once it encounters one of the specified stop tokens.
    """

    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

model = AutoModelForCausalLM.from_pretrained("mattshumer/mistral-8x7b-chat", low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("mattshumer/mistral-8x7b-chat")
print ("Model loaded!")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0,
    do_sample=False,
    max_new_tokens=256,
    no_repeat_ngram_size=2,
)

history = "<|im_start|>system" + "\n" + "You are an AI assistant.<|im_end|>" + "\n" + "<|im_start|>user" + "\n" + "Tell me about the powers of AI and Nvidia<|im_end|>"

stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])
stop = StopOnTokens(stop_token_ids)

completion = (
    pipe(
        history,
        temperature=0.7,
        max_new_tokens=256,
        no_repeat_ngram_size=2,
        do_sample=False,
        eos_token_id=pipe.tokenizer.eos_token_id,
        pad_token_id=pipe.tokenizer.pad_token_id,
        stopping_criteria=StoppingCriteriaList([stop]),
    )[0]["generated_text"]
    .split(":")[-1]
    .replace(str(history), "")
)
print ("Completion generated!")
print(completion)