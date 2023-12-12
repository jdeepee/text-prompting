import time
import torch
import argparse
import bittensor as bt
import deepspeed
import os

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

class MisteralMiner(Miner):
    """
    A Bittensor Miner subclass specific to the Mistral-7B model.
    """

    def config(self) -> "bt.config":
        """
        Configures the Misteral Miner with relevant arguments.
        """
        parser = argparse.ArgumentParser(description="Mistral-7B Miner")
        self.add_args(parser)
        return bt.config(parser)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        """
        Adds specific arguments to the argparse parser for Miner configuration.
        """
        parser.add_argument(
            "--misteral.device", type=str, help="Device to load model", default="cuda"
        )
        parser.add_argument(
            "--misteral.max_new_tokens",
            type=int,
            help="Max tokens for model output.",
            default=256,
        )
        parser.add_argument(
            "--misteral.do_sample",
            action="store_true",
            default=False,
            help="Whether to use sampling or not (if not, uses greedy decoding).",
        )
        parser.add_argument(
            "--misteral.no_repeat_ngram_size",
            type=int,
            default=2,
            help="The size of the n-grams to avoid repeating in the generated text.",
        )
        parser.add_argument(
            "--misteral.do_prompt_injection",
            action="store_true",
            default=False,
            help='Whether to use a custom "system" prompt instead of the one sent by bittensor.',
        )
        parser.add_argument(
            "--misteral.system_prompt",
            type=str,
            help="What prompt to replace the system prompt with",
            default="A chat between a curious user and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions. ",
        )
        parser.add_argument(
            "--misteral.use_deepspeed",
            action="store_true",
            default=False,
            help="Whether to use deepspeed or not (if not, uses vanilla huggingface).",
        )
        parser.add_argument(
            "--misteral.temperature", type=float, default=0.7, help="Sampling temperature."
        )

    def __init__(self, *args, **kwargs):
        """
        Initializes the MisteralMiner, loading the tokenizer and model.
        """
        super(MisteralMiner, self).__init__(*args, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("Open-Orca/Mistral-7B-OpenOrca")
        self.model = AutoModelForCausalLM.from_pretrained("Open-Orca/Mistral-7B-OpenOrca", low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True)
        bt.logging.info("Model loaded!")

        self.stop_token_ids = self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])
        self.stop = StopOnTokens(self.stop_token_ids)

        # Determine correct device id (int) from device string.
        if self.config.misteral.device == "cuda":
            self.config.misteral.device = 0
        elif len(self.config.misteral.device.split(":") == 2):
            try:
                self.config.misteral.device = int(self.config.misteral.device.split(":")[1])
            except:
                raise ValueError(
                    "Invalid device string: {}".format(self.config.misteral.device)
                )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            do_sample=self.config.misteral.do_sample,
            max_new_tokens=self.config.misteral.max_new_tokens,
            no_repeat_ngram_size=self.config.misteral.no_repeat_ngram_size,
        )
        # Optionally initialize deepspeed for inference speedup
        if self.config.misteral.use_deepspeed:
            self.pipe.model = deepspeed.init_inference(
                self.pipe.model,
                mp_size=int(os.getenv("WORLD_SIZE", "1")),
                dtype=torch.float,
                replace_with_kernel_inject=False,
            )

    def _process_history(self, roles: List[str], messages: List[str]) -> str:
        """
        Processes the conversation history for model input.

        This method takes the roles and messages from the incoming request and constructs
        a conversation history suitable for model input. It also injects a system prompt
        if the configuration specifies to do so.
        """
        processed_history = ""
        # if self.config.btlm.do_prompt_injection:
        #     processed_history += self.config.btlm.system_prompt
        for role, message in zip(roles, messages):
            if role == "system":
                #if not self.config.btlm.do_prompt_injection or message != messages[0]:
                if not message != messages[0]:
                    processed_history += "<|im_start|>system" + "\n" + message + "<|im_end|>"
            if role == "assistant":
                processed_history += "<|im_start|>assistant" + "\n" + message + "<|im_end|>"
            if role == "user":
                processed_history += "<|im_start|>user" + "\n" + message + "<|im_end|>"
        return processed_history

    def prompt(self, synapse: Prompting) -> Prompting:
        """
        Overrides the Miner's abstract `prompt` method to process incoming requests using Misteral.
        """

        history = self._process_history(roles=synapse.roles, messages=synapse.messages)
        history += "<|im_start|>assistant\n"
        bt.logging.debug("History: {}".format(history))

        completion = (
            self.pipe(
                history,
                temperature=self.config.misteral.temperature,
                max_new_tokens=self.config.misteral.max_length,
                no_repeat_ngram_size=self.config.misteral.no_repeat_ngram_size,
                do_sample=self.config.misteral.do_sample,
                eos_token_id=self.pipe.tokenizer.eos_token_id,
                pad_token_id=self.pipe.tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([self.stop]),
            )
        )
        bt.logging.debug("Completion: {}".format(completion))
        synapse.completion = completion
        return synapse

if __name__ == "__main__":
    """
    Main execution point for the MisteralMiner.

    This script initializes and runs the MisteralMiner, connecting it to the Bittensor network.
    The miner listens for incoming requests and responds using the Misteral Model.

    Developers can start the miner by executing this script. It uses the context manager to ensure
    proper cleanup of resources after the miner is stopped.
    """
    bt.debug()
    miner = MisteralMiner()
    with miner:
        while True:
            time.sleep(1)