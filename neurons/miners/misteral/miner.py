import time
import torch
import argparse
import bittensor as bt
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteriaList
from prompting.baseminer.miner import Miner
from prompting.protocol import Prompting

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
            "--openorca.device", type=str, help="Device to load model", default="cuda"
        )
        parser.add_argument(
            "--openorca.max_new_tokens",
            type=int,
            help="Max tokens for model output.",
            default=256,
        )
        # Add additional arguments as needed

    def __init__(self, *args, **kwargs):
        """
        Initializes the MisteralMiner, loading the tokenizer and model.
        """
        super(MisteralMiner, self).__init__(*args, **kwargs)
        bt.logging.info("Loading " + str(self.config.openorca.model_name))
        self.tokenizer = AutoTokenizer.from_pretrained("mattshumer/mistral-8x7b-chat")
        self.model = AutoModelForCausalLM.from_pretrained("mattshumer/mistral-8x7b-chat", low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True)
        bt.logging.info("Model loaded!")

        if self.config.openorca.device != "cpu":
            self.model = self.model.to(self.config.openorca.device)

    def _process_history(self, roles: List[str], messages: List[str]) -> str:
        """
        Processes the conversation history for model input.

        This method takes the roles and messages from the incoming request and constructs
        a conversation history suitable for model input. It also injects a system prompt
        if the configuration specifies to do so.
        """
        processed_history = ""
        if self.config.btlm.do_prompt_injection:
            processed_history += self.config.btlm.system_prompt
        for role, message in zip(roles, messages):
            if role == "system":
                if not self.config.btlm.do_prompt_injection or message != messages[0]:
                    processed_history += "<|im_start|>system" + "\n" + message + "<|im_end|>"
            if role == "assistant":
                processed_history += "<|im_start|>assistant" + "\n" + message + "<|im_end|>"
            if role == "user":
                processed_history += "<|im_start|>user" + "\n" + message + "<|im_end|>"
        return processed_history

    def prompt(self, synapse: Prompting) -> Prompting:
        """
        Overrides the Miner's abstract `prompt` method to process incoming requests using OpenOrca.
        """

        history = self._process_history(roles=synapse.roles, messages=synapse.messages)
        history += "<|im_start|>assistant\n"
        bt.logging.debug("History: {}".format(history))

        x = self.tokenizer.encode(history, return_tensors="pt").cuda()
        x = self.model.generate(x, max_new_tokens=self.openorca.max_new_tokens)
        # completion = (
        #     self.pipe(
        #         history,
        #         temperature=self.config.btlm.temperature,
        #         max_new_tokens=self.config.btlm.max_length,
        #         no_repeat_ngram_size=self.config.btlm.no_repeat_ngram_size,
        #         do_sample=self.config.btlm.do_sample,
        #         eos_token_id=self.pipe.tokenizer.eos_token_id,
        #         pad_token_id=self.pipe.tokenizer.pad_token_id,
        #         stopping_criteria=StoppingCriteriaList([self.stop]),
        #     )[0]["generated_text"]
        #     .split(":")[-1]
        #     .replace(str(history), "")
        # )
        bt.logging.debug("Completion: {}".format(x))
        synapse.completion = x
        return synapse

if __name__ == "__main__":
    with MisteralMiner():
        while True:
            print("running...", time.time())
            time.sleep(1)