import logging
import time
import random
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
from .base_client import BaseClient

try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger(__name__)
# This is the full implementation of OpenAIClient with no abstract methods left.
class OpenAIClient(BaseClient):
    """
    OpenAI API client that encapsulates OpenAI ChatCompletion to support LLM data generation.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> None:
        super().__init__(model, temperature)

        if openai is None:
            logger.fatal("Package `openai` is required. Please install it using `pip install openai`")
            exit(-1)
        
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        
    def generate(self, prompt: list[dict], temperature: Optional[float] = None, n: int = 1):
        """
        Generate LLM responses, based on OpenAI ChatCompletion with retry and random sleep.
        """
        temperature = temperature or self.temperature

        max_retries = 5
        backoff_base = 1.0
        for attempt in range(max_retries):
            try:
                time.sleep(random.uniform(0.5, 1.5))  # avoid triggering rate limit
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=prompt,
                    temperature=temperature,
                    n=n,
                    stream=False,
                    timeout=300
                )
                if n>1:
                    return [choice.message.content for choice in response.choices]
                else:
                    return response.choices[0].message.content if response.choices else None
            except Exception as e:
                logger.warning(f"[generate] Attempt {attempt+1}/{max_retries} failed: {e}")
                time.sleep(backoff_base * (attempt + 1))  # exponential backoff
        logger.error("[generate] All retry attempts failed.")
        return None

    def multi_chat_completion(
            self,
            prompts: list[list[dict]],
            temperature: Optional[float] = None,
            n: int = 1,
            concurrent: bool = True
        ):
            """
            Generate LLM responses in batch mode.

            :param prompts: List of messages (each a list of dicts)
            :param temperature: Sampling temperature
            :param n: Number of responses per prompt
            :param concurrent: Whether to use multi-threaded concurrent execution
            :return: List of generated responses
            """
            assert isinstance(prompts, list), "prompts should be a list of message lists."
            if not isinstance(prompts[0], list) or not isinstance(prompts[0][0], dict):
                raise ValueError("Each prompt must be a list of dictionaries (Chat messages).")


            if concurrent:
                if n != 1:
                    raise ValueError("Concurrent mode only supports n=1.")
                with ThreadPoolExecutor() as executor:
                    return list(executor.map(lambda p: self.generate(p, temperature, 1), prompts))
            else:
                if len(prompts) == 1 and n > 1:
                    # single prompt, multiple responses
                    return self.generate(prompts[0], temperature, n)
                else:
                    # multiple prompts, each with n=1
                    return [self.generate(p, temperature, n) for p in prompts]

