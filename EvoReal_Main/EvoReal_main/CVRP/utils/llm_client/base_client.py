import logging
from abc import ABC, abstractmethod
from typing import Optional, List

logger = logging.getLogger(__name__)

class BaseClient(ABC):
    """
    Abstract base class that defines the basic interface of LLM clients.
    """
    def __init__(self, model: str, temperature: float = 1.0) -> None:
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def generate(self, prompt: list[dict], temperature: Optional[float] = None, n: int = 1):
        """
        Generate an LLM response.
        """
        pass

    @abstractmethod
    def multi_chat_completion(self, prompts: list[list[dict]], temperature: Optional[float] = None, n: int = 1, concurrent: bool = True):
        """
        Generate LLM responses in batches for parallel tasks.
        
        :param prompts: a list of prompts that need to be processed
        :param temperature: Sampling temperature
        :param n: The number of responses generated per Prompt
        :return: A list of multiple LLM responses
        """
        pass
