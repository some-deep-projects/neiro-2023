import typing as tp
from enum import Enum
from abc import ABC, abstractmethod

from transformers import AutoTokenizer

from trie import Trie


class GenState(Enum):
    NO_STOP = 1
    IN_STOP = 2
    FULL_STOP = 3


class StopCriterion(ABC):
    @abstractmethod
    def on_gen(self, gen_id: int):
        pass

    @abstractmethod
    def get_state(self) -> GenState:
        pass


class WordStopCriterion(StopCriterion):
    def __init__(self, tokenizer: AutoTokenizer, stop_words: tp.List[str],
                 stop_words_ids: tp.Optional[tp.List[int]] = None):
        self.trie = Trie()
        for stop_word in (stop_words_ids or []):
            self.trie.insert(stop_word)
        for stop_word in stop_words:
            word_ids = tokenizer(stop_word, add_special_tokens=False).input_ids
            self.trie.insert(word_ids)
        self.state_node = self.trie.root

    def on_gen(self, gen_id):
        self.state_node = self.state_node.children.get(gen_id, self.trie.root)

    def get_state(self):
        if self.state_node is self.trie.root:
            return GenState.NO_STOP
        if self.state_node.is_end:
            return GenState.FULL_STOP
        return GenState.IN_STOP
