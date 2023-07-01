from collections import defaultdict


class Node:
    def __init__(self):
        self.is_end = False
        self.children = defaultdict(Node)


class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.children[ch]
        node.is_end = True
