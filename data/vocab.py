from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        chars = set()
        for c in text:
            chars.add(c)
        chars = sorted(chars)
        stoi, itos = {}, {}
        for idx, c in enumerate(chars):
            stoi[c] = idx
            itos[idx] = c
        return (stoi, itos)
        
    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        res = []
        for c in text:
            res.append(stoi[c])
        return res

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        res = []
        for i in ids:
            res.append(itos[i])
        return ''.join(res)
