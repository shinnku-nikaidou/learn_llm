"""
Simple BPE Tokenizer Implementation
Based on Qwen2 model's vocab.json, merges.txt and tokenizer_config.json files
"""

import json
import re
from typing import List, Dict, Tuple, Union
from pathlib import Path


class SimpleBPETokenizer:
    def __init__(self, model_path: str):
        """
        Initialize BPE Tokenizer

        Args:
            model_path: Path to model folder containing vocab.json, merges.txt, tokenizer_config.json
        """
        self.model_path = Path(model_path)

        # Load vocabulary
        self.vocab = self._load_vocab()
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Load BPE merge rules
        self.merges = self._load_merges()

        # Load configuration
        self.config = self._load_config()

        # Special tokens
        self.special_tokens = self._extract_special_tokens()

        # Pre-compile regex pattern (simplified version)
        # Note: Python's re doesn't support \p{L}, \p{N} directly
        # This is a simplified pattern for basic tokenization
        self.pat = re.compile(
            r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w]?\w+|\d+| ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
            re.UNICODE,
        )

    def _load_vocab(self) -> Dict[str, int]:
        """Load vocabulary from vocab.json"""
        vocab_path = self.model_path / "vocab.json"
        with open(vocab_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_merges(self) -> List[Tuple[str, str]]:
        """Load BPE merge rules from merges.txt"""
        merges_path = self.model_path / "merges.txt"
        merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Skip first line (usually version info)
            for line in lines[1:]:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))
        return merges

    def _load_config(self) -> Dict:
        """Load tokenizer configuration"""
        config_path = self.model_path / "tokenizer_config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_special_tokens(self) -> Dict[str, int]:
        """Extract special tokens from config"""
        special_tokens = {}
        if "added_tokens_decoder" in self.config:
            for token_id, token_info in self.config["added_tokens_decoder"].items():
                special_tokens[token_info["content"]] = int(token_id)
        return special_tokens

    def _get_pairs(self, word: List[str]) -> set:
        """Get all adjacent character pairs in word"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _bpe_encode(self, token: str) -> List[str]:
        """
        Perform BPE encoding on single token

        Args:
            token: Input token string

        Returns:
            List of BPE-encoded tokens
        """
        # If special token, return directly
        if token in self.special_tokens:
            return [token]

        # If token in vocabulary, return directly
        if token in self.vocab:
            return [token]

        # Convert token to character list
        word = list(token)
        pairs = self._get_pairs(word)

        if not pairs:
            return [token]

        while True:
            # Find pair with highest priority in merges
            bigram = None
            min_merge_rank = float("inf")

            for pair in pairs:
                if pair in self.merges:
                    merge_rank = self.merges.index(pair)
                    if merge_rank < min_merge_rank:
                        min_merge_rank = merge_rank
                        bigram = pair

            if bigram is None:
                break

            # Perform merge
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self._get_pairs(word)

        return word

    def _preprocess_text(self, text: str) -> str:
        """Text preprocessing (simplified version)"""
        # More complex preprocessing logic can be added here
        return text

    def _split_text(self, text: str) -> List[str]:
        """Split text into tokens (simplified version)"""
        # Simplified version: split by whitespace
        # Real Qwen2 uses complex regex patterns
        tokens = []
        words = text.split()
        for word in words:
            # Add space prefix (similar to GPT processing)
            if tokens:  # Not the first word
                word = "Ġ" + word
            tokens.append(word)
        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs

        Args:
            text: Input text

        Returns:
            List of token IDs
        """
        # Preprocess text
        text = self._preprocess_text(text)

        # Split text
        tokens = self._split_text(text)

        # BPE encode each token
        bpe_tokens = []
        for token in tokens:
            bpe_tokens.extend(self._bpe_encode(token))

        # Convert to IDs
        ids = []
        for token in bpe_tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                # Handle unknown tokens (simplified handling)
                # Should have more complex fallback mechanism
                ids.append(self.vocab.get("<|endoftext|>", 0))

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text

        Args:
            ids: List of token IDs

        Returns:
            Decoded text
        """
        tokens = []
        for id in ids:
            if id in self.id_to_token:
                tokens.append(self.id_to_token[id])
            else:
                tokens.append("<|endoftext|>")  # Fallback for unknown IDs

        # Merge tokens to text
        text = "".join(tokens)

        # Handle special characters
        text = text.replace("Ġ", " ")  # Convert space markers to actual spaces

        return text.strip()

    def encode_chat(self, messages: List[Dict[str, str]]) -> List[int]:
        """
        Encode chat format messages

        Args:
            messages: Message list in format [{"role": "user", "content": "..."}]

        Returns:
            List of encoded token IDs
        """
        if "chat_template" not in self.config:
            raise ValueError("Chat template not found in config")

        # Simplified chat format handling
        formatted_text = ""

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        formatted_text += "<|im_start|>assistant\n"

        return self.encode(formatted_text)

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)

    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token mappings"""
        return self.special_tokens.copy()


def main():
    """Test function"""
    # Initialize tokenizer
    model_path = "./data/qwen2-0.5b"
    tokenizer = SimpleBPETokenizer(model_path)

    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens: {tokenizer.get_special_tokens()}")

    # Test encoding and decoding
    test_text = "Hello world! How are you today?"
    print(f"\nOriginal text: {test_text}")

    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")

    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")

    # Test chat format
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello! How can I help you?"},
    ]

    chat_encoded = tokenizer.encode_chat(messages)
    print(f"\nChat encoded: {chat_encoded[:20]}...")  # Show only first 20 tokens


if __name__ == "__main__":
    main()
