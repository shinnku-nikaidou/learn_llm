"""
Comparison between official Qwen2 tokenizer and our simple BPE tokenizer
"""

import torch
from transformers import AutoTokenizer
from simple_bpe_tokenizer import SimpleBPETokenizer


def compare_tokenizers():
    """Compare official and custom tokenizers"""

    # Initialize both tokenizers
    MODEL_DIR = "./data/qwen2-0.5b"

    print("Loading tokenizers...")
    official_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    custom_tokenizer = SimpleBPETokenizer(MODEL_DIR)

    print("=" * 60)
    print("TOKENIZER COMPARISON")
    print("=" * 60)

    # Basic info comparison
    print(f"\n📊 BASIC INFO:")
    print(f"Official vocab size: {official_tokenizer.vocab_size}")
    print(f"Custom vocab size:   {custom_tokenizer.get_vocab_size()}")

    print(f"\n🔧 SPECIAL TOKENS:")
    print(f"Official special tokens: {len(official_tokenizer.added_tokens_decoder)}")
    for token_id, token in official_tokenizer.added_tokens_decoder.items():
        print(f"  {token}: {token_id}")

    print(f"\nCustom special tokens: {len(custom_tokenizer.get_special_tokens())}")
    for token, token_id in custom_tokenizer.get_special_tokens().items():
        print(f"  {token}: {token_id}")

    # Test cases
    test_texts = [
        "Hello world!",
        "How are you today?",
        "用两句话解释量子纠缠。",
        "The quick brown fox jumps over the lazy dog.",
        "人工智能是未来的趋势",
        "Hello! 你好世界 123",
    ]

    print(f"\n📝 ENCODING COMPARISON:")
    print("-" * 60)

    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. Text: '{text}'")

        # Official tokenizer
        official_ids = official_tokenizer.encode(text, add_special_tokens=False)
        official_decoded = official_tokenizer.decode(official_ids)

        # Custom tokenizer
        custom_ids = custom_tokenizer.encode(text)
        custom_decoded = custom_tokenizer.decode(custom_ids)

        print(f"   Official IDs: {official_ids}")
        print(f"   Custom IDs:   {custom_ids}")
        print(f"   Official decoded: '{official_decoded}'")
        print(f"   Custom decoded:   '{custom_decoded}'")

        # Check if they match
        ids_match = official_ids == custom_ids
        text_match = official_decoded.strip() == custom_decoded.strip()

        print(f"   ✅ IDs match: {ids_match}")
        print(f"   ✅ Text match: {text_match}")

        if not ids_match:
            print(f"   ⚠️  Length diff: {len(official_ids)} vs {len(custom_ids)}")

    # Chat format comparison
    print(f"\n💬 CHAT FORMAT COMPARISON:")
    print("-" * 60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How can you help me?"},
    ]

    # Official chat format
    official_chat_prompt = official_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    official_chat_ids = official_tokenizer.encode(
        official_chat_prompt, add_special_tokens=False
    )

    # Custom chat format
    custom_chat_ids = custom_tokenizer.encode_chat(messages)

    print(f"Official chat prompt:\n{repr(official_chat_prompt)}")
    print(
        f"\nOfficial chat IDs: {official_chat_ids[:20]}... (length: {len(official_chat_ids)})"
    )
    print(
        f"Custom chat IDs:   {custom_chat_ids[:20]}... (length: {len(custom_chat_ids)})"
    )

    # Analyze differences
    print(f"\n🔍 ANALYSIS:")
    print("-" * 60)

    # Find most common differences
    all_official_ids = []
    all_custom_ids = []

    for text in test_texts:
        all_official_ids.extend(
            official_tokenizer.encode(text, add_special_tokens=False)
        )
        all_custom_ids.extend(custom_tokenizer.encode(text))

    print(f"Total official tokens: {len(all_official_ids)}")
    print(f"Total custom tokens:   {len(all_custom_ids)}")
    print(
        f"Average tokens per text (official): {len(all_official_ids)/len(test_texts):.1f}"
    )
    print(
        f"Average tokens per text (custom):   {len(all_custom_ids)/len(test_texts):.1f}"
    )


if __name__ == "__main__":
    compare_tokenizers()
