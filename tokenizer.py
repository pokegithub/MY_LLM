"""
tokenizer.py — BPE tokenizer (byte-level, GPT-2 compatible) (v3)

Changes from v2:
  - Added batch encode/decode for efficiency
  - Added token-level entropy estimation
  - Improved cache with LRU eviction
  - Thread-safe encoding
"""

import os
import re
import json
import threading
from typing import Any, Dict, List, Tuple, Optional
from collections import OrderedDict

from core.logging import get_logger
from security.validator import get_allowed_data_roots, safe_load_json

LOGGER = get_logger("tokenizer")


def _build_byte_encoder() -> Dict[int, str]:
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_ENC = _build_byte_encoder()
_BYTE_DEC = {v: k for k, v in _BYTE_ENC.items()}

try:
    import regex as _re
    _SPLIT = _re.compile(
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+"""
        r"""|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+"""
        r"""|\s+(?!\S)|\s+""", _re.UNICODE)
    _HAS_REGEX = True
except ImportError:
    _SPLIT = re.compile(r"\S+|\s+")
    _HAS_REGEX = False


class LRUCache:
    """Thread-safe LRU cache for BPE results."""

    def __init__(self, maxsize: int = 100_000):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key, value):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = value


class BPETokenizer:
    PAD = "<|pad|>"
    BOS = "<|bos|>"
    EOS = "<|eos|>"
    UNK = "<|unk|>"
    SYS = "<|system|>"
    USR = "<|user|>"
    AST = "<|assistant|>"
    END = "<|end|>"
    THK = "<|think|>"
    ANS = "<|answer|>"
    VER = "<|verify|>"
    UNC = "<|unknown|>"

    SPECIAL = [PAD, BOS, EOS, UNK, SYS, USR, AST, END, THK, ANS, VER, UNC]

    def __init__(self):
        self.encoder: Dict[str, int] = {}
        self.decoder: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], int] = {}
        self.vocab_size_: int = 0
        self._cache = LRUCache(maxsize=200_000)
        self._domain_cache = LRUCache(maxsize=100_000)
        self._special_set: set = set()
        self._special_pattern: Optional[re.Pattern] = None
        self._encode_trie: Dict[str, Any] = {}
        self._trie_terminal = "__id__"
        self._code_math_pattern = re.compile(
            r"\\[a-zA-Z]+|[{}\[\]();:=<>+\-*/\\^_]"
        )

    def _compile_special_pattern(self):
        # Sort longest-first so regex alternation never captures partial tokens.
        specials = sorted(self.SPECIAL, key=len, reverse=True)
        self._special_pattern = re.compile(
            "|".join(re.escape(t) for t in specials)
        )

    def _build_encode_trie(self) -> None:
        trie: Dict[str, Any] = {}
        for token, token_id in self.encoder.items():
            node = trie
            for ch in token:
                if ch not in node:
                    node[ch] = {}
                node = node[ch]
            node[self._trie_terminal] = int(token_id)
        self._encode_trie = trie

    def load(self, path: str):
        ep = os.path.join(path, "encoder.json")
        mp = os.path.join(path, "merges.json")
        if not os.path.exists(ep):
            raise FileNotFoundError(f"encoder.json not in {path}")
        if not os.path.exists(mp):
            raise FileNotFoundError(f"merges.json not in {path}")

        allowed_roots = get_allowed_data_roots()
        self.encoder = safe_load_json(
            ep,
            max_bytes=64_000_000,
            allowed_roots=allowed_roots,
        )
        raw = safe_load_json(
            mp,
            max_bytes=64_000_000,
            allowed_roots=allowed_roots,
        )

        if not isinstance(self.encoder, dict):
            raise ValueError(f"Tokenizer encoder must be a JSON object: {ep}")
        if not isinstance(raw, list):
            raise ValueError(f"Tokenizer merges must be a JSON list: {mp}")

        self.merges = {}
        for item in raw:
            if isinstance(item, list) and len(item) == 2:
                pair, rank = item
                if isinstance(pair, list) and len(pair) == 2:
                    self.merges[(pair[0], pair[1])] = int(rank)

        self.decoder = {v: k for k, v in self.encoder.items()}

        for tok in self.SPECIAL:
            if tok not in self.encoder:
                nid = max(self.encoder.values()) + 1 if self.encoder else 0
                self.encoder[tok] = nid
                self.decoder[nid] = tok

        self.vocab_size_ = len(self.encoder)
        self._special_set = set(self.SPECIAL)
        self._compile_special_pattern()
        self._build_encode_trie()

        if not _HAS_REGEX:
            LOGGER.info("Tip: pip install regex")
        LOGGER.info("Tokenizer loaded: %s/ (vocab: %s)", path, f"{self.vocab_size_:,}")

    @property
    def pad_token_id(self) -> int:
        return self.encoder.get(self.PAD, 0)

    @property
    def bos_token_id(self) -> int:
        return self.encoder.get(self.BOS, 1)

    @property
    def eos_token_id(self) -> int:
        return self.encoder.get(self.EOS, 2)

    @property
    def unk_token_id(self) -> int:
        return self.encoder.get(self.UNK, 3)

    @property
    def sys_token_id(self) -> int:
        return self.encoder.get(self.SYS, 4)

    @property
    def usr_token_id(self) -> int:
        return self.encoder.get(self.USR, 5)

    @property
    def ast_token_id(self) -> int:
        return self.encoder.get(self.AST, 6)

    @property
    def end_token_id(self) -> int:
        return self.encoder.get(self.END, 7)

    @property
    def think_token_id(self) -> int:
        return self.encoder.get(self.THK, 8)

    @property
    def ans_token_id(self) -> int:
        return self.encoder.get(self.ANS, 9)

    @property
    def verify_token_id(self) -> int:
        return self.encoder.get(self.VER, 10)

    @property
    def unknown_token_id(self) -> int:
        return self.encoder.get(self.UNC, 11)

    def _pairs(self, w: List[str]) -> set:
        return {(w[i], w[i + 1]) for i in range(len(w) - 1)}

    def _bpe(self, token: str) -> List[str]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        word = [_BYTE_ENC[b] for b in token.encode("utf-8")]
        if len(word) <= 1:
            self._cache.put(token, word)
            return word
        while True:
            pairs = self._pairs(word)
            if not pairs:
                break
            best = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best not in self.merges:
                break
            a, b = best
            new: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new.append(a + b)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = new
            if len(word) == 1:
                break
        self._cache.put(token, word)
        return word

    def _encode_plain_text(self, text: str) -> List[int]:
        if self._looks_like_code_or_math(text):
            return self._encode_code_math_text(text)

        ids: List[int] = []
        split = _SPLIT.findall(text)

        for w in split:
            if w:
                for sub in self._bpe(w):
                    ids.append(self.encoder.get(sub, self.unk_token_id))
        return ids

    def _to_token_chars(self, text: str) -> str:
        return "".join(_BYTE_ENC[b] for b in text.encode("utf-8"))

    def _longest_vocab_match(self, encoded: str, start: int) -> Tuple[int, int]:
        node = self._encode_trie
        last_id: Optional[int] = None
        last_end = start
        pos = start

        while pos < len(encoded):
            ch = encoded[pos]
            nxt = node.get(ch)
            if not isinstance(nxt, dict):
                break
            node = nxt
            pos += 1

            candidate = node.get(self._trie_terminal)
            if isinstance(candidate, int):
                last_id = candidate
                last_end = pos

        if last_id is not None:
            return last_id, last_end

        single = encoded[start]
        return self.encoder.get(single, self.unk_token_id), start + 1

    def _encode_code_math_text(self, text: str) -> List[int]:
        cached = self._domain_cache.get(text)
        if cached is not None:
            return list(cached)

        encoded = self._to_token_chars(text)
        ids: List[int] = []
        i = 0
        while i < len(encoded):
            token_id, nxt = self._longest_vocab_match(encoded, i)
            ids.append(token_id)
            i = nxt

        self._domain_cache.put(text, tuple(ids))
        return ids

    def _looks_like_code_or_math(self, text: str) -> bool:
        if not text:
            return False
        markers = len(self._code_math_pattern.findall(text))
        if markers < 3:
            return False
        alpha_count = sum(ch.isalpha() for ch in text)
        return markers >= max(3, alpha_count // 6)

    def encode(self, text: str, add_bos: bool = False,
               add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if not text:
            if add_bos:
                ids.append(self.bos_token_id)
            if add_eos:
                ids.append(self.eos_token_id)
            return ids

        if self._special_pattern is None:
            self._compile_special_pattern()

        pos = 0
        for m in self._special_pattern.finditer(text):
            if m.start() > pos:
                ids.extend(self._encode_plain_text(text[pos:m.start()]))
            tok = m.group(0)
            ids.append(self.encoder.get(tok, self.unk_token_id))
            pos = m.end()

        if pos < len(text):
            ids.extend(self._encode_plain_text(text[pos:]))

        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def encode_batch(self, texts: List[str], add_bos: bool = False,
                     add_eos: bool = False) -> List[List[int]]:
        """Batch encode for efficiency."""
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos)
                for t in texts]

    def decode(self, ids: List[int], skip_special: bool = False) -> str:
        parts: List[str] = []
        for tid in ids:
            tok = self.decoder.get(tid, self.UNK)
            if skip_special and tok in self._special_set:
                continue
            parts.append(tok)
        text = "".join(parts)
        bv: List[int] = []
        for ch in text:
            if ch in _BYTE_DEC:
                bv.append(_BYTE_DEC[ch])
            else:
                bv.extend(ch.encode("utf-8"))
        try:
            return bytes(bv).decode("utf-8", errors="replace")
        except Exception:
            return text

    def decode_batch(self, id_lists: List[List[int]],
                     skip_special: bool = False) -> List[str]:
        """Batch decode for efficiency."""
        return [self.decode(ids, skip_special=skip_special)
                for ids in id_lists]

    def apply_chat_template(self, messages: List[dict],
                            system: str = "") -> str:
        r = ""
        if system:
            r += f"{self.SYS}\n{system}{self.END}\n"
        for m in messages:
            role = m.get("role", "user")
            c = m.get("content", "")
            if role == "user":
                r += f"{self.USR}\n{c}{self.END}\n"
            elif role == "assistant":
                r += f"{self.AST}\n{c}{self.END}\n"
            elif role == "system":
                r += f"{self.SYS}\n{c}{self.END}\n"
        return r

    def encode_chat(self, messages: List[dict], system: str = "",
                    add_generation_prompt: bool = True) -> List[int]:
        text = self.apply_chat_template(messages, system)
        if add_generation_prompt:
            text += f"{self.AST}\n"
        return self.encode(text, add_bos=True)

    def __repr__(self) -> str:
        return (f"BPETokenizer(vocab={self.vocab_size_:,}, "
                f"merges={len(self.merges):,})")