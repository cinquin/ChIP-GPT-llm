import shelve
import signal
from contextlib import contextmanager
from functools import wraps
from inspect import signature
from typing import Callable, Optional, Tuple, List


def memoize_instance_method(method):
    cache = {}
    
    def wrapper(self, *args, **kwargs):
        key = (args, tuple(kwargs.items()))
        if key not in cache:
            cache[key] = method(self, *args, **kwargs)
        return cache[key]
    return wrapper


def use_defaults_on_none(func):
    sig = signature(func)
    defaults = {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not v.empty
    }
    print(defaults)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Bind the provided arguments to the function signature
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        
        # Use the default value for any argument that is None
        for arg_name, arg_value in bound_args.arguments.items():
            if arg_value is None and arg_name in defaults:
                bound_args.arguments[arg_name] = defaults[arg_name]
        
        return func(*bound_args.args, **bound_args.kwargs)
    return wrapper


def parse_bool_string(string_value: str) -> bool:
    parsed_value = string_value.strip().lower()
    if parsed_value == 'true':
        return True
    elif parsed_value == 'false':
        return False
    else:
        raise ValueError(f"Invalid boolean string: {string_value}")


def count_lines_starting_with_zero_dot(input_string):
    lines = input_string.split('\n')
    count = 0
    for line in lines:
        if line.strip().startswith('0.'):
            count += 1
    return count


class AutoComputedShelfDB:
    def __init__(self, filename: str, compute_default_value: Callable[[str], str], flags='c', mode=0o666):
        self.db: shelve.Shelf[str] = shelve.open(filename)
        self._encoding = 'utf-8'
        self._compute_default_value = compute_default_value
    
    @staticmethod
    def _encode(value: Optional[str]) -> str:
        return value.strip()
    
    @staticmethod
    def _decode(value: str) -> str:
        return value
    
    def __getitem__(self, key: str) -> str:
        encoded_key = self._encode(key)
        try:
            value = self._decode(self.db.__getitem__(encoded_key))
        except KeyError:
            value = self._compute_default_value(encoded_key)
            self.db.__setitem__(encoded_key, self._encode(value))
            self.db.sync()
        return value
    
    def __setitem__(self, key: str, value: str):
        self.db.__setitem__(self._encode(key), self._encode(value))
        self.db.sync()
    
    def __delitem__(self, key: str):
        self.db.__delitem__(self._encode(key))
    
    def get(self, key: str, default: str = None) -> str:
        encoded_key: str = self._encode(key)
        try:
            value: str = self._decode(self.db.__getitem__(encoded_key))
        except KeyError:
            if default is not None:
                value = default
            else:
                value = self._compute_default_value(encoded_key)
            self.db.__setitem__(encoded_key, self._encode(value))
            self.db.sync()
        return value
    
    def pop(self, key: str, default=None) -> str:
        return self._decode(super().pop(self._encode(key), self._encode(default)))
    
    def popitem(self) -> Tuple[str, str]:
        key: bytes
        value: bytes
        key, value = self.db.popitem()
        return self._decode(key), self._decode(value)


def join_strings_with_dots(strings: List[str]):
    result = ""
    
    counter: int = 0
    
    for i, string in enumerate(strings):
        if string.lstrip(' .') == '':
            continue
        if counter == 0:
            result += string
        elif result.endswith("."):
            result += " " + string
        else:
            result += ". " + string
        counter += 1
    
    return result


def find_nth_occurrence(substring: str, string: str, n: int):
    """
    Finds the nth occurrence of 'substring' in 'string'.
    Returns the index of the nth occurrence, or -1 if not found.
    """
    index = -1
    for _ in range(n):
        # Find the next occurrence of the substring
        index = string.find(substring, index + 1)
        # If the substring is not found, return -1
        if index == -1:
            return -1
    return index


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def tokenized_length(text, tokenizer):
    return len(tokenizer.encode(text, return_tensors="pt", truncation=False).squeeze())


def join_strings_with_period(strings: List[str]):
    if not strings:
        return ''
    result = strings[0]
    for s in strings[1:]:
        if not result.endswith('.'):
            result += '.'
        result += ' ' + s.rstrip(' . \n')
    return result
