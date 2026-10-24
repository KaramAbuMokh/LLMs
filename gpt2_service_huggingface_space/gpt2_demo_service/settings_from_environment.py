# Typed readers for environment variables: each takes the mapping (never os.environ itself) and names the variable in every error
import re  # commit pattern
from enum import StrEnum  # choice type
from typing import Mapping, TypeVar  # environment mapping and the enum type variable

TRUE_TEXTS = frozenset({"1", "true", "yes", "on"})  # accepted spellings of "on"
FALSE_TEXTS = frozenset({"0", "false", "no", "off"})  # accepted spellings of "off"
FULL_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")  # a full commit hash
ChoiceType = TypeVar("ChoiceType", bound=StrEnum)  # any string enumeration


def read_text_variable(environment: Mapping[str, str], name: str, default: str) -> str:  # a plain string; unset or empty keeps the default
    return environment.get(name) or default  # empty counts as unset


def read_boolean_variable(environment: Mapping[str, str], name: str, default: bool) -> bool:  # true/false/1/0/yes/no/on/off in any case
    raw_text = environment.get(name, "").strip().lower()  # what was set
    if not raw_text:  # unset or empty
        return default  # keep the default
    if raw_text in TRUE_TEXTS:  # on
        return True  # enabled
    if raw_text in FALSE_TEXTS:  # off
        return False  # disabled
    raise ValueError("%s must be true or false, got %r" % (name, raw_text))  # names the variable


def read_positive_integer_variable(environment: Mapping[str, str], name: str, default: int) -> int:  # a whole number of at least 1
    raw_text = environment.get(name, "").strip()  # what was set
    if not raw_text:  # unset or empty
        return default  # keep the default
    if not raw_text.isdigit() or int(raw_text) < 1:  # not a positive whole number
        raise ValueError("%s must be a whole number of at least 1, got %r" % (name, raw_text))  # names the variable
    return int(raw_text)  # the number


def read_choice_variable(environment: Mapping[str, str], name: str, default: ChoiceType) -> ChoiceType:  # one member of a string enumeration
    raw_text = environment.get(name, "").strip()  # what was set
    if not raw_text:  # unset or empty
        return default  # keep the default
    try:  # the enumeration knows its values
        return type(default)(raw_text)  # the member
    except ValueError:  # not one of them
        raise ValueError("%s must be one of %s, got %r" % (name, [member.value for member in type(default)], raw_text)) from None  # names the variable


def read_commit_variable(environment: Mapping[str, str], name: str, default: str) -> str:  # a full 40-character commit hash
    raw_text = environment.get(name, "").strip().lower()  # what was set
    if not raw_text:  # unset or empty
        return default  # keep the default
    if not FULL_COMMIT_PATTERN.fullmatch(raw_text):  # a branch name or short hash could move under us
        raise ValueError("%s must be a full 40-character commit hash" % name)  # names the variable
    return raw_text  # the pinned commit
