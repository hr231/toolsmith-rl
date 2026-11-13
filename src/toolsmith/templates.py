"""Synthetic repair programs as text, with independent host-side value oracles.

- Defines original MIT-licensed synthetic program templates and their bug variants.
- Provides a challenge table with explicit edge specifications and inspection notes.
- Computes expected values with trusted algorithms; generated source is never executed.
"""

from dataclasses import dataclass

FAMILY_ASSIGNMENTS = {
    "train": ("filter_floor", "affine_map", "delimiter_values"),
    "dev": ("filter_ceiling", "affine_sum", "delimiter_first"),
    "test_iid": ("interval_filter", "affine_even", "delimiter_last"),
    "test_compositional": ("filter_then_affine", "parse_then_affine"),
    "challenge": ("challenge_window", "challenge_clamp", "challenge_chunk", "challenge_split"),
}
# Each split may hold at most MAX_SPLIT_COUNT tasks, so offsets never overlap.
MAX_SPLIT_COUNT = 10_000
SPLIT_OFFSETS = {"train": 0, "dev": 20_000, "test_iid": 40_000, "test_compositional": 60_000}
PROVENANCE = "Original deterministic Toolsmith synthetic templates; no external project code or human-review claim."
IID_CAVEAT = "IID means familiar bug/task classes in unseen project/template families, not an identical project distribution."


@dataclass(frozen=True)
class ChallengeCase:
    family: str
    parameters: tuple
    visible: object
    review_note: str


# Explicit inputs deliberately exercise terminal boundaries, repeated values,
# signed values, Unicode delimiters, and exact delimiter matching.
CHALLENGE_CASES = (
    ChallengeCase(
        "challenge_window", (1,), [0], "Width one on one zero: the sole valid window must survive."
    ),
    ChallengeCase(
        "challenge_window",
        (2,),
        [-1, 1],
        "Width two on canceling signed values: include the exact-length window.",
    ),
    ChallengeCase(
        "challenge_window",
        (3,),
        [2, 2, 2],
        "Width three with duplicates: an exact-length window is not empty.",
    ),
    ChallengeCase(
        "challenge_window",
        (4,),
        [1, -1, 1, -1, 8],
        "Width four with alternating signs: retain the terminal overlapping window.",
    ),
    ChallengeCase(
        "challenge_window",
        (5,),
        [0, 0, 0, 0, 0],
        "Width five on all zeros: a zero-valued result must not disappear.",
    ),
    ChallengeCase(
        "challenge_window",
        (6,),
        [-6, -5, -4, -3, -2, -1],
        "Width six on negatives: preserve the exact-length negative sum.",
    ),
    ChallengeCase(
        "challenge_window",
        (7,),
        [1, 2, 3, 4, 5, 6, 7, 8],
        "Width seven over eight values: both overlapping windows are required.",
    ),
    ChallengeCase(
        "challenge_window",
        (8,),
        [9, 0, 0, 0, 0, 0, 0, -9],
        "Width eight with canceling endpoints: include a terminal zero sum.",
    ),
    ChallengeCase(
        "challenge_window",
        (9,),
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "Width nine repeated ones: count every member in the only window.",
    ),
    ChallengeCase(
        "challenge_window",
        (10,),
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        "Width ten descending values: the final zero still belongs to a complete window.",
    ),
    ChallengeCase(
        "challenge_clamp",
        (-1, 1),
        [-2, -1, 0, 1, 2],
        "Symmetric clamp: values equal to the lower bound must remain unchanged.",
    ),
    ChallengeCase(
        "challenge_clamp",
        (0, 2),
        [-1, 0, 1, 2, 3],
        "Zero lower clamp: negative inputs clamp to zero, not one.",
    ),
    ChallengeCase(
        "challenge_clamp",
        (-5, -2),
        [-9, -5, -3, -2, 0],
        "Negative-only interval: saturation must preserve its negative lower endpoint.",
    ),
    ChallengeCase(
        "challenge_clamp",
        (2, 3),
        [1, 2, 2, 3, 4],
        "Adjacent positive bounds: repeated lower endpoints must not become upper endpoints.",
    ),
    ChallengeCase(
        "challenge_clamp",
        (-10, 10),
        [-100, -10, 0, 10, 100],
        "Wide signed interval: extreme values saturate at the documented endpoints.",
    ),
    ChallengeCase(
        "challenge_clamp",
        (5, 9),
        [5, 4, 9, 10],
        "Positive interval with reversed input order: clamp without sorting.",
    ),
    ChallengeCase(
        "challenge_clamp",
        (-9, -8),
        [-10, -9, -8, -7],
        "Adjacent negative bounds: do not shift the lower endpoint upward.",
    ),
    ChallengeCase(
        "challenge_clamp",
        (10, 100),
        [0, 10, 11, 100, 101],
        "Asymmetric positive interval: preserve the exact lower bound and its neighbor.",
    ),
    ChallengeCase(
        "challenge_clamp",
        (-100, 0),
        [-101, -100, -99, 0, 1],
        "Interval ending at zero: distinguish the lower bound from its successor.",
    ),
    ChallengeCase(
        "challenge_clamp",
        (-1000, 1000),
        [-1001, -1000, 1000, 1001],
        "Large symmetric endpoints: both saturation directions are observable.",
    ),
    ChallengeCase(
        "challenge_chunk", (1,), [0, 1, 2], "Size-one chunks: the final singleton is a real chunk."
    ),
    ChallengeCase(
        "challenge_chunk",
        (2,),
        [-2, -1, 0, 1, 2],
        "Size-two chunks with odd length: retain the signed final singleton.",
    ),
    ChallengeCase(
        "challenge_chunk",
        (3,),
        [7, 7, 7, 7, 7, 7, 7],
        "Size-three chunks of duplicates: multiplicity includes the remainder.",
    ),
    ChallengeCase(
        "challenge_chunk",
        (4,),
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "Size-four chunks: a ninth value begins a partial final chunk.",
    ),
    ChallengeCase(
        "challenge_chunk",
        (5,),
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Size-five chunks of zeros: a zero remainder is not padding.",
    ),
    ChallengeCase(
        "challenge_chunk",
        (6,),
        [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
        "Size-six signed sequence: preserve the last positive singleton.",
    ),
    ChallengeCase(
        "challenge_chunk",
        (7,),
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1],
        "Size-seven alternating sequence: chunk boundaries must not drop the tail.",
    ),
    ChallengeCase(
        "challenge_chunk",
        (8,),
        [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        "Size-eight descending values: retain the final zero singleton.",
    ),
    ChallengeCase(
        "challenge_chunk",
        (9,),
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        "Size-nine chunks: nineteen values require three ordered chunks.",
    ),
    ChallengeCase(
        "challenge_chunk",
        (10,),
        [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        "Size-ten duplicate groups: the twenty-first element remains observable.",
    ),
    ChallengeCase(
        "challenge_split",
        ("|", 1),
        "|a|b",
        "One split with a leading delimiter: preserve the empty first field and unsplit suffix.",
    ),
    ChallengeCase(
        "challenge_split",
        ("::", 2),
        "a::::b::c",
        "Two multi-character splits: adjacent delimiters produce an empty field.",
    ),
    ChallengeCase(
        "challenge_split",
        (",", 3),
        "a,b,c,",
        "Three comma splits ending at the delimiter: preserve the trailing empty field.",
    ),
    ChallengeCase(
        "challenge_split",
        ("aa", 4),
        "aaaaaaaaaa",
        "Four overlapping-looking delimiter matches: consume nonoverlapping matches left to right.",
    ),
    ChallengeCase(
        "challenge_split",
        ("λ", 5),
        "λaλbλcλdλeλf",
        "Five Unicode-letter splits: operate on exact text, not encoded bytes.",
    ),
    ChallengeCase(
        "challenge_split",
        ("\t", 6),
        "a\tb\tc\td\te\tf\tg\th",
        "Six literal tab splits: do not use generic whitespace splitting.",
    ),
    ChallengeCase(
        "challenge_split",
        (" ", 7),
        " a  b c d e f g",
        "Seven literal-space splits: preserve repeated-space empty fields.",
    ),
    ChallengeCase(
        "challenge_split",
        ("<> ", 8),
        "a<> b<> c<> d<> e<> f<> g<> h<> i<> j",
        "Eight delimiters containing a space: match the entire separator exactly.",
    ),
    ChallengeCase(
        "challenge_split",
        ("雪", 9),
        "0雪1雪2雪3雪4雪5雪6雪7雪8雪9雪10",
        "Nine Unicode ideograph splits: leave later separators in the final suffix.",
    ),
    ChallengeCase(
        "challenge_split",
        ("--", 10),
        "--a--b--c--d--e--f--g--h--i--j--",
        "Ten double-hyphen splits: keep a leading empty field and the remaining terminal delimiter.",
    ),
)


@dataclass(frozen=True)
class TemplateInstance:
    family: str
    bug_family: str
    composition_family: str | None
    issue: str
    source: str
    fixed_source: str
    visible_input: object
    visible_expected: object
    private_cases: tuple[tuple[object, object], ...]
    review_note: str | None = None


def _program(argument: str, body: str) -> str:
    return f"def solve({argument}):\n" + "".join(f"    {line}\n" for line in body.splitlines())


def _parse_numbers(text: str, separator: str) -> list[int]:
    """Split text with an independent scanner instead of the generated str.split code."""
    if not text:
        return []
    result = []
    start = 0
    while True:
        end = text.find(separator, start)
        if end < 0:
            result.append(int(text[start:].strip()))
            return result
        result.append(int(text[start:end].strip()))
        start = end + len(separator)


def _affine(value: int, scale: int, bias: int) -> int:
    # Repeated addition is independent of the generated multiply/add expression.
    total = bias
    for _ in range(scale):
        total += value
    return total


def _oracle(family: str, value: object, parameters: tuple) -> object:
    """Compute expected values from input data only, never from generated Python."""
    if family in {"filter_floor", "filter_ceiling", "interval_filter", "filter_then_affine"}:
        threshold, scale, bias = parameters
        output = []
        for number in value:
            if family == "filter_ceiling":
                keep = not number > threshold
            elif family == "interval_filter":
                keep = not (number < threshold or number > threshold + scale)
            else:
                keep = not number < threshold
            if keep:
                output.append(
                    _affine(number, scale, bias) if family == "filter_then_affine" else number
                )
        return output
    if family in {"affine_map", "affine_sum", "affine_even"}:
        _, scale, bias = parameters
        mapped = []
        for number in value:
            if family != "affine_even" or number % 2 == 0:
                mapped.append(_affine(number, scale, bias))
        if family == "affine_sum":
            total = 0
            for number in mapped:
                total += number
            return total
        return mapped
    if family in {"delimiter_values", "delimiter_first", "delimiter_last", "parse_then_affine"}:
        separator, scale, bias = parameters
        numbers = _parse_numbers(value, separator)
        if family == "delimiter_first":
            for number in numbers:
                return number
            return None
        if family == "delimiter_last":
            last = None
            for number in numbers:
                last = number
            return last
        if family == "parse_then_affine":
            return [_affine(number, scale, bias) for number in numbers]
        return numbers
    if family == "challenge_window":
        (width,) = parameters
        result = []
        running = 0
        for position, number in enumerate(value):
            running += number
            if position >= width:
                running -= value[position - width]
            if position >= width - 1:
                result.append(running)
        return result
    if family == "challenge_clamp":
        lower, upper = parameters
        output = []
        for number in value:
            if number < lower:
                output.append(lower)
            elif number > upper:
                output.append(upper)
            else:
                output.append(number)
        return output
    if family == "challenge_chunk":
        (size,) = parameters
        result, current = [], []
        for number in value:
            current.append(number)
            if len(current) == size:
                result.append(current)
                current = []
        if current:
            result.append(current)
        return result
    if family == "challenge_split":
        separator, limit = parameters
        result, current = [], []
        position = 0
        while position < len(value):
            if len(result) < limit and value.startswith(separator, position):
                result.append("".join(current))
                current = []
                position += len(separator)
            else:
                current.append(value[position])
                position += 1
        result.append("".join(current))
        return result
    raise ValueError(f"Unknown template family: {family}")


def _challenge(index: int) -> TemplateInstance:
    spec = CHALLENGE_CASES[index]
    family, parameters, visible = spec.family, spec.parameters, spec.visible
    argument = "values"
    if family == "challenge_window":
        (width,) = parameters
        correct = f"return [sum(values[i:i + {width}]) for i in range(len(values) - {width} + 1)]"
        buggy = correct.replace(f"- {width} + 1", f"- {width}")
        issue = f"For a list of integers, return sums of every contiguous window of width {width}, ordered by starting position. Include the final complete window; return [] when fewer than {width} values are supplied. Do not modify the input."
        private = ([], list(range(width - 1)), list(range(width + 2)))
    elif family == "challenge_clamp":
        lower, upper = parameters
        correct = f"return [min({upper}, max({lower}, value)) for value in values]"
        buggy = f"return [min({upper}, max({lower + 1}, value)) for value in values]"
        issue = f"Clamp each integer in the input list to the inclusive interval [{lower}, {upper}]. Preserve order and duplicates, leave in-range values unchanged, return [] for empty input, and do not modify the input."
        private = ([], [lower, lower - 1, upper, upper + 1], [upper, lower, lower, upper])
    elif family == "challenge_chunk":
        (size,) = parameters
        correct = f"return [values[i:i + {size}] for i in range(0, len(values), {size})]"
        buggy = correct.replace("len(values)", "len(values) - 1")
        issue = f"Partition a list of integers into consecutive chunks of size {size}. Preserve order and duplicates, retain any shorter final chunk, return [] for empty input, and do not modify the input."
        private = ([], [42], list(range(size * 2)))
    else:
        separator, limit = parameters
        argument = "text"
        correct = f"return text.split({separator!r}, {limit})"
        buggy = f"return text.split({separator!r}, {limit - 1})"
        issue = f"Split arbitrary input text on the exact nonempty separator {separator!r}, using at most {limit} nonoverlapping matches from left to right. Preserve empty fields including leading/trailing fields; leave any remaining separators in the final suffix. Empty input returns ['']. Do not strip whitespace."
        private = ("", separator.join(str(i) for i in range(limit + 2)), separator * (limit + 1))
    return TemplateInstance(
        family,
        "boundary_off_by_one",
        None,
        issue,
        _program(argument, buggy),
        _program(argument, correct),
        visible,
        _oracle(family, visible, parameters),
        tuple((item, _oracle(family, item, parameters)) for item in private),
        spec.review_note,
    )


def make_template(split: str, seed: int, index: int) -> TemplateInstance:
    """Select a fixed family first, then a bounded collision-free semantic variant."""
    if split not in FAMILY_ASSIGNMENTS or type(index) is not int or index < 0:
        raise ValueError("Invalid split or instance index")
    if split == "challenge":
        if index >= len(CHALLENGE_CASES):
            raise ValueError("Only forty explicit challenge cases exist")
        return _challenge(index)
    families = FAMILY_ASSIGNMENTS[split]
    family = families[index % len(families)]
    # Disjoint constant ranges per split: otherwise a held-out task can be the
    # exact reversed edit of a development task that shares its parameters.
    variant = seed * 100000 + SPLIT_OFFSETS[split] + index // len(families)
    threshold, scale, bias = variant - 40, 2 + variant % 11, variant + 1
    separator = f"::{variant}::"
    parameters = (threshold, scale, bias)
    argument = "values"
    private = ([], [threshold, threshold + 1, threshold - 1, -2, 0, 2, threshold])
    visible = [threshold - 1, threshold, threshold + 1]
    composition = None
    bug = "boundary_off_by_one"
    suffix = " Input is a list of integers. Preserve order and duplicates, return [] for empty input, and do not modify the input."
    if family == "filter_floor":
        correct = f"return [value for value in values if value >= {threshold}]"
        buggy = correct.replace(">=", ">")
        issue = f"Keep exactly the values greater than or equal to {threshold}." + suffix
    elif family == "filter_ceiling":
        correct = f"return [value for value in values if value <= {threshold}]"
        buggy = correct.replace("<=", "<")
        issue = f"Keep exactly the values less than or equal to {threshold}." + suffix
    elif family == "interval_filter":
        correct = (
            f"return [value for value in values if {threshold} <= value <= {threshold + scale}]"
        )
        buggy = correct.replace(f"value <= {threshold + scale}", f"value < {threshold + scale}")
        visible = [threshold - 1, threshold, threshold + scale, threshold + scale + 1]
        private = ([], [threshold + scale, threshold, threshold + scale - 1, threshold - 1])
        issue = (
            f"Keep exactly the values in the inclusive interval [{threshold}, {threshold + scale}]."
            + suffix
        )
    elif family in {"affine_map", "affine_sum", "affine_even"}:
        bug = "wrong_arithmetic_operator"
        visible = [-2, 0, 3]
        private = ([], [-4, -1, 2, 2, 7])
        expression = f"{scale} * value + {bias}"
        if family == "affine_sum":
            correct = f"return sum({expression} for value in values)"
            issue = f"Return the sum of ({scale} * value + {bias}) over every integer in the input list. Empty input returns 0. Do not modify the input."
        elif family == "affine_even":
            correct = f"return [{expression} for value in values if value % 2 == 0]"
            issue = (
                f"Keep the even input values (including negative even integers and zero), then replace each with {scale} * value + {bias}."
                + suffix
            )
        else:
            correct = f"return [{expression} for value in values]"
            issue = f"Replace every input value with {scale} * value + {bias}." + suffix
        buggy = correct.replace(f"+ {bias}", f"- {bias}")
    elif family == "filter_then_affine":
        composition = "inclusive_floor_then_affine"
        correct = f"return [{scale} * value + {bias} for value in values if value >= {threshold}]"
        buggy = correct.replace(">=", ">")
        issue = (
            f"First retain input values greater than or equal to {threshold}; then transform each retained value to {scale} * value + {bias}. The threshold applies before transformation."
            + suffix
        )
    else:
        argument = "text"
        parameters = (separator, scale, bias)
        visible = separator.join([" -2 ", "0", " 5"])
        private = ("", separator.join(["7", " -11 ", "7", "2"]))
        prefix = f"numbers = [int(field) for field in text.split({separator!r})] if text else []\n"
        if family == "delimiter_values":
            correct, buggy = prefix + "return numbers", prefix + "return numbers[1:]"
            result = (
                "Return all parsed integers in order, preserving duplicates; empty text returns []."
            )
        elif family == "delimiter_first":
            correct, buggy = (
                prefix + "return numbers[0] if numbers else None",
                prefix + "return numbers[-1] if numbers else None",
            )
            result = "Return the first parsed integer, or None for empty text."
        elif family == "delimiter_last":
            correct, buggy = (
                prefix + "return numbers[-1] if numbers else None",
                prefix + "return numbers[0] if numbers else None",
            )
            result = "Return the last parsed integer, or None for empty text."
        else:
            composition = "delimiter_parse_then_affine"
            correct = prefix + f"return [{scale} * value + {bias} for value in numbers]"
            buggy = correct.replace(f"+ {bias}", f"- {bias}")
            result = f"Transform every parsed integer to {scale} * value + {bias}, preserving order and duplicates; empty text returns []."
        bug = "wrong_sequence_selection" if composition is None else "wrong_arithmetic_operator"
        issue = f"Parse text separated by the exact separator {separator!r}. Nonempty input consists of signed decimal integer fields with optional surrounding whitespace; fields themselves are nonempty and contain no separator. {result}"
    return TemplateInstance(
        family,
        bug,
        composition,
        issue,
        _program(argument, buggy),
        _program(argument, correct),
        visible,
        _oracle(family, visible, parameters),
        tuple((item, _oracle(family, item, parameters)) for item in private),
    )
