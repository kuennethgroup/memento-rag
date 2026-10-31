"""Tolerant JSON parsing for model output: models copy LaTeX from the sources (a lone backslash is not valid JSON, so a
whole answer's claims would be lost) and copy quotations without escaping the inner quotes."""
import json
import re

# One escape at a time, consumed in pairs, so an already escaped backslash is kept while a lone one is fixed. Kept as
# JSON escapes: an escaped backslash, quote or slash, unicode escapes, and b f n r t when not followed by a lowercase
# letter (else it is LaTeX: rho, frac, beta, nu, theta, tau …). Everything else gets its backslash escaped.
_ESCAPE = re.compile(r'\\(?:(?P<keep>u[0-9a-fA-F]{4}|[bfnrt](?![a-z])|["\\/])|(?P<bad>.))', re.S)


def _fix_escapes(s: str) -> str:
    return _ESCAPE.sub(lambda m: m.group(0) if m.group("keep") is not None else "\\\\" + m.group("bad"), s)


def _escape_inner_quotes(s: str) -> str:
    """Inside a JSON string, a '"' that is not followed by , : } ] (or the end) cannot close it → escape it."""
    out, in_str, i = [], False, 0
    while i < len(s):
        ch = s[i]
        if in_str and ch == "\\":
            out.append(s[i:i + 2]); i += 2
            continue
        if ch == '"':
            if not in_str:
                in_str = True
            else:
                rest = s[i + 1:].lstrip()
                if rest and rest[0] not in ",:}]":
                    out.append('\\"'); i += 1
                    continue
                in_str = False
        out.append(ch); i += 1
    return "".join(out)


def loads_lenient(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    fixed = _fix_escapes(s)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return json.loads(_escape_inner_quotes(fixed))
