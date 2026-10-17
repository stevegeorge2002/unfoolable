import random, re, datetime, unicodedata
import pandas as pd
from llm import SUPER, extract_code
from data import make_clean, TRAIN_DOMAINS

TRICK_SYS = """You are the TRICKSTER. You invent realistic ways data files get messed up, to test a data-cleaning agent.
Write TWO Python functions:

def trick(df, schema, rng, p):
    # df: pandas DataFrame, every value is a string. schema: list of 3-tuples (column_name, type, allowed_values_or_None)
    # types: id, name, email, phone, date, money, decimal, int, category. rng: random.Random. p: chance per cell.
    # return the modified df

def untrick(df, schema):
    # receives ONLY the messed-up df and the schema, and must restore EVERY value to the EXACT original string.
    # This is your proof that the trick is fair. If untrick cannot restore the exact original, the trick is rejected.

RULES:
- Never delete rows or columns, never blank out values, never destroy information.
- Only touch about a fraction p of cells, using rng. Only use built-ins, re, datetime and unicodedata. No files, no network.
- trick and untrick run in separate sandboxes: they cannot share hidden state.
- Be realistic and sneaky (Excel exports, European locales, OCR, legacy systems, copy-paste from emails, unicode lookalikes...).
- Do NOT repeat these existing tricks:
{existing}
{history}
Return ONE ```python``` block. Its first line must be a comment: # TRICK: short name - one line description"""

BANNED = ["import os", "subprocess", "open(", "eval(", "exec(", "__import__", "socket", "shutil", "sys.", "requests",
          "globals(", "setattr(", "builtins"]


def _load(code):
    env = {"re": re, "datetime": datetime, "random": random, "pd": pd, "unicodedata": unicodedata}
    exec(code, env)
    return env


def compile_trick(code, require_inverse=False):
    """Validate a trick. With require_inverse=True the Trickster must PROVE fairness with an exact untrick()."""
    if any(b in code for b in BANNED):
        return None, "uses forbidden calls"
    try:
        env_t, env_u = _load(code), _load(code)   # separate copies: trick cannot pass secrets to untrick
    except Exception as e:
        return None, f"does not compile: {e}"
    fn, inv = env_t.get("trick"), env_u.get("untrick")
    if not callable(fn):
        return None, "no function named trick"
    if require_inverse and not callable(inv):
        return None, "no untrick() proof of fairness"
    changed = 0
    for d in TRAIN_DOMAINS[:3]:
        truth, schema = make_clean(d, seed=7)
        before = truth.astype(str)
        try:
            after = fn(before.copy(), schema, random.Random(0), 0.5)
        except Exception as e:
            return None, f"crashed on {d}: {type(e).__name__}: {e}"
        if not isinstance(after, pd.DataFrame) or after.shape != before.shape or list(after.columns) != list(before.columns):
            return None, "changed table shape or column names (not allowed)"
        after = after.astype(str)
        if ((after.apply(lambda s: s.str.strip()) == "") & (before != "")).any().any():
            return None, "blanked out values (not recoverable)"
        changed += int((after != before).sum().sum())
        if require_inverse:
            try:
                restored = inv(after.copy(), schema)
            except Exception as e:
                return None, f"untrick crashed on {d}: {type(e).__name__}: {e}"
            if (not isinstance(restored, pd.DataFrame) or restored.shape != before.shape
                    or not (restored.astype(str).values == before.values).all()):
                return None, "UNFAIR: your own untrick() could not restore the exact original values"
    if changed == 0:
        return None, "did not change anything"
    return fn, ""


def trick_name(code):
    m = re.search(r"#\s*TRICK:\s*(.+)", code)
    return (m.group(1).strip() if m else "unnamed trick")[:90]


def invent_trick(llm, existing, history="", tries=3, verbose=True):
    """Ask the Trickster for a new trick that it can prove is fair. Returns (name, fn, code) or None."""
    feedback = ""
    for attempt in range(tries):
        sys_prompt = TRICK_SYS.format(existing="\n".join(f"- {e}" for e in existing), history=history)
        reply = llm.chat(SUPER, sys_prompt, "Invent the next trick now. Reply with the python code block."
                         + feedback, max_tokens=3500, temperature=1.0)
        code = extract_code(reply)
        fn, err = compile_trick(code, require_inverse=True)
        if fn:
            return trick_name(code), fn, code
        if verbose:
            print(f"  [referee rejected attempt {attempt + 1}] {trick_name(code)} -> {err}")
        feedback = f"\n\nYour last trick was rejected: {err}. Fix it.\n```python\n{code[:3000]}\n```"
    return None


if __name__ == "__main__":
    from llm import LLM
    from tricks import BASE_TRICKS
    llm = LLM()
    got = invent_trick(llm, list(BASE_TRICKS))
    if not got:
        print("Trickster could not prove any trick fair in 3 tries.")
    else:
        name, fn, code = got
        print("NEW TRICK (proven fair: untrick restores exact originals):", name)
        truth, schema = make_clean("customers", seed=1)
        before = truth.astype(str)
        after = fn(before.copy(), schema, random.Random(1), 0.5)
        diff = [(c, before.at[i, c], after.at[i, c]) for c in before.columns for i in before.index
                if before.at[i, c] != str(after.at[i, c])]
        for c, b, a in diff[:8]:
            print(f"  {c}: {b!r}  ->  {a!r}")
    print(f"COST ${llm.spent:.3f}")