import random, datetime
import pandas as pd
from data import make_clean


def _cols(schema, *types):
    return [c[0] for c in schema if c[1] in types]


def _apply(df, cols, rng, p, fn):
    for col in cols:
        df[col] = [fn(v) if rng.random() < p else v for v in df[col]]
    return df


def t_dates(df, schema, rng, p):
    fmts = ["%m/%d/%Y", "%d %b %Y", "%b %d, %Y", "%Y/%m/%d", "%d.%m.%Y"]
    return _apply(df, _cols(schema, "date"), rng, p,
                  lambda v: datetime.date.fromisoformat(v).strftime(rng.choice(fmts)))


def t_money(df, schema, rng, p):
    def f(v):
        x = float(v)
        eu = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return rng.choice([f"${x:,.2f}", f"{x:.2f} USD", f"USD {x:,.2f}", f"  {x:.2f} ", f"{eu} EUR-format"])
    df = _apply(df, _cols(schema, "money"), rng, p, f)
    return _apply(df, _cols(schema, "decimal"), rng, p,
                  lambda v: rng.choice([f"{float(v)} kg", f"{float(v):.2f}".replace(".", ","), f" {v} "]))


def t_names(df, schema, rng, p):
    def f(v):
        first, last = v.split(" ", 1)
        return rng.choice([v.lower(), v.upper(), f"{last}, {first}", f"  {first}   {last} "])
    return _apply(df, _cols(schema, "name"), rng, p, f)


def t_emails(df, schema, rng, p):
    return _apply(df, _cols(schema, "email"), rng, p,
                  lambda v: rng.choice([v.upper(), f"  {v} ", f"mailto:{v}"]))


def t_phones(df, schema, rng, p):
    def f(v):
        d = v.replace("-", "")
        return rng.choice([f"({d[:3]}) {d[3:6]}-{d[6:]}", f"{d[:3]}.{d[3:6]}.{d[6:]}",
                           f"+1 {d[:3]} {d[3:6]} {d[6:]}", d])
    return _apply(df, _cols(schema, "phone"), rng, p, f)


def t_categories(df, schema, rng, p):
    return _apply(df, _cols(schema, "category"), rng, p,
                  lambda v: rng.choice([v.lower(), v.upper(), f" {v} "]))


def t_ints(df, schema, rng, p):
    df = _apply(df, _cols(schema, "int"), rng, p, lambda v: rng.choice([f"{v}.0", f" {v} "]))
    return _apply(df, _cols(schema, "id"), rng, p / 3, lambda v: rng.choice([f"#{v}", f"ID-{v}"]))


BASE_TRICKS = {"dates": t_dates, "money": t_money, "names": t_names, "emails": t_emails,
               "phones": t_phones, "categories": t_categories, "ints": t_ints}


def make_task(domain, seed, extra_tricks=None, p=0.4):
    """Build one cleaning task: messy table (all text) + hidden perfect answer."""
    truth, schema = make_clean(domain, seed)
    rng = random.Random(f"mess-{domain}-{seed}")
    messy = truth.astype(str)
    dup = messy.sample(n=3, random_state=seed)
    messy = pd.concat([messy, dup], ignore_index=True)
    for fn in BASE_TRICKS.values():
        messy = fn(messy, schema, rng, p)
    for fn in (extra_tricks or []):
        messy = fn(messy, schema, rng, p)
    messy = messy.sample(frac=1, random_state=seed).reset_index(drop=True)
    messy.columns = [rng.choice([c, c.replace("_", " ").title(), c.upper(), f" {c} "]) for c in messy.columns]
    return {"id": f"{domain}-{seed}", "domain": domain, "messy": messy, "truth": truth, "schema": schema}


def _same(got, exp, ctype):
    try:
        if ctype in ("money", "decimal"):
            return abs(float(got) - float(exp)) < 0.006
        if ctype in ("int", "id"):
            return int(float(got)) == int(exp)
        return str(got) == str(exp)
    except (ValueError, TypeError):
        return False


def score(cleaned, truth, schema):
    """Cell accuracy vs the perfect answer (0..1) plus a few example mistakes."""
    idcol, cols = schema[0][0], [c[0] for c in schema]
    types = {c[0]: c[1] for c in schema}
    if cleaned is None or idcol not in getattr(cleaned, "columns", []):
        return 0.0, [f"output missing id column '{idcol}'"]
    got, extra = {}, 0
    for _, r in cleaned.iterrows():
        try:
            k = int(float(r[idcol]))
        except (ValueError, TypeError):
            extra += 1
            continue
        if k in got:
            extra += 1
        else:
            got[k] = r
    truth_ids = set(int(x) for x in truth[idcol])
    extra += sum(1 for k in got if k not in truth_ids)
    correct, errors = 0, []
    for _, t in truth.iterrows():
        k = int(t[idcol])
        r = got.get(k)
        if r is None:
            if len(errors) < 12:
                errors.append(f"row with id {k} is missing from the output "
                              f"(its id may be written in a non-standard way in the input)")
            continue
        for c in cols:
            val = r[c] if c in r.index else "COLUMN NOT FOUND"
            if val != "COLUMN NOT FOUND" and _same(val, t[c], types[c]):
                correct += 1
            elif len(errors) < 12:
                errors.append(f"{c} (id {k}): got {val!r}, expected {t[c]!r}")
    if extra and len(errors) < 12:
        errors.append(f"{extra} extra/duplicate rows in output")
    acc = correct / (len(truth) * len(cols) + extra * len(cols))
    return acc, errors


if __name__ == "__main__":
    task = make_task("customers", seed=1)
    print(f"MESSY ({len(task['messy'])} rows):")
    print(task["messy"].head(6).to_string(index=False))
    perfect, _ = score(task["truth"], task["truth"], task["schema"])
    print(f"\nScore if perfect: {perfect:.0%}")
    raw = task["messy"].copy()
    raw.columns = [c[0] for c in task["schema"]]
    lazy, errs = score(raw, task["truth"], task["schema"])
    print(f"Score if you do nothing but fix headers: {lazy:.0%}")
    print("Example mistakes:", *errs[:4], sep="\n  ")