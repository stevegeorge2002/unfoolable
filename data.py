import random, datetime
import pandas as pd

FIRST = ["James", "Maria", "Wei", "Aisha", "Carlos", "Priya", "John", "Sofia", "Ahmed", "Emma",
         "Liam", "Olga", "Kenji", "Fatima", "Noah", "Chloe", "Ravi", "Grace", "Mateo", "Hana"]
LAST = ["Smith", "Garcia", "Chen", "Khan", "Lopez", "Patel", "Brown", "Rossi", "Ali", "Wilson",
        "Murphy", "Ivanova", "Tanaka", "Haddad", "Clark", "Martin", "Nair", "Lee", "Silva", "Kim"]
MAILS = ["gmail.com", "outlook.com", "acme.io", "example.org"]
CITIES = ["Boston", "Austin", "Denver", "Seattle", "Chicago", "Miami"]

DOMAINS = {
    "customers": [("customer_id", "id"), ("full_name", "name"), ("email", "email"), ("phone", "phone"),
                  ("signup_date", "date"), ("city", "category", CITIES), ("lifetime_value", "money")],
    "invoices": [("invoice_id", "id"), ("vendor", "category", ["Acme Corp", "Globex", "Initech", "Umbrella"]),
                 ("amount", "money"), ("issue_date", "date"), ("due_date", "date"),
                 ("status", "category", ["paid", "unpaid", "overdue"])],
    "orders": [("order_id", "id"), ("customer_name", "name"), ("order_date", "date"),
               ("product", "category", ["Laptop", "Monitor", "Keyboard", "Mouse", "Dock"]),
               ("quantity", "int"), ("unit_price", "money")],
    "employees": [("employee_id", "id"), ("full_name", "name"), ("email", "email"),
                  ("department", "category", ["Engineering", "Sales", "Finance", "Support"]),
                  ("hire_date", "date"), ("salary", "money")],
    "subscriptions": [("subscription_id", "id"), ("email", "email"),
                      ("plan", "category", ["Basic", "Pro", "Enterprise"]), ("start_date", "date"),
                      ("monthly_fee", "money"), ("seats", "int")],
    # held-out kinds: never seen during training
    "products": [("sku", "id"), ("category", "category", ["Audio", "Video", "Gaming", "Office"]),
                 ("price", "money"), ("stock", "int"), ("launch_date", "date")],
    "payments": [("payment_id", "id"), ("payer", "name"), ("amount", "money"), ("paid_on", "date"),
                 ("method", "category", ["card", "bank_transfer", "cash"])],
    "shipments": [("shipment_id", "id"), ("carrier", "category", ["UPS", "FedEx", "DHL", "USPS"]),
                  ("ship_date", "date"), ("weight_kg", "decimal"), ("destination", "category", CITIES),
                  ("phone", "phone")],
}
TRAIN_DOMAINS = ["customers", "invoices", "orders", "employees", "subscriptions"]
HELDOUT_DOMAINS = ["products", "payments", "shipments"]

TYPE_RULES = {
    "id": "positive integer, unique per row",
    "name": "'First Last' in Title Case with single spaces",
    "email": "lowercase, no spaces",
    "phone": "string formatted XXX-XXX-XXXX",
    "date": "string formatted YYYY-MM-DD",
    "money": "float rounded to 2 decimals, no currency symbols, no commas",
    "decimal": "float rounded to 2 decimals",
    "int": "integer",
    "category": "exactly one of the allowed values (case-sensitive)",
}


def _value(rng, ctype, extra, row):
    if ctype == "name":
        f, l = rng.choice(FIRST), rng.choice(LAST)
        row["_first"], row["_last"] = f, l
        return f"{f} {l}"
    if ctype == "email":
        f, l = row.get("_first") or rng.choice(FIRST), row.get("_last") or rng.choice(LAST)
        return f"{f.lower()}.{l.lower()}{rng.randint(1, 99)}@{rng.choice(MAILS)}"
    if ctype == "phone":
        return f"{rng.randint(200, 989)}-{rng.randint(200, 989)}-{rng.randint(1000, 9999)}"
    if ctype == "date":
        d = datetime.date(2023, 1, 1) + datetime.timedelta(days=rng.randint(0, 1000))
        return d.isoformat()
    if ctype == "money":
        return round(rng.uniform(5, 5000), 2)
    if ctype == "decimal":
        return round(rng.uniform(0.1, 80), 2)
    if ctype == "int":
        return rng.randint(1, 50)
    if ctype == "category":
        return rng.choice(extra)
    raise ValueError(ctype)


def make_clean(domain, seed, n=30):
    """Return (clean DataFrame, schema list) for one dataset. Same domain+seed -> same data."""
    rng = random.Random(f"{domain}-{seed}")
    schema = DOMAINS[domain]
    ids = rng.sample(range(1000, 9999), n)
    rows = []
    for i in range(n):
        row = {}
        for col in schema:
            name, ctype = col[0], col[1]
            extra = col[2] if len(col) > 2 else None
            row[name] = ids[i] if ctype == "id" else _value(rng, ctype, extra, row)
        rows.append({k: v for k, v in row.items() if not k.startswith("_")})
    return pd.DataFrame(rows), [(c[0], c[1], c[2] if len(c) > 2 else None) for c in schema]

def schema_text(schema):
    """Human-readable target spec that agents will see."""
    lines = []
    for col in schema:
        name, ctype = col[0], col[1]
        rule = TYPE_RULES[ctype]
        if ctype == "category":
            rule += f": {col[2]}"
        lines.append(f"- {name}: {rule}")
    lines.append("- one row per unique id; drop exact duplicate records")
    return "\n".join(lines)


if __name__ == "__main__":
    for d in ["customers", "shipments"]:
        df, schema = make_clean(d, seed=1)
        print(f"\n=== {d} ({len(df)} rows) ===")
        print(df.head(4).to_string(index=False))
        print(schema_text(schema))