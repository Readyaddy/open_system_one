"""Synthetic support-ticket triage dataset.

Each example is a short ticket (context) that should be routed to exactly
one of a fixed set of outcomes. Outcomes are described in natural language
(not just a label string) because the whole point of the JEPA-style setup
is that the outcome encoder reads a *description* of the candidate, not a
one-hot class id.
"""
import random

OUTCOMES = {
    "refund": "The customer wants their money back for a purchase.",
    "escalate": "This issue is urgent or serious and needs a human manager right away.",
    "faq_reply": "This is a common question that can be answered with existing help documentation.",
    "bug_report": "The customer is describing a software defect or something broken in the product.",
    "ignore": "This message is spam, a test, or not a real support request.",
    "compliment": "The customer is happy and praising the product or support team.",
}

# Paraphrased descriptions, held out from training, used only to probe
# whether the outcome encoder generalizes semantically rather than
# memorizing training strings.
OUTCOMES_PARAPHRASED = {
    "refund": "Please give my money back, I returned the item.",
    "escalate": "This needs to go straight to a supervisor, it's critical.",
    "faq_reply": "A quick lookup in the knowledge base would answer this.",
    "bug_report": "Something in the app is crashing or not working correctly.",
    "ignore": "This looks like junk mail or an automated test ping, not a real ticket.",
    "compliment": "The user is thanking us and loves the service.",
}

OUTCOME_KEYS = list(OUTCOMES.keys())

TEMPLATES = {
    "refund": [
        "I want a refund for my {item}, it arrived {defect}.",
        "Please refund my order, the {item} was {defect}.",
        "Can I get my money back for the {item} I bought last week?",
        "I'd like a refund, this {item} is not what I ordered.",
        "Requesting a refund on {item}, totally {defect}.",
    ],
    "escalate": [
        "This is urgent, my {item} caused a serious safety issue, I need a manager now.",
        "I have been charged {money} twice and no one is helping, escalate this immediately.",
        "Critical problem with my account, please get a supervisor on the phone now.",
        "This is the third time I'm contacting support about {item}, I demand escalation.",
        "Emergency: my {item} stopped working and I have a deadline today, need urgent help.",
    ],
    "faq_reply": [
        "How do I reset my password for my account?",
        "What are your business hours for support?",
        "Where can I find the size chart for {item}?",
        "How long does shipping usually take for {item}?",
        "Can you tell me how to change my email address on file?",
    ],
    "bug_report": [
        "The {item} app crashes every time I try to open the settings page.",
        "I found a bug where the {item} button does nothing when clicked.",
        "The website throws an error when I try to check out with {item} in my cart.",
        "There's a glitch in the {item} feature, it shows the wrong numbers.",
        "The app freezes and then closes whenever I use the {item} tool.",
    ],
    "ignore": [
        "asdkjaslkdj test test 12345",
        "This is a test message, please disregard.",
        "buy cheap watches now click here www.example.com",
        "hello world just testing the form",
        "asdf asdf asdf asdf",
    ],
    "compliment": [
        "Just wanted to say your support team was amazing helping me with {item}!",
        "I love the new {item}, it works perfectly and looks great.",
        "Thank you so much, the team resolved my {item} issue so fast, you're the best.",
        "Great product, {item} exceeded my expectations!",
        "Really happy with {item}, will definitely buy again.",
    ],
}

ITEMS = ["blender", "laptop", "headphones", "jacket", "subscription", "phone case",
         "keyboard", "backpack", "monitor", "shoes"]
DEFECTS = ["broken", "damaged", "the wrong color", "missing parts", "defective"]
MONEY = ["$50", "$120", "$19.99", "$300"]


def _fill(template: str, rng: random.Random) -> str:
    return template.format(
        item=rng.choice(ITEMS),
        defect=rng.choice(DEFECTS),
        money=rng.choice(MONEY),
    )


def generate_examples(n_per_class: int, seed: int):
    rng = random.Random(seed)
    examples = []
    for label in OUTCOME_KEYS:
        templates = TEMPLATES[label]
        for _ in range(n_per_class):
            t = rng.choice(templates)
            text = _fill(t, rng)
            examples.append((text, label))
    rng.shuffle(examples)
    return examples


def build_splits(seed: int = 0):
    train = generate_examples(200, seed)
    val = generate_examples(30, seed + 1)
    test = generate_examples(30, seed + 2)
    return train, val, test
