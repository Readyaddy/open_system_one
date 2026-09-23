"""Real dataset: CLINC150 (Larson et al., 2019).

150 real user-intent classes across 10 domains (banking, travel, kitchen &
dining, auto & commute, work, small talk, meta, credit cards, home,
utility) plus an explicit "out of scope" (oos) class of inputs that match
none of the 150 intents.

We use the "plus" config: 15250 train / 3100 val / 5500 test examples --
real, human-written utterances, not templated synthetic text.

Each intent needs a natural-language OUTCOME DESCRIPTION (not just its
label string), because the whole point of this model is that the outcome
encoder reads a description, not a class id. We auto-generate a base
description from the intent name, and hand-write PARAPHRASED descriptions
(different wording, same meaning) for a subset of intents spanning every
domain, to measure whether the outcome encoder generalizes to descriptions
it never saw during training.
"""
from datasets import load_dataset

OOS_LABEL = "oos"


def _humanize(name: str) -> str:
    return name.replace("_", " ")


def load_clinc150():
    ds = load_dataset("clinc_oos", "plus")
    label_names = ds["train"].features["intent"].names  # includes 'oos'
    return ds, label_names


def build_base_descriptions(label_names):
    """Auto-generated outcome descriptions from the intent name itself."""
    desc = {}
    for name in label_names:
        if name == OOS_LABEL:
            desc[name] = "This request does not match any known category, it is out of scope."
        else:
            desc[name] = f"The user wants help with: {_humanize(name)}."
    return desc


# Hand-written paraphrases for a subset of intents spanning every domain in
# CLINC150, used ONLY at evaluation time to test generalization. Never seen
# during training.
PARAPHRASED_DESCRIPTIONS = {
    "restaurant_reviews": "Looking for opinions or ratings about a place to eat.",
    "nutrition_info": "Asking about the nutritional content of a food item.",
    "account_blocked": "The customer's account got locked or frozen.",
    "weather": "Wants to know the current or forecasted weather conditions.",
    "translate": "Needs a phrase converted from one language to another.",
    "book_flight": "Trying to reserve a plane ticket.",
    "book_hotel": "Trying to reserve a hotel room.",
    "car_rental": "Wants to rent a vehicle for a trip.",
    "traffic": "Curious about current road congestion or delays.",
    "directions": "Needs step-by-step navigation to a destination.",
    "spending_history": "Wants to review past transactions or purchases.",
    "credit_score": "Asking what their credit score is.",
    "pay_bill": "Wants to submit a payment for a bill that is due.",
    "transfer": "Wants to move money between accounts.",
    "balance": "Wants to know how much money is currently in an account.",
    "alarm": "Wants to set a wake-up or reminder alarm.",
    "timer": "Wants to start a countdown timer.",
    "calendar": "Wants to check or manage scheduled events.",
    "reminder": "Wants to be reminded about something later.",
    "recipe": "Looking for cooking instructions for a dish.",
    "calories": "Wants to know how many calories something has.",
    "tell_joke": "Wants to hear something funny.",
    "greeting": "Just saying hello.",
    "goodbye": "Saying farewell, ending the conversation.",
    "thank_you": "Expressing gratitude.",
    "meaning_of_life": "Asking a big philosophical question about life's purpose.",
    "who_made_you": "Curious who built or created this assistant.",
    "gas": "Wants to know where to find fuel nearby.",
    "mpg": "Asking about a vehicle's fuel efficiency.",
    "tire_pressure": "Wants to know the correct tire inflation level.",
}


def get_examples(split_dataset, label_names):
    return [(ex["text"], label_names[ex["intent"]]) for ex in split_dataset]
