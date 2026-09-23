"""Hand-written paraphrased outcome descriptions, held out of ALL training
(base descriptions, augmentation pool, and the zero-shot split), used only
to measure generalization at eval time on intents the model DID train on
(different wording, same class -- as opposed to the zero-shot eval, which
tests classes the model never trained on at all).
"""

CLINC_PARAPHRASES = {
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

BANKING_PARAPHRASES = {
    "card_arrival": "Wondering when their new card will show up.",
    "card_delivery_estimate": "Wants a timeframe for when the card will be delivered.",
    "lost_or_stolen_card": "Their card has gone missing or been taken.",
    "card_swallowed": "The ATM machine kept their card and didn't give it back.",
    "compromised_card": "Suspects someone else has been using their card fraudulently.",
    "declined_card_payment": "A purchase with their card was rejected.",
    "declined_cash_withdrawal": "Trying to take out cash but it was refused.",
    "pending_card_payment": "A charge is showing as not yet finalized.",
    "card_not_working": "Their card isn't functioning when they try to use it.",
    "virtual_card_not_working": "Their digital/virtual card won't work.",
    "contactless_not_working": "Tap-to-pay isn't functioning on their card.",
    "activate_my_card": "Needs to turn on a new card before using it.",
    "change_pin": "Wants to update their card's PIN number.",
    "pin_blocked": "Their PIN has been locked after too many wrong attempts.",
    "exchange_rate": "Asking what the current currency conversion rate is.",
    "transfer_not_received_by_recipient": "The person they sent money to says it never arrived.",
    "top_up_failed": "Tried to add funds to the account but it didn't go through.",
    "request_refund": "Wants their money returned for a transaction.",
    "terminate_account": "Wants to permanently close their account.",
    "age_limit": "Asking if there's a minimum or maximum age to use the service.",
}

SNIPS_PARAPHRASES = {
    "AddToPlaylist": "Wants a song or artist added to an existing music playlist.",
    "BookRestaurant": "Trying to reserve a table at a restaurant.",
    "GetWeather": "Asking what the weather will be like.",
    "PlayMusic": "Wants a song, artist, or album played.",
    "RateBook": "Wants to give a book a star rating or review score.",
    "SearchCreativeWork": "Looking for a specific movie, show, book, or song by name.",
    "SearchScreeningEvent": "Looking for movie showtimes at a theater.",
}


def get_paraphrases():
    combined = {}
    for k, v in CLINC_PARAPHRASES.items():
        combined[f"clinc::{k}"] = v
    for k, v in BANKING_PARAPHRASES.items():
        combined[f"banking::{k}"] = v
    for k, v in SNIPS_PARAPHRASES.items():
        combined[f"snips::{k}"] = v
    return combined
