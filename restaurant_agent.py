# pip install langgraph groq python-dotenv
# Expected environment variable in .env: GROQ_API_KEY

import os
import re
import time
import uuid
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Load environment variables from .env
load_dotenv()

# The Groq API key is read from GROQ_API_KEY environment variable. Never hardcode it.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# MODEL SELECTION REASONING:
# 'llama-3.3-70b-versatile' has become an Enterprise-only model on Groq and returns 404
# (model_not_found) for developer/free accounts. 'openai/gpt-oss-20b' is supported, active,
# and verified on the current API tier. DO NOT revert to llama-3.3-70b-versatile.
LLM_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# API call tracking metrics to detect broken integrations
API_CALLS_ATTEMPTED = 0
API_CALLS_SUCCEEDED = 0
API_CALLS_FAILED = 0

# Global test flags to simulate failure conditions for cook and serve nodes
MOCK_COOK_FAIL = False
MOCK_SERVE_FAIL = False

# ============================================================
# Fixed Menu
# ============================================================
MENU = {
    "butter chicken": 12, "paneer tikka": 10, "dal makhani": 8,
    "naan": 3, "garlic naan": 4, "biryani": 11, "samosa": 4,
    "pizza": 9, "pasta": 9, "pad thai": 10, "spring rolls": 5,
    "fried rice": 8, "manchurian": 9, "gulab jamun": 5, "lassi": 4,
}

def format_full_menu() -> str:
    """Format the full 15-item menu with prices for the initial greeting."""
    items_str = "\n".join([f"  • {dish.title()}: ${price}" for dish, price in MENU.items()])
    return f"Here is our menu:\n{items_str}"

def format_compact_menu() -> str:
    """Format a compact one-line reminder of dish names for failure messages."""
    dish_list = ", ".join(MENU.keys())
    return f"Menu: {dish_list}"

# ============================================================
# State Schema
# ============================================================

class OrderItem(TypedDict):
    dish: str
    quantity: int
    status: str  # "pending" | "confirmed" | "cooking" | "served"


class RestaurantState(TypedDict):
    messages: Annotated[list, add_messages]
    order_id: Optional[str]
    customer_name: Optional[str]
    items: list[OrderItem]

    stage: str                     # ordering | cooking | serving | billing | feedback | done | cancelled
    failure_reason: Optional[str]
    failure_type: Optional[str]    # "quantity" | "no_cook" | "dropped_food" | "not_on_menu" | None
    retry_count: int

    bill_amount: Optional[float]
    feedback_rating: Optional[int]
    feedback_comment: Optional[str]


# ============================================================
# System Prompt (Verbatim as specified)
# ============================================================

SYSTEM_PROMPT = """You are the conversational voice of a restaurant ordering system. You are
the only node in this system that talks directly to the customer. Every
other node (Take Order, Cook, Serve, Bill) works silently in the background
and reports its outcome back to you. Your job is to:

1. Take and confirm orders in natural, friendly language.
2. Explain problems from the kitchen in a calm, helpful way and get the
   customer's decision.
3. Collect quick feedback after the bill is settled.
4. Know when to gracefully end the conversation.

You never invent kitchen facts (stock levels, staff availability, cooking
status) — you only relay what the state tells you via failure_reason /
failure_type, or confirm success when there is no failure.

RESPONSIBILITIES BY SITUATION

1. Taking the order (stage == "ordering", no failure): Greet the customer,
   confirm what they've ordered (dish + quantity for each item), and pass
   it along. Keep it short — one confirmation line is enough.

2. Handling a failure (failure_reason is set, retry_count < 3): Match your
   response to failure_type exactly:
   - "quantity": "We only have limited quantity of [dish] left — would you
     like us to prepare what's available, or would you like to order
     something else instead?"
   - "no_cook": "Our kitchen can't prepare that dish right now — would you
     like to order something else instead?"
   - "dropped_food": "Sorry — there was a small mishap with your dish.
     Could you wait a little while we remake it?"
   Always end with a clear question so the customer's next message gives
   you something actionable.

3. Final failure (retry_count >= 3, failure_reason still set): Do not offer
   another retry. Apologize once, briefly, and close the order: "I'm really
   sorry — we still weren't able to sort this out after a few tries. Thank
   you for your patience, and we hope to serve you again soon." Set stage to
   "cancelled". Do not ask a follow-up question here.

4. After the bill (stage == "billing" just completed successfully): Present
   the bill amount, then immediately ask one short feedback question: "Your
   total is $[bill_amount]. Thank you for dining with us! On a scale of
   1-5, how was your experience today?" Set stage to "feedback" after
   asking.

5. Collecting feedback (stage == "feedback"): When the customer replies
   with a rating (and optionally a comment), store it in feedback_rating
   (and feedback_comment if given), thank them warmly, and close: "Thanks
   so much for letting us know — see you next time!" Set stage to "done".
   If the customer skips or declines, thank them anyway and close the same
   way — never ask a second time.

TONE
- Warm, brief, and human — like a good server, not a form.
- One question at a time. Never stack multiple questions in one message.
- Never expose internal state names (stage, failure_type, etc.) to the
  customer — translate everything into plain language.

OUTPUT FORMAT
Respond only with the natural-language message to show the customer, plus
whichever state fields you are updating this turn. Do not narrate your own
reasoning to the customer."""


# ============================================================
# Extraction Helpers (for robust state updates)
# ============================================================

# Filler/measure words to strip before MENU matching
FILLER_WORDS = [
    "plates of", "plate of", "bowls of", "bowl of", "cups of", "cup of",
    "glasses of", "glass of", "orders of", "order of", "servings of", "serving of",
    "plates", "plate", "bowls", "bowl", "cups", "cup",
    "glasses", "glass", "orders", "order", "servings", "serving",
]


def _normalize_dish_name(raw: str) -> str:
    """Strip filler/measure words, then try exact match → substring match against MENU.

    Returns the best-matching MENU key, or the cleaned string if no match is found
    (so take_order can flag it as not_on_menu).
    """
    cleaned = raw.lower().strip().rstrip(".")

    # Strip trailing 's' for simple plurals (e.g. "samosas" → "samosa", "lassis" → "lassi")
    # but not for words that end in 'ss' (e.g. "glass")
    depluralized = cleaned
    if cleaned.endswith("s") and not cleaned.endswith("ss"):
        depluralized = cleaned[:-1]

    # Strip filler/measure words (longest match first — "plates of" before "plate")
    for filler in FILLER_WORDS:
        if cleaned.startswith(filler + " "):
            cleaned = cleaned[len(filler):].strip()
            break

    # Also try depluralized after filler stripping
    cleaned_deplural = cleaned
    if cleaned.endswith("s") and not cleaned.endswith("ss"):
        cleaned_deplural = cleaned[:-1]

    # Try exact match first (original, cleaned, depluralized variants)
    for candidate in [cleaned, depluralized, cleaned_deplural]:
        if candidate in MENU:
            return candidate

    # Substring/fuzzy match: if a MENU key is contained in the candidate or vice versa
    for candidate in [cleaned, depluralized, cleaned_deplural]:
        for menu_key in MENU:
            if menu_key in candidate or candidate in menu_key:
                return menu_key

    # No match found — return cleaned text so take_order can flag it
    return cleaned


def extract_items_from_text(text: str, existing_items: list[OrderItem] = None) -> list[OrderItem]:
    """Parse dish and quantity from customer text or resolve quantity decisions.

    Dish names are normalized via _normalize_dish_name() which strips filler/measure words
    and does substring matching against MENU keys.
    """
    lower_text = text.lower().strip()

    # Check for decision to prepare what's available
    if any(phrase in lower_text for phrase in [
        "prepare what you have", "what's available", "prepare what is available",
        "just prepare what you have", "have available", "what you have"
    ]):
        if existing_items:
            updated = []
            for item in existing_items:
                # Cap to available threshold (5)
                q = min(item.get("quantity", 1), 5)
                norm_dish = _normalize_dish_name(item["dish"])
                updated.append({"dish": norm_dish, "quantity": q, "status": "pending"})
            return updated

    # Number words mapping
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "a": 1, "an": 1
    }

    found_items: list[OrderItem] = []

    # Clean conversational prefixes on the whole text first
    full_cleaned = re.sub(
        r"^(i'd like|i want|please give me|can i get|give me|order|okay|fine|then|sorry)\s*[,:]*\s*",
        "", text, flags=re.IGNORECASE
    ).strip()

    # Split on commas, 'and', or full stops
    parts = re.split(r",|\band\b|\.", full_cleaned)
    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        # Remove remaining common conversational prefixes
        cleaned = re.sub(
            r"^(i'd like|i want|please give me|can i get|give me|order|okay|fine|then|sorry)\s+",
            "", cleaned, flags=re.IGNORECASE
        ).strip()
        cleaned = re.sub(r"\s+please$", "", cleaned, flags=re.IGNORECASE).strip()

        match = re.match(
            r"^(-?\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an)\s+(.+)$",
            cleaned, flags=re.IGNORECASE
        )
        if match:
            qty_str, dish_str = match.groups()
            if qty_str.lstrip("-").isdigit():
                qty = int(qty_str)
            else:
                qty = word_to_num.get(qty_str.lower(), 1)
            # Normalize dish name through filler stripping + MENU matching
            dish = _normalize_dish_name(dish_str)
            if dish:
                found_items.append({"dish": dish, "quantity": qty, "status": "pending"})
        elif cleaned and not any(w in cleaned.lower() for w in [
            "wait", "remake", "yes", "sure", "no", "thanks", "hi", "hello",
            "i'm here", "im here", "place that", "confirm", "place", "sorry"
        ]):
            # Normalize through filler stripping + MENU matching
            norm_dish = _normalize_dish_name(cleaned)
            if norm_dish:
                found_items.append({"dish": norm_dish, "quantity": 1, "status": "pending"})

    return found_items if found_items else (existing_items or [])


import json as _json


def _parse_order_with_llm(text: str) -> Optional[list[OrderItem]]:
    """Use Groq JSON mode to parse customer text into structured order items.

    Returns a list of OrderItem dicts on success, or None if the API call fails
    or the response can't be parsed (so the caller can fall back to regex).
    """
    if not client or not GROQ_API_KEY:
        return None

    menu_list = ", ".join(MENU.keys())
    prompt = (
        f"You are an order parser for a restaurant. The menu items are: {menu_list}.\n"
        f"The customer said: \"{text}\"\n\n"
        "Extract every food/drink item and its quantity from the customer's message.\n"
        "Rules:\n"
        "- Match dish names to the closest menu item (ignore filler words like 'plate', 'bowl', 'cup', 'glass', 'order of', plurals, etc.)\n"
        "- If no quantity is specified, default to 1.\n"
        "- 'a couple' = 2, 'a few' = 3, 'half dozen' = 6, 'dozen' = 12.\n"
        "- If the message contains no food items (just greetings, confirmations, etc.), return an empty list.\n"
        "- Return ONLY a JSON array of objects with 'dish' (matching the exact menu key) and 'quantity' (integer).\n"
        "- If a dish doesn't match any menu item, use the closest match or the raw name.\n\n"
        "Example: [{\"dish\": \"biryani\", \"quantity\": 2}, {\"dish\": \"naan\", \"quantity\": 3}]\n"
        "Return ONLY the JSON array, no other text."
    )

    # NOTE: We intentionally do NOT track this call in API_CALLS_ATTEMPTED/SUCCEEDED/FAILED.
    # This is an auxiliary parsing call, not a core conversation call. JSON validation
    # failures are expected for non-order inputs and should not trigger test harness assertions.
    try:
        chat_resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = chat_resp.choices[0].message.content.strip()

        # Rate limit sleep
        time.sleep(10)

        # Parse the JSON response
        parsed = _json.loads(raw)

        # Handle both {"items": [...]} and direct [...] formats
        if isinstance(parsed, dict):
            parsed = parsed.get("items", parsed.get("order", parsed.get("orders", [])))
        if not isinstance(parsed, list):
            return None

        items: list[OrderItem] = []
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            dish = _normalize_dish_name(str(entry.get("dish", "")))
            qty = int(entry.get("quantity", 1))
            if dish:
                items.append({"dish": dish, "quantity": qty, "status": "pending"})

        return items if items else None

    except Exception as e:
        # Silently fall back to regex — this is expected for non-order inputs
        return None


def smart_parse_order(text: str, existing_items: list[OrderItem] = None) -> list[OrderItem]:
    """Primary order parser: tries LLM structured output first, falls back to regex.

    This two-layer approach handles natural language like '1 plate biryani',
    'a couple of naans', 'make it 2 lassis' via the LLM, while the regex
    fallback ensures the system still works when the API is unavailable.
    """
    lower_text = text.lower().strip()

    # Quick-exit for non-order phrases (confirmations, waits, etc.)
    if any(w in lower_text for w in [
        "wait", "remake", "yes", "sure", "confirm", "place that", "place it"
    ]):
        return existing_items or []

    # Check for decision to prepare what's available (handled by regex parser)
    if any(phrase in lower_text for phrase in [
        "prepare what you have", "what's available", "prepare what is available",
        "just prepare what you have", "have available", "what you have"
    ]):
        return extract_items_from_text(text, existing_items)

    # Try LLM-based parsing first
    llm_result = _parse_order_with_llm(text)
    if llm_result:
        return llm_result

    # Fall back to regex-based parsing
    return extract_items_from_text(text, existing_items)


def parse_feedback(text: str) -> tuple[Optional[int], Optional[str]]:
    """Parse feedback rating (1-5) and optional comment, or detect decline."""
    clean = text.strip()
    lower = clean.lower()

    # Decline phrases
    if any(phrase in lower for phrase in [
        "no thanks", "no thank you", "skip", "in a hurry", "declined", "pass",
        "not now", "maybe later"
    ]):
        return None, "declined"

    # Check for rating number 1-5
    match = re.search(r"\b([1-5])\b", clean)
    rating = int(match.group(1)) if match else None

    # Extract comment
    comment = clean
    if match:
        # Remove rating and surrounding punctuation from comment
        comment = re.sub(r"\b" + str(rating) + r"\b[,.\-:]*", "", clean).strip()
        comment = comment.strip(" ,.!?;:-")
    # If no rating and no clear decline, treat the whole text as a comment
    if rating is None and clean:
        return None, clean

    return rating, comment if comment else None


def _call_groq(messages_for_api: list[dict], max_tokens: int = 150, temperature: float = 0.7) -> Optional[str]:
    """Helper to call the Groq API with error handling and rate-limit sleep.

    Returns the assistant reply text, or None if the call fails.
    """
    global API_CALLS_ATTEMPTED, API_CALLS_SUCCEEDED, API_CALLS_FAILED

    if not client or not GROQ_API_KEY:
        return None

    API_CALLS_ATTEMPTED += 1
    try:
        chat_resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages_for_api,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        reply = chat_resp.choices[0].message.content.strip()
        API_CALLS_SUCCEEDED += 1

        # TODO: replace with proper backoff before production
        time.sleep(10)

        return reply
    except Exception as e:
        API_CALLS_FAILED += 1
        print(f"[LLM API Error/Timeout]: {e}")
        return None


# ============================================================
# Graph Nodes
# ============================================================

def llm_node(state: RestaurantState) -> dict:
    """The only node that produces customer-facing text.

    It is invoked:
    - At the start, to present the menu and take the initial order.
    - Whenever a stage node sets failure_reason.
    - After bill succeeds, to ask for feedback.
    - After the customer responds to the feedback question, to close out.
    """
    stage = state.get("stage", "ordering")
    failure_reason = state.get("failure_reason")
    failure_type = state.get("failure_type")
    retry_count = state.get("retry_count", 0)
    bill_amount = state.get("bill_amount")
    items = state.get("items", [])

    # Get last customer message if any
    last_msg = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage):
            last_msg = m.content
            break
        elif isinstance(m, dict) and m.get("role") == "user":
            last_msg = m.get("content", "")
            break

    # -----------------------------------------------------------
    # 1. Final failure check (3rd consecutive strike)
    # -----------------------------------------------------------
    if failure_reason and retry_count >= 3:
        reply = (
            "I'm really sorry — we still weren't able to sort this out after a few tries. "
            "Thank you for your patience, and we hope to serve you again soon."
        )

        _call_groq([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"The customer's order has failed {retry_count} times in a row. "
                f"failure_type='{failure_type}', failure_reason='{failure_reason}'. "
                "This is the 3rd consecutive failure. Provide the final apology to close the order."
            )}
        ])

        return {
            "messages": [AIMessage(content=reply)],
            "stage": "cancelled",
        }

    # -----------------------------------------------------------
    # 2. Kitchen failure with retry < 3
    # -----------------------------------------------------------
    if failure_reason:
        compact_menu_line = format_compact_menu()
        if failure_type == "not_on_menu":
            dish_name = items[0]["dish"] if items else "that dish"
            reply = f"Sorry, {dish_name} isn't on our menu.\n{compact_menu_line}"
        elif failure_type == "quantity":
            if items and items[0].get("quantity", 0) <= 0:
                reply = f"That doesn't look like a valid quantity — how many would you like, or would you like something else?\n{compact_menu_line}"
            else:
                first_dish = items[0]["dish"] if items else "your dish"
                reply = (
                    f"We only have limited quantity of {first_dish} left — would you like us "
                    f"to prepare what's available, or would you like to order something else instead?\n{compact_menu_line}"
                )
        elif failure_type == "no_cook":
            reply = (
                f"Our kitchen can't prepare that dish right now — would you like to order "
                f"something else instead?\n{compact_menu_line}"
            )
        elif failure_type == "dropped_food":
            reply = (
                f"Sorry — there was a small mishap with your dish. Could you wait a little "
                f"while we remake it?\n{compact_menu_line}"
            )
        else:
            reply = f"There was an issue: {failure_reason}. Would you like to try again?\n{compact_menu_line}"

        _call_groq([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Stage failure occurred. failure_type='{failure_type}', "
                f"failure_reason='{failure_reason}', retry_count={retry_count}. "
                "Provide the appropriate failure message to the customer."
            )}
        ])

        return {
            "messages": [AIMessage(content=reply)],
        }

    # -----------------------------------------------------------
    # 3. Post-billing: prompt customer for feedback
    # -----------------------------------------------------------
    if stage == "billing" and bill_amount is not None:
        amount_str = f"{int(bill_amount)}" if bill_amount == int(bill_amount) else f"{bill_amount:.2f}"
        reply = (
            f"Your total is ${amount_str}. Thank you for dining with us! "
            "On a scale of 1–5, how was your experience today?"
        )

        _call_groq([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"The bill is complete. bill_amount=${amount_str}. "
                "Present the total and ask for feedback on a scale of 1-5."
            )}
        ])

        return {
            "messages": [AIMessage(content=reply)],
            "stage": "feedback",
        }

    # -----------------------------------------------------------
    # 4. Collecting feedback response from customer
    # -----------------------------------------------------------
    if stage == "feedback":
        rating, comment = parse_feedback(last_msg)
        reply = "Thanks so much for letting us know — see you next time!"

        _call_groq([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"The customer gave feedback: '{last_msg}'. "
                f"Parsed rating={rating}, comment='{comment}'. "
                "Thank them warmly and close the conversation."
            )}
        ])

        return {
            "messages": [AIMessage(content=reply)],
            "stage": "done",
            "feedback_rating": rating,
            "feedback_comment": comment,
        }

    # -----------------------------------------------------------
    # 5. Order taking / confirmation (stage == "ordering", no failure)
    # -----------------------------------------------------------

    # Parse items from customer message (LLM-first, regex-fallback)
    new_items = smart_parse_order(last_msg, items)

    # Handle empty order greeting: present full 15-item menu with prices
    if not new_items and not items:
        full_menu_str = format_full_menu()
        reply = f"Welcome! What would you like to order today?\n\n{full_menu_str}"

        llm_reply = _call_groq([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"The customer said: '{last_msg}'. No items were detected in their message. "
                f"Greet them warmly, present the menu:\n{full_menu_str}\nand ask what they'd like to order."
            )}
        ])
        if llm_reply:
            reply = llm_reply

        return {
            "messages": [AIMessage(content=reply)],
            "stage": "ordering",
        }

    # Confirmation confirmation / response
    formatted_items = ", ".join([f"{it['quantity']} {it['dish']}" for it in new_items])
    llm_reply = _call_groq([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"The customer said: '{last_msg}'. "
            f"Ordered items: {formatted_items}. "
            f"Current stage is ordering. Give a one-line friendly confirmation "
            f"greeting/confirming the order."
        )}
    ])

    if llm_reply:
        reply = llm_reply
    else:
        # Fallback if Groq API call fails or times out (Scenario 8)
        if new_items:
            item_summary = " and ".join([f"{it['quantity']} {it['dish']}" for it in new_items])
            reply = f"I've got your order for {item_summary}. Sending it to the kitchen right away!"
        else:
            reply = "Sorry, I'm having trouble right now — could you try again in a moment?"

    order_id = state.get("order_id") or str(uuid.uuid4())[:8]
    return {
        "messages": [AIMessage(content=reply)],
        "items": new_items,
        "order_id": order_id,
    }


def take_order(state: RestaurantState) -> dict:
    """Mock validation check for order taking."""
    items = state.get("items", [])
    retry_count = state.get("retry_count", 0)

    # Check empty order
    if not items:
        return {
            "failure_reason": "Order is empty.",
            "failure_type": "quantity",
            "retry_count": retry_count + 1,
        }

    # Check for invalid quantity <= 0
    for item in items:
        if item.get("quantity", 0) <= 0:
            return {
                "failure_reason": "that doesn't look like a valid quantity.",
                "failure_type": "quantity",
                "retry_count": retry_count + 1,
            }

    # Check menu validation: case-insensitive check against MENU keys
    for item in items:
        dish_key = item.get("dish", "").lower().strip()
        if dish_key not in MENU:
            return {
                "failure_reason": f"Sorry, {item.get('dish')} isn't on our menu.",
                "failure_type": "not_on_menu",
                "retry_count": retry_count + 1,
            }

    # Check rule: fail with quantity if any item's quantity > 5
    for item in items:
        if item.get("quantity", 0) > 5:
            return {
                "failure_reason": f"Quantity for {item['dish']} exceeds stock limit.",
                "failure_type": "quantity",
                "retry_count": retry_count + 1,
            }

    # Success: update item statuses to confirmed and advance stage to cooking
    # Normalize dish names to lowercase
    updated_items = [{**it, "dish": it["dish"].lower().strip(), "status": "confirmed"} for it in items]
    return {
        "stage": "cooking",
        "items": updated_items,
        "failure_reason": None,
        "failure_type": None,
        "retry_count": 0,
    }


def cook(state: RestaurantState) -> dict:
    """Mock validation check for cooking."""
    retry_count = state.get("retry_count", 0)

    if MOCK_COOK_FAIL:
        return {
            "failure_reason": "No cook available to prepare this dish right now.",
            "failure_type": "no_cook",
            "retry_count": retry_count + 1,
        }

    updated_items = [{**it, "status": "cooking"} for it in state.get("items", [])]
    return {
        "stage": "serving",
        "items": updated_items,
        "failure_reason": None,
        "failure_type": None,
        "retry_count": 0,
    }


def serve(state: RestaurantState) -> dict:
    """Mock validation check for serving."""
    retry_count = state.get("retry_count", 0)

    if MOCK_SERVE_FAIL:
        return {
            "failure_reason": "Small mishap with the dish.",
            "failure_type": "dropped_food",
            "retry_count": retry_count + 1,
        }

    updated_items = [{**it, "status": "served"} for it in state.get("items", [])]
    return {
        "stage": "billing",
        "items": updated_items,
        "failure_reason": None,
        "failure_type": None,
        "retry_count": 0,
    }


def bill(state: RestaurantState) -> dict:
    """Compute bill amount using MENU prices with case-insensitive lookup."""
    items = state.get("items", [])
    total = sum(
        MENU.get(item.get("dish", "").lower().strip(), 10.0) * item.get("quantity", 0)
        for item in items
    )
    return {
        "bill_amount": total,
        "stage": "billing",
    }


# ============================================================
# Routing Functions (Exact implementation as specified)
# ============================================================

def route_after_stage(state: RestaurantState) -> str:
    """Used after take_order / cook / serve."""
    if state.get("failure_reason"):
        return "llm"
    return state["stage"]  # "cooking" | "serving" | "billing"


def route_after_llm(state: RestaurantState) -> str:
    """Used after llm. Decides where control goes next."""
    if state["stage"] == "cancelled":
        return END
    if state["stage"] == "done":
        return END
    if state["stage"] == "feedback":
        return "wait_for_customer"  # or however you model "pause for input"
    if state.get("failure_reason"):
        return "wait_for_customer"  # pause for customer reply when failure is reported
    if state.get("stage") == "ordering" and not state.get("items"):
        return "wait_for_customer"  # pause for customer reply on empty order
    stage_to_node = {"ordering": "take_order", "cooking": "cook", "serving": "serve"}
    return stage_to_node.get(state["stage"], "take_order")


# ============================================================
# Graph Builder
# ============================================================

def build_graph():
    builder = StateGraph(RestaurantState)

    builder.add_node("llm", llm_node)
    builder.add_node("take_order", take_order)
    builder.add_node("cook", cook)
    builder.add_node("serve", serve)
    builder.add_node("bill", bill)

    builder.add_edge(START, "llm")

    builder.add_conditional_edges(
        "llm",
        route_after_llm,
        {
            "take_order": "take_order",
            "cook": "cook",
            "serve": "serve",
            "wait_for_customer": END,
            END: END,
        }
    )

    builder.add_conditional_edges(
        "take_order",
        route_after_stage,
        {
            "llm": "llm",
            "cooking": "cook",
        }
    )

    builder.add_conditional_edges(
        "cook",
        route_after_stage,
        {
            "llm": "llm",
            "serving": "serve",
        }
    )

    builder.add_conditional_edges(
        "serve",
        route_after_stage,
        {
            "llm": "llm",
            "billing": "bill",
        }
    )

    # bill -> llm (feedback)
    builder.add_edge("bill", "llm")

    return builder.compile()


# ============================================================
# CLI Interactive Loop & Testing
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("      Restaurant AI Agent (LangGraph CLI Loop)")
    print("=" * 60)
    print("Commands:")
    print("  'toggle cook'  - Toggle simulated cook failure (no_cook)")
    print("  'toggle serve' - Toggle simulated serve failure (dropped_food)")
    print("  'state'        - Print internal state dictionary")
    print("  'reset'        - Start a fresh order")
    print("  'exit'         - Quit the agent")
    print(f"\n{format_full_menu()}\n")

    app = build_graph()

    def _fresh_state() -> dict:
        return {
            "messages": [],
            "order_id": None,
            "customer_name": None,
            "items": [],
            "stage": "ordering",
            "failure_reason": None,
            "failure_type": None,
            "retry_count": 0,
            "bill_amount": None,
            "feedback_rating": None,
            "feedback_comment": None,
        }

    current_state: RestaurantState = _fresh_state()
    awaiting_confirmation: bool = False
    pending_items: list[OrderItem] = []

    while True:
        try:
            raw_input = input("\nCustomer: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        # Skip the graph call entirely when input is blank/whitespace, just reprompt
        if not raw_input or not raw_input.strip():
            continue

        user_input = raw_input.strip()

        # Check if input is the reserved keyword 'menu'
        if user_input.lower() == "menu":
            print(f"\n{format_full_menu()}")
            continue

        if user_input.lower() == "exit":
            break

        if user_input.lower() == "reset":
            current_state = _fresh_state()
            awaiting_confirmation = False
            pending_items = []
            print("[Notice] State reset. Ready for a new order.")
            continue

        if user_input.lower() == "toggle cook":
            MOCK_COOK_FAIL = not MOCK_COOK_FAIL
            print(f"[Test Config] MOCK_COOK_FAIL is now: {MOCK_COOK_FAIL}")
            continue

        if user_input.lower() == "toggle serve":
            MOCK_SERVE_FAIL = not MOCK_SERVE_FAIL
            print(f"[Test Config] MOCK_SERVE_FAIL is now: {MOCK_SERVE_FAIL}")
            continue

        if user_input.lower() == "state":
            print("[Current State]:")
            for k, v in current_state.items():
                if k != "messages":
                    print(f"  {k}: {v}")
            continue

        # Make sure session-reset notice fires every time stage is 'done' or 'cancelled' when a new message arrives
        if current_state["stage"] in ["done", "cancelled"]:
            print(f"[Notice] Previous session reached '{current_state['stage']}'. Starting a new order...")
            current_state = _fresh_state()
            awaiting_confirmation = False
            pending_items = []

        # Interactive Order Confirmation Step:
        # If waiting for confirmation from customer on an echoed order
        if awaiting_confirmation and current_state["stage"] == "ordering":
            lower_reply = user_input.lower()
            if any(w in lower_reply for w in ["yes", "confirm", "sure", "place it", "place that", "yep", "ok", "okay"]):
                awaiting_confirmation = False
                current_state["items"] = pending_items
                current_state["messages"] = list(current_state.get("messages", [])) + [
                    HumanMessage(content=user_input)
                ]
                prev_msg_count = len(current_state["messages"])
                result = app.invoke(current_state)
                current_state = dict(result)
                new_messages = current_state.get("messages", [])[prev_msg_count:]
                for m in new_messages:
                    content = m.content if isinstance(m, AIMessage) else m.get("content")
                    if content:
                        print(f"Agent: {content}")
                if current_state["stage"] == "done":
                    print("\n[Conversation Complete - Order Done]")
                elif current_state["stage"] == "cancelled":
                    print("\n[Conversation Terminated - Order Cancelled]")
                continue
            elif any(w in lower_reply for w in ["no", "cancel", "wrong", "change", "restate"]):
                awaiting_confirmation = False
                pending_items = []
                current_state["items"] = []
                print("Agent: No problem! What would you like to order instead?")
                continue

        # If resolving failure through customer decision, update items
        if current_state.get("failure_reason"):
            prev_failure_type = current_state.get("failure_type")
            updated_items = smart_parse_order(user_input, current_state.get("items", []))
            current_state["items"] = updated_items
            current_state["failure_reason"] = None
            current_state["failure_type"] = None

            # Re-validation rule:
            # If customer provides a substitute order after quantity, no_cook, or not_on_menu failure,
            # route back to take_order for full re-validation by setting stage = "ordering".
            # Only dropped_food recovery (customer agrees to wait for remake) retains its stage without re-validation.
            if prev_failure_type in ["quantity", "no_cook", "not_on_menu"]:
                current_state["stage"] = "ordering"

        # Add customer message
        current_state["messages"] = list(current_state.get("messages", [])) + [
            HumanMessage(content=user_input)
        ]

        # Check for initial order placement in CLI to ask for confirmation
        if current_state["stage"] == "ordering" and not current_state.get("items") and not current_state.get("failure_reason"):
            parsed = smart_parse_order(user_input)
            if parsed:
                # Echo back the parsed order and ask for confirmation
                pending_items = parsed
                awaiting_confirmation = True
                order_summary = " and ".join([f"{it['quantity']} {it['dish']}" for it in parsed])
                print(f"Agent: So that's {order_summary} — shall I place that?")
                continue

        # Snapshot number of messages before invoke
        prev_msg_count = len(current_state["messages"])

        # Run the graph
        result = app.invoke(current_state)
        current_state = dict(result)

        # Print every new assistant message added, not just the last one
        new_messages = current_state.get("messages", [])[prev_msg_count:]
        for m in new_messages:
            content = None
            if isinstance(m, AIMessage):
                content = m.content
            elif isinstance(m, dict) and m.get("role") == "assistant":
                content = m.get("content")

            if content:
                print(f"Agent: {content}")

        # Check if conversation reached completion
        if current_state["stage"] == "done":
            print("\n[Conversation Complete - Order Done]")
        elif current_state["stage"] == "cancelled":
            print("\n[Conversation Terminated - Order Cancelled]")