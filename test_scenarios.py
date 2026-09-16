import sys
from contextlib import contextmanager
import restaurant_agent

@contextmanager
def track_api_calls(test_name: str, expect_failures: bool = False):
    """
    Test harness tracking context manager:
    - Records API calls attempted, succeeded, and failed during a test.
    - Prints a per-test summary line showing real-LLM-vs-fallback usage.
    - Asserts zero API failures for normal tests (Tests 1-7, 9).
    - Asserts at least one API failure for Test 8 (Groq API failure / graceful degradation).
    """
    prev_att = restaurant_agent.API_CALLS_ATTEMPTED
    prev_suc = restaurant_agent.API_CALLS_SUCCEEDED
    prev_fail = restaurant_agent.API_CALLS_FAILED

    try:
        yield
    finally:
        delta_att = restaurant_agent.API_CALLS_ATTEMPTED - prev_att
        delta_suc = restaurant_agent.API_CALLS_SUCCEEDED - prev_suc
        delta_fail = restaurant_agent.API_CALLS_FAILED - prev_fail

        # Summary line for real-LLM vs fallback usage
        print(f"[{test_name} LLM Tracking] Real-LLM Succeeded: {delta_suc} | "
              f"API Failures / Fallbacks: {delta_fail} | Total Calls: {delta_att}")

        if expect_failures:
            assert delta_fail > 0, (
                f"AssertionError in {test_name}: Expected at least one API failure for degradation test, "
                f"but recorded {delta_fail} failures (attempted: {delta_att}, succeeded: {delta_suc})."
            )
        else:
            assert delta_fail == 0, (
                f"AssertionError in {test_name}: API integration failed! Recorded {delta_fail} API failures "
                f"(attempted: {delta_att}, succeeded: {delta_suc}). Broken API calls must not pass silently!"
            )


def run_tests():
    print("Testing Graph and Scenarios with API Health Assertions...")
    print(f"Configured Model: {restaurant_agent.LLM_MODEL}")
    app = restaurant_agent.build_graph()

    # ============================================================
    # 1. Happy path: 2 butter chicken ($12) + 1 naan ($3) = $27
    # ============================================================
    print("\n--- Test 1: Happy Path ---")
    with track_api_calls("Test 1 (Happy Path)", expect_failures=False):
        state = {
            "messages": [restaurant_agent.HumanMessage(content="I'd like 2 butter chicken and 1 naan.")],
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
        res = app.invoke(state)
        print("After Turn 1 & 2 (auto bill):")
        print("  stage:", res["stage"])
        print("  bill_amount:", res["bill_amount"])
        print("  last message:", res["messages"][-1].content)
        assert res["stage"] == "feedback", f"Expected feedback stage, got {res['stage']}"
        assert res["bill_amount"] == 27.0, f"Expected 27.0, got {res['bill_amount']}"

        # Feedback response
        res["messages"] = [restaurant_agent.HumanMessage(content="5, great service!")]
        res2 = app.invoke(res)
        print("After Turn 3:")
        print("  stage:", res2["stage"])
        print("  feedback_rating:", res2["feedback_rating"])
        print("  feedback_comment:", res2["feedback_comment"])
        print("  last message:", res2["messages"][-1].content)
        assert res2["stage"] == "done", f"Expected done stage, got {res2['stage']}"
        assert res2["feedback_rating"] == 5
        assert res2["feedback_comment"] == "great service"
    print("Test 1 Passed!")

    # ============================================================
    # 2. Quantity failure, resolved: 5 butter chicken @ $12 = $60
    # ============================================================
    print("\n--- Test 2: Quantity failure ---")
    with track_api_calls("Test 2 (Quantity failure)", expect_failures=False):
        state_qty = {
            "messages": [restaurant_agent.HumanMessage(content="I'd like 8 butter chicken.")],
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
        res_q1 = app.invoke(state_qty)
        print("Turn 1/2 response:")
        print("  stage:", res_q1["stage"])
        print("  failure_type:", res_q1["failure_type"])
        print("  retry_count:", res_q1["retry_count"])
        print("  last message:", res_q1["messages"][-1].content)
        assert res_q1["failure_type"] == "quantity"
        assert res_q1["retry_count"] == 1
        assert "limited quantity of butter chicken left" in res_q1["messages"][-1].content

        # Resolution
        res_q1["items"] = restaurant_agent.extract_items_from_text("Just prepare what you have.", res_q1["items"])
        res_q1["failure_reason"] = None
        res_q1["failure_type"] = None
        res_q1["messages"] = [restaurant_agent.HumanMessage(content="Just prepare what you have.")]
        res_q2 = app.invoke(res_q1)
        print("Turn 3 response:")
        print("  stage:", res_q2["stage"])
        print("  retry_count:", res_q2["retry_count"])
        print("  failure_reason:", res_q2["failure_reason"])
        print("  bill_amount:", res_q2["bill_amount"])
        assert res_q2["retry_count"] == 0
        assert res_q2["failure_reason"] is None
        assert res_q2["stage"] == "feedback"
        assert res_q2["bill_amount"] == 60.0, f"Expected 60.0, got {res_q2['bill_amount']}"
    print("Test 2 Passed!")

    # ============================================================
    # 3. No-cook failure, 3 strikes -> cancellation (no bill)
    # ============================================================
    print("\n--- Test 3: No-cook 3 strikes ---")
    restaurant_agent.MOCK_COOK_FAIL = True
    try:
        with track_api_calls("Test 3 (No-cook 3 strikes)", expect_failures=False):
            s3 = {
                "messages": [restaurant_agent.HumanMessage(content="I'd like a pizza.")],
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
            # Turn 1: strike 1
            r1 = app.invoke(s3)
            assert r1["retry_count"] == 1
            assert r1["failure_type"] == "no_cook"
            print("Strike 1:", r1["messages"][-1].content)

            # Turn 2: strike 2
            r1["items"] = [{"dish": "pasta", "quantity": 1, "status": "pending"}]
            r1["failure_reason"] = None
            r1["failure_type"] = None
            r1["messages"] = [restaurant_agent.HumanMessage(content="Okay, a pasta then.")]
            r2 = app.invoke(r1)
            assert r2["retry_count"] == 2
            print("Strike 2:", r2["messages"][-1].content)

            # Turn 3: strike 3
            r2["items"] = [{"dish": "salad", "quantity": 1, "status": "pending"}]
            r2["failure_reason"] = None
            r2["failure_type"] = None
            r2["messages"] = [restaurant_agent.HumanMessage(content="Fine, a salad.")]
            r3 = app.invoke(r2)
            assert r3["retry_count"] == 3
            assert r3["stage"] == "cancelled"
            print("Strike 3 (cancelled):", r3["messages"][-1].content)
            assert "I'm really sorry" in r3["messages"][-1].content
        print("Test 3 Passed!")
    finally:
        restaurant_agent.MOCK_COOK_FAIL = False

    # ============================================================
    # 4. Dropped food, single recovery: 1 pad thai @ $10 = $10
    # ============================================================
    print("\n--- Test 4: Dropped food recovery ---")
    restaurant_agent.MOCK_SERVE_FAIL = True
    try:
        with track_api_calls("Test 4 (Dropped food)", expect_failures=False):
            s4 = {
                "messages": [restaurant_agent.HumanMessage(content="One pad thai please.")],
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
            r1 = app.invoke(s4)
            assert r1["failure_type"] == "dropped_food"
            assert r1["retry_count"] == 1
            print("Dropped food message:", r1["messages"][-1].content)

            # Recover
            restaurant_agent.MOCK_SERVE_FAIL = False
            r1["failure_reason"] = None
            r1["failure_type"] = None
            r1["messages"] = [restaurant_agent.HumanMessage(content="Sure, I'll wait.")]
            r2 = app.invoke(r1)
            assert r2["failure_reason"] is None
            assert r2["retry_count"] == 0
            assert r2["stage"] == "feedback"
            assert r2["bill_amount"] == 10.0, f"Expected 10.0, got {r2['bill_amount']}"
            print("Recovered bill/feedback message:", r2["messages"][-1].content)
        print("Test 4 Passed!")
    finally:
        restaurant_agent.MOCK_SERVE_FAIL = False

    # ============================================================
    # 5. Feedback declined: 2 paneer tikka @ $10 = $20
    # ============================================================
    print("\n--- Test 5: Feedback declined ---")
    with track_api_calls("Test 5 (Feedback declined)", expect_failures=False):
        s5 = {
            "messages": [restaurant_agent.HumanMessage(content="No thanks, I'm in a hurry.")],
            "order_id": "test1234",
            "customer_name": None,
            "items": [{"dish": "paneer tikka", "quantity": 2, "status": "served"}],
            "stage": "feedback",
            "failure_reason": None,
            "failure_type": None,
            "retry_count": 0,
            "bill_amount": 20.0,
            "feedback_rating": None,
            "feedback_comment": None,
        }
        r5 = app.invoke(s5)
        print("Declined feedback response:", r5["messages"][-1].content)
        assert r5["stage"] == "done"
        assert r5["bill_amount"] == 20.0, f"Expected 20.0, got {r5['bill_amount']}"
        assert r5["feedback_rating"] is None
        assert r5["feedback_comment"] == "declined"
    print("Test 5 Passed!")

    # ============================================================
    # 6. Empty order: 2 butter chicken @ $12 = $24
    # ============================================================
    print("\n--- Test 6: Empty order ---")
    with track_api_calls("Test 6 (Empty order)", expect_failures=False):
        s6 = {
            "messages": [restaurant_agent.HumanMessage(content="Hi, I'm here.")],
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
        r6_1 = app.invoke(s6)
        print("Turn 1 response:", r6_1["messages"][-1].content)
        assert r6_1["items"] == []
        assert r6_1["stage"] == "ordering"

        # Turn 2: Provide valid order
        r6_1["messages"] = [restaurant_agent.HumanMessage(content="2 butter chicken please.")]
        r6_2 = app.invoke(r6_1)
        print("Turn 2 response:", r6_2["messages"][-1].content)
        assert len(r6_2["items"]) == 1
        assert r6_2["items"][0]["dish"] == "butter chicken"
        assert r6_2["items"][0]["quantity"] == 2
        assert r6_2["stage"] == "feedback"
        assert r6_2["bill_amount"] == 24.0, f"Expected 24.0, got {r6_2['bill_amount']}"
    print("Test 6 Passed!")

    # ============================================================
    # 7. Invalid quantity (<= 0): 2 pizzas @ $9 = $18
    # ============================================================
    print("\n--- Test 7: Invalid quantity ---")
    with track_api_calls("Test 7 (Invalid quantity)", expect_failures=False):
        s7 = {
            "messages": [restaurant_agent.HumanMessage(content="I'd like 0 pizza.")],
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
        r7_1 = app.invoke(s7)
        print("Turn 1 response:", r7_1["messages"][-1].content)
        assert r7_1["failure_type"] == "quantity"
        assert r7_1["retry_count"] == 1
        assert "That doesn't look like a valid quantity" in r7_1["messages"][-1].content

        # Turn 3: Fix quantity
        r7_1["items"] = [{"dish": "pizza", "quantity": 2, "status": "pending"}]
        r7_1["failure_reason"] = None
        r7_1["failure_type"] = None
        r7_1["messages"] = [restaurant_agent.HumanMessage(content="Sorry, 2 pizza.")]
        r7_2 = app.invoke(r7_1)
        print("Turn 3 response:", r7_2["messages"][-1].content)
        assert r7_2["retry_count"] == 0
        assert r7_2["failure_reason"] is None
        assert r7_2["stage"] == "feedback"
        assert r7_2["bill_amount"] == 18.0, f"Expected 18.0, got {r7_2['bill_amount']}"
    print("Test 7 Passed!")

    # ============================================================
    # 8. Groq API failure (graceful degradation): 1 pizza @ $9 = $9
    # ============================================================
    print("\n--- Test 8: Groq API failure graceful degradation ---")
    orig_client = restaurant_agent.client
    try:
        # Mock client to raise an exception
        class MockFailingClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(*args, **kwargs):
                        raise Exception("Simulated connection timeout")

        restaurant_agent.client = MockFailingClient()
        with track_api_calls("Test 8 (API Failure Degradation)", expect_failures=True):
            s8 = {
                "messages": [restaurant_agent.HumanMessage(content="I'd like a pizza.")],
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
            r8 = app.invoke(s8)
            print("Fallback message:", r8["messages"][-1].content)
            assert len(r8["messages"]) > 0
            assert r8["bill_amount"] == 9.0, f"Expected 9.0, got {r8['bill_amount']}"
        print("Test 8 Passed!")
    finally:
        restaurant_agent.client = orig_client

    # ============================================================
    # 9. Non-numeric feedback: 1 fried rice @ $8 = $8
    # ============================================================
    print("\n--- Test 9: Non-numeric feedback ---")
    with track_api_calls("Test 9 (Non-numeric feedback)", expect_failures=False):
        s9 = {
            "messages": [restaurant_agent.HumanMessage(content="It was okay I guess, a bit slow.")],
            "order_id": "test789",
            "customer_name": None,
            "items": [{"dish": "fried rice", "quantity": 1, "status": "served"}],
            "stage": "feedback",
            "failure_reason": None,
            "failure_type": None,
            "retry_count": 0,
            "bill_amount": 8.0,
            "feedback_rating": None,
            "feedback_comment": None,
        }
        r9 = app.invoke(s9)
        print("Non-numeric feedback response:", r9["messages"][-1].content)
        assert r9["stage"] == "done"
        assert r9["bill_amount"] == 8.0, f"Expected 8.0, got {r9['bill_amount']}"
        assert r9["feedback_rating"] is None
        assert r9["feedback_comment"] == "It was okay I guess, a bit slow."
    print("Test 9 Passed!")

    print("\n" + "=" * 60)
    print("ALL 9 TEST SCENARIOS PASSED WITH VERIFIED API INTEGRATION!")
    print(f"Total API Calls Attempted: {restaurant_agent.API_CALLS_ATTEMPTED}")
    print(f"Total API Calls Succeeded: {restaurant_agent.API_CALLS_SUCCEEDED}")
    print(f"Total API Calls Failed (Test 8 degradation): {restaurant_agent.API_CALLS_FAILED}")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
