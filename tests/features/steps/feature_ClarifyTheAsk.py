from behave import given, when, then  # type: ignore
from process_analyst_copilot import ClarifyTheAsk


@given("I have input data")
def step_given_data(context):
    context.input_data = ...  # Set up your input data


@when("I process the data with the AI agent")
def step_process_data(context):
    context.result = your_agent_method(context.input_data)


@then("the response should be as expected")
def step_check_result(context):
    assert context.result == expected_output


@given('I have a request to "{input_ask}"')
def step_impl_given_request(context, input_ask):
    context.input_ask = input_ask


@when("I document the process steps")
def step_impl_when_document_process(context):
    context.result = draft_process(context.input_ask)


@then("I should get the following steps:")
def step_impl_then_verify_steps(context):
    expected_steps = [
        "Step 1: Boil water",
        "Step 2: Add tea leaves to cup",
        "Step 3: Pour hot water into cup",
        "Step 4: Steep tea for desired time",
        "Step 5: Remove tea bag or leaves",
        "Step 6: Serve tea",
    ]
    assert (
        context.result == expected_steps
    ), f"Expected {expected_steps}, but got {context.result}"
