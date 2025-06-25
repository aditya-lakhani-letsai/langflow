import re

from langflow.components.logic.conditional_router import ConditionalRouterComponent
from langflow.io import (
    BoolInput,
    DropdownInput,
    IntInput,
    MessageInput,
    MessageTextInput,
    Output,
)
from langflow.schema.message import Message


class LetsAIConditionalRouterComponent(ConditionalRouterComponent):
    """
    A conditional router component for Langflow, extending the original ConditionalRouterComponent.
    Routes an input message based on string comparison between input text and match text, with custom outputs for true and false paths.
    Supports equals, not equals, contains, starts with, ends with, and regex operations.
    """

    display_name = "LetsAI If-Else"
    description = "Routes an input message to a corresponding output based on text comparison with custom true/false outputs."
    icon = "split"
    name = "LetsAIConditionalRouter"

    inputs = ConditionalRouterComponent.inputs + [
        MessageTextInput(
            name="true_output",
            display_name="True Output",
            info="Result when condition evaluates to True.",
            required=True,
        ),
        MessageTextInput(
            name="false_output",
            display_name="False Output",
            info="Result when condition evaluates to False.",
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="True", name="true_result", method="true_response", group_outputs=True),
        Output(display_name="False", name="false_result", method="false_response", group_outputs=True),
    ]

    def true_response(self) -> Message:
        """
        Returns the true_output if condition is True, else fallback or message.
        """
        result = self.evaluate_condition(
            self.input_text, self.match_text, self.operator, case_sensitive=self.case_sensitive
        )
        if result:
            self.status = self.true_output
            self.log(f"Condition met. Routing to True: {self.true_output}")
            self.iterate_and_stop_once("false_result")
            return self.true_output

        self.log("Condition not met. Routing to fallback from true_response.")
        self.iterate_and_stop_once("true_result")
        return Message(content=self.message)

    def false_response(self) -> Message:
        """
        Returns the false_output if condition is False, else fallback or message.
        """
        result = self.evaluate_condition(
            self.input_text, self.match_text, self.operator, case_sensitive=self.case_sensitive
        )
        if not result:
            self.status = self.false_output
            self.log(f"Condition NOT met. Routing to False: {self.false_output}")
            self.iterate_and_stop_once("true_result")
            return self.false_output

        self.log("Condition was True. Routing to fallback from false_response.")
        self.iterate_and_stop_once("false_result")
        return Message(content=self.message)