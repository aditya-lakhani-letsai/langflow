from langchain_openai import AzureChatOpenAI
from langflow.components.languagemodels import AzureChatOpenAIComponent
from langflow.field_typing import LanguageModel
from langflow.io import BoolInput
from typing_extensions import override


class LetsAIAzureChatOpenAIComponent(AzureChatOpenAIComponent):
    """LetsAI Azure OpenAI Component with JSON object response format support.

    Extends the Azure OpenAI LLM to include an option for JSON object response format.
    """

    display_name: str = "LetsAI Azure OpenAI"
    description: str = "Generate text using Azure OpenAI LLMs with optional JSON response format."
    name = "LetsaiAzureOpenAIModel"

    inputs = AzureChatOpenAIComponent.inputs + [
        BoolInput(
            name="json_object",
            display_name="JSON Object",
            value=False,
            info="Enable JSON object response format for structured output.",
        ),
    ]

    @override
    def build_model(self) -> LanguageModel:
        """Builds the AzureChatOpenAI model with optional JSON object response format."""
        azure_endpoint = self.azure_endpoint
        azure_deployment = self.azure_deployment
        api_version = self.api_version
        api_key = self.api_key
        temperature = self.temperature
        max_tokens = self.max_tokens
        stream = self.stream

        try:
            if self.json_object:
                output = AzureChatOpenAI(
                    azure_endpoint=azure_endpoint,
                    azure_deployment=azure_deployment,
                    api_version=api_version,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens or None,
                    streaming=stream,
                    response_format={"type": "json_object"},
                )
            else:
                output = AzureChatOpenAI(
                    azure_endpoint=azure_endpoint,
                    azure_deployment=azure_deployment,
                    api_version=api_version,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens or None,
                    streaming=stream,
                )
        except Exception as e:
            msg = f"Could not connect to AzureOpenAI API: {e}"
            raise ValueError(msg) from e

        return output