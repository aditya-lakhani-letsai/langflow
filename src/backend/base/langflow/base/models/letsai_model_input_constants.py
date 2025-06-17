from langflow.base.models.model_input_constants import *
from langflow.components.languagemodels.letsai_azure_openai import LetsAIAzureChatOpenAIComponent
from langflow.components.languagemodels.azure_openai import AzureChatOpenAIComponent
from copy import deepcopy


# Override MODEL_PROVIDERS_DICT to replace Azure OpenAI with LetsAIAzureChatOpenAIComponent
MODEL_PROVIDERS_DICT = deepcopy(MODEL_PROVIDERS_DICT)
try:
    azure_inputs = get_filtered_inputs(LetsAIAzureChatOpenAIComponent)
    azure_fields = create_input_fields_dict(azure_inputs, "")
    MODEL_PROVIDERS_DICT["LetsAI Azure OpenAI"] = {
        "fields": azure_fields,
        "inputs": azure_inputs,
        "prefix": "",
        "component_class": LetsAIAzureChatOpenAIComponent(),
        "icon": AzureChatOpenAIComponent.icon,
    }
    # Remove the original Azure OpenAI entry if it exists
    MODEL_PROVIDERS_DICT.pop("Azure OpenAI", None)
except ImportError as e:
    msg = "LetsAI Azure OpenAI is not installed. Please ensure langchain-openai is installed."
    raise ImportError(msg) from e

# Update related variables
MODEL_PROVIDERS = list(MODEL_PROVIDERS_DICT.keys())
ALL_PROVIDER_FIELDS = [field for provider in MODEL_PROVIDERS_DICT.values() for field in provider["fields"]]
MODELS_METADATA = {
    key: {"icon": MODEL_PROVIDERS_DICT[key]["icon"] if key in MODEL_PROVIDERS_DICT else None}
    for key in MODEL_PROVIDERS_DICT
}