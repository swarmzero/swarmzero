import logging
import os

from dotenv import load_dotenv

from swarmzero.sdk_context import SDKContext

if "OTEL_EXPORTER_OTLP_ENDPOINT" in os.environ:
    import openlit

    openlit.init()

elif "LANGTRACE_API_KEY" in os.environ:
    from langtrace_python_sdk import langtrace  # type: ignore # noqa

    langtrace.init(api_key=os.getenv("LANGTRACE_API_KEY"))

from llama_index.llms.anthropic import Anthropic
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.gemini import Gemini
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.nebius import NebiusLLM as Nebius
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

from swarmzero.config import Config
from swarmzero.llms.claude import ClaudeLLM
from swarmzero.llms.llm import LLM
from swarmzero.llms.mistral import MistralLLM
from swarmzero.llms.nebius import NebiuslLLM
from swarmzero.llms.ollama import OllamaLLM
from swarmzero.llms.openai import AzureOpenAILLM, OpenAILLM, OpenAIMultiModalLLM

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _create_llm(llm_type: str, config: Config):
    timeout = config.get("timeout")

    ollama_server_url = config.get("ollama_server_url")
    model = config.get("model")
    enable_multi_modal = config.get("enable_multi_modal")

    if llm_type == "OpenAI":
        if model.startswith("gpt-4") and enable_multi_modal is True:
            return OpenAIMultiModal(model, request_timeout=timeout, max_new_tokens=300)
        elif "gpt" in model:
            return OpenAI(model=model, request_timeout=timeout)
    elif llm_type == "AzureOpenAI":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not api_key:
            logger.error("AZURE_OPENAI_API_KEY is missing")
            raise ValueError("AZURE_OPENAI_API_KEY is required for Azure Open AI")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            logger.error("AZURE_OPENAI_ENDPOINT is missing")
            raise ValueError("AZURE_OPENAI_ENDPOINT is required for Azure Open AI")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        if not api_version:
            logger.error("AZURE_OPENAI_API_VERSION is missing")
            raise ValueError("AZURE_OPENAI_API_VERSION is required for Azure Open AI")
        if model is None:
            logger.error("model is missing")
            raise ValueError("model is required for Azure Open AI")
        azure_deployment = model.replace("azure/", "")
        return AzureOpenAI(
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            timeout=timeout,
        )
    elif llm_type == "Nebius":
        api_key = os.getenv("NEBIUS_API_KEY")
        api_base = os.getenv("NEBIUS_API_BASE")
        model = os.getenv("NEBIUS_MODEL")

        if not api_key:
            logger.error("NEBIUS_API_KEY is missing")
            raise ValueError("NEBIUS_API_KEY is required for Nebius models")
        if not api_base:
            logger.error("NEBIUS_API_BASE is missing")
            raise ValueError("NEBIUS_API_BASE is required for Nebius models")
        if not model:
            logger.error("NEBIUS_MODEL is missing")
            raise ValueError("NEBIUS_MODEL is required for Nebius models")

        return Nebius(model=model, api_key=api_key, api_base=api_base)
    elif llm_type == "Anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY is missing")
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic models")
        return Anthropic(model=model, api_key=api_key)
    elif llm_type == "Ollama":
        return Ollama(model=model, base_url=ollama_server_url, timeout=timeout)
    elif llm_type == "Mistral":
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            logger.error("MISTRAL_API_KEY is missing")
            raise ValueError("MISTRAL_API_KEY is required for Mistral models")
        return MistralAI(model=model, api_key=api_key)
    elif llm_type == "Gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY is missing")
            raise ValueError("GEMINI_API_KEY is required for Gemini models")
        else:
            return Gemini(model='models/' + model, api_key=api_key)
    else:
        logger.error("Unsupported LLM type")
        raise ValueError("Unsupported LLM type")


# used when `llm` is provided in Agent or Swarm creation
def llm_from_wrapper(llm_wrapper: LLM, config: Config):
    if isinstance(llm_wrapper, OpenAILLM):
        return _create_llm("OpenAI", config)
    if isinstance(llm_wrapper, AzureOpenAILLM):
        return _create_llm("AzureOpenAI", config)
    elif isinstance(llm_wrapper, ClaudeLLM):
        return _create_llm("Anthropic", config)
    elif isinstance(llm_wrapper, OllamaLLM):
        return _create_llm("Ollama", config)
    elif isinstance(llm_wrapper, MistralLLM):
        return _create_llm("Mistral", config)
    elif isinstance(llm_wrapper, NebiuslLLM):
        return _create_llm("Nebius", config)
    else:
        logger.error("Unsupported LLM wrapper type")
        raise ValueError("Unsupported LLM wrapper type")


# default to config file if `llm` is not provided in Agent or Swarm creation
def llm_from_config(config: Config):
    model = config.get("model")

    if "gpt" in model and "azure/" not in model:
        logger.info("OpenAI model selected")
        return _create_llm("OpenAI", config)
    elif "azure/" in model:
        logger.info("AzureOpenAI model selected")
        return _create_llm("AzureOpenAI", config)
    elif "claude" in model:
        logger.info("Claude model selected")
        return _create_llm("Anthropic", config)
    elif "llama" in model:
        logger.info("Llama model selected")
        return _create_llm("Ollama", config)
    elif any(keyword in model for keyword in ["mixtral", "mistral", "codestral"]):
        logger.info("Mistral model selected")
        return _create_llm("Mistral", config)
    elif "nebius" in model:
        logger.info("Nebius model selected")
        return _create_llm("Nebius", config)
    else:
        logger.info("Default OpenAI model selected")
        return _create_llm("OpenAI", config)


def llm_from_config_without_agent(config: Config, sdk_context: SDKContext):
    model = config.get("model")
    enable_multi_modal = config.get("enable_multi_modal")

    if "gpt" in model and "azure/" not in model:
        if model.startswith("gpt-4") and enable_multi_modal is True:
            logger.info("OpenAIMultiModalLLM selected")
            return OpenAIMultiModalLLM(llm=llm_from_config(config), sdk_context=sdk_context)
        logger.info("OpenAILLM selected")
        return OpenAILLM(llm=llm_from_config(config), sdk_context=sdk_context)
    elif "azure/" in model:
        logger.info("AzureOpenAILLM model selected")
        return AzureOpenAILLM(llm=llm_from_config(config), sdk_context=sdk_context)
    elif "claude" in model:
        logger.info("ClaudeLLM selected")
        return ClaudeLLM(llm=llm_from_config(config), sdk_context=sdk_context)
    elif "llama" in model:
        logger.info("OllamaLLM selected")
        return OllamaLLM(llm=llm_from_config(config), sdk_context=sdk_context)
    elif any(keyword in model for keyword in ["mixtral", "mistral", "codestral"]):
        logger.info("MistralLLM selected")
        return MistralLLM(llm=llm_from_config(config), sdk_context=sdk_context)
    elif "nebius" in model:
        logger.info("NebiuslLLM selected")
        return NebiuslLLM(llm=llm_from_config(config), sdk_context=sdk_context)
    else:
        logger.info("Default OpenAILLM selected")
        return OpenAILLM(llm=llm_from_config(config), sdk_context=sdk_context)

    return llm_from_config(config)
