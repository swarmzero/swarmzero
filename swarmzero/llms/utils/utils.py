import logging
import os

from dotenv import load_dotenv

if "LANGTRACE_API_KEY" in os.environ:
    from langtrace_python_sdk import langtrace  # type: ignore # noqa

    langtrace.init(api_key=os.getenv("LANGTRACE_API_KEY"))

from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini
from llama_index.llms.mistralai import MistralAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

from swarmzero.config import Config
from swarmzero.llms.claude import ClaudeLLM
from swarmzero.llms.llm import LLM
from swarmzero.llms.mistral import MistralLLM
from swarmzero.llms.ollama import OllamaLLM
from swarmzero.llms.openai import OpenAILLM

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
    elif isinstance(llm_wrapper, ClaudeLLM):
        return _create_llm("Anthropic", config)
    elif isinstance(llm_wrapper, OllamaLLM):
        return _create_llm("Ollama", config)
    elif isinstance(llm_wrapper, MistralLLM):
        return _create_llm("Mistral", config)
    else:
        logger.error("Unsupported LLM wrapper type")
        raise ValueError("Unsupported LLM wrapper type")


# default to config file if `llm` is not provided in Agent or Swarm creation
def llm_from_config(config: Config):
    model = config.get("model")

    if "gpt" in model:
        logger.info("OpenAI model selected")
        return _create_llm("OpenAI", config)
    elif "claude" in model:
        logger.info("Claude model selected")
        return _create_llm("Anthropic", config)
    elif "llama" in model:
        logger.info("Llama model selected")
        return _create_llm("Ollama", config)
    elif any(keyword in model for keyword in ["mixtral", "mistral", "codestral"]):
        logger.info("Mistral model selected")
        return _create_llm("Mistral", config)
    else:
        logger.info("Default OpenAI model selected")
        return _create_llm("OpenAI", config)


def llm_from_config_without_agent(config: Config):
    model = config.get("model")
    if "gpt" in model:
        logger.info("OpenAILLM selected")
        return OpenAILLM(llm=llm_from_config(config))
    elif "claude" in model:
        logger.info("ClaudeLLM selected")
        return ClaudeLLM(llm=llm_from_config(config))
    elif "llama" in model:
        logger.info("OllamaLLM selected")
        return OllamaLLM(llm=llm_from_config(config))
    elif any(keyword in model for keyword in ["mixtral", "mistral", "codestral"]):
        logger.info("MistralLLM selected")
        return MistralLLM(llm=llm_from_config(config))
    else:
        logger.info("Default OpenAILLM selected")
        return OpenAILLM(llm=llm_from_config(config))

    return llm_from_config(config)
