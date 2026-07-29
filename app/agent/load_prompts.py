import yaml
from app.config import settings

def load_prompts() -> dict[str, str]:
    with open(settings.PROMPTS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

prompts = load_prompts()