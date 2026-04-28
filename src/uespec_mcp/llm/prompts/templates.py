from __future__ import annotations

from jinja2 import Environment, PackageLoader, StrictUndefined


_ENV = Environment(
    loader=PackageLoader("uespec_mcp.llm", "prompts/templates"),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)


def render(template_name: str, **context: object) -> str:
    return _ENV.get_template(template_name).render(**context)
