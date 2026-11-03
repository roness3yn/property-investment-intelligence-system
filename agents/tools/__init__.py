"""Read-only tools the agents can call. See docs/agents.md for the catalogue."""
from __future__ import annotations

from agents.config import AgentConfig
from agents.tools.base import ToolContext, ToolSpec
from agents.tools.data_tools import (data_availability_tool, neighbourhood_stats_tool,
                                     review_signals_tool)
from agents.tools.evidence_tools import (listing_profile_tool, model_card_tool,
                                         segment_profile_tool)
from agents.tools.model_tools import predict_listing_price_tool, rank_opportunities_tool
from agents.tools.roi import roi_calculator_tool


def web_search_spec(config: AgentConfig) -> ToolSpec:
    """Anthropic's server-side web search, for outside context (regulation, market news).
    The API runs it; results are external context, never inputs to the score."""
    schema = {"type": "web_search_20250305", "name": "web_search",
              "max_uses": config.web_search_max_uses}
    return ToolSpec("web_search", schema, None, server_tool=True, external=True)


def build_registry(ctx: ToolContext) -> dict[str, ToolSpec]:
    specs = [rank_opportunities_tool(ctx), predict_listing_price_tool(ctx), model_card_tool(ctx),
             segment_profile_tool(ctx), listing_profile_tool(ctx),
             neighbourhood_stats_tool(ctx), review_signals_tool(ctx),
             data_availability_tool(ctx), roi_calculator_tool(), web_search_spec(ctx.config)]
    return {s.name: s for s in specs}


__all__ = ["ToolContext", "ToolSpec", "build_registry"]
