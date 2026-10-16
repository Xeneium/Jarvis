"""NativeReActAgent -- Thought-Action-Observation loop agent.

Renamed from ``ReActAgent`` to clarify this is OpenJarvis's native
implementation, not an integration with an external project.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, List, Optional

from openjarvis.agents._stubs import AgentContext, AgentResult, ToolUsingAgent
from openjarvis.agents.prompt_loader import (
    load_few_shot_exemplars,
    load_system_prompt_override,
)
from openjarvis.core.events import EventBus
from openjarvis.core.registry import AgentRegistry
from openjarvis.core.types import Message, Role, ToolCall, ToolResult, _message_to_dict
from openjarvis.engine._stubs import InferenceEngine
from openjarvis.tools._stubs import BaseTool, build_tool_descriptions

logger = logging.getLogger(__name__)

REACT_SYSTEM_PROMPT = """\
You are a ReAct agent. For each step, respond with exactly one of:

1. To think and act:
Thought: <your reasoning>
Action: <tool_name>
Action Input: <json arguments>

2. To give a final answer:
Thought: <your reasoning>
Final Answer: <your answer>

# Current Temporal Reference (Ground Truth Clock)
- Current Date and Time: {current_time_str}
- Reference Timestamp (UTC): {current_time_utc}
- Today is: {current_day}, {current_date_str}
- TEMPORAL GROUNDING RULES: All user queries referring to "latest", "recent", "today", "yesterday", "this week", or "current" MUST be anchored to {current_date_str}. Under no circumstances should you describe articles or events from past weeks or months as "today" or "recent" without explicitly citing their true publication date.

# Deep View Web Research Exclusivity & Real-Time News Policy
- MANDATORY TOOL USAGE: You DO NOT possess real-time or current news in your pretraining weights. For ANY query asking about recent AI news, updates, announcements, developments, newsletters, or current events (such as "What is the latest news in AI?", "What are the recent updates in AI?"), you MUST FIRST invoke `deep_view_search` with action="search" (query="AI" or the relevant topic) or action="latest_newsletters".
- DO NOT answer news queries from your memory. You MUST NEVER invent, guess, hallucinate, or fabricate articles, dates, or URLs. Every news article, date, and URL MUST originate from a real `deep_view_search` observation.
- The Deep View (https://www.thedeepview.com) is your SOLE and EXCLUSIVE web research source. Never consult, mention, or use Google, Bing, DuckDuckGo, Tavily, or other websites.
- ONLY answer directly without calling tools for general knowledge questions (e.g. coding concepts, mathematics, general science).
- When answering from a `deep_view_search` observation:
  1. Your Final Answer MUST directly synthesize the concrete news stories, company announcements, model releases, and key takeaways reported in the observation.
  2. Organize your answer into clear bullet points highlighting what occurred, the companies/models involved, and key details.
  3. Include the exact source URL and publication date for each story from The Deep View observation.
  4. NEVER output meta-commentary, placeholders, or vague statements telling the user to read the links. Deliver the real news content directly in your response.
- If The Deep View does not contain information on the requested topic, state clearly that no coverage was found on The Deep View.
- Clearly distinguish facts retrieved from The Deep View from your own general knowledge or reasoning.

# Using Skills

Tools whose names begin with `skill_` are SKILLS. When you call a skill tool,
the response can take one of two forms:

- **Computed result**: The skill ran a deterministic pipeline and returned a
  value (number, string, JSON, etc.). Use the value directly in your answer.

- **Procedural instructions**: The skill returned markdown text describing
  HOW to accomplish a task. Recognize this when the response starts with
  `#` headings, contains bullet lists, or uses phrases like "When asked
  to...", "First...", "Steps:". When you receive instructions:
  1. READ the instructions carefully — they are your playbook
  2. FOLLOW the steps using your OTHER tools (e.g. calculator, web_search,
     shell_exec, file_read) — not the same skill
  3. DO NOT call the same skill again — you already have its instructions
  4. Synthesize a Final Answer from what you learned

{skill_examples}{tool_descriptions}"""


def _is_time_sensitive_news_query(query: str) -> bool:
    """Detect if a user prompt is requesting latest or real-time news/updates."""
    q = query.lower()
    time_markers = (
        "latest",
        "recent",
        "recently",
        "today",
        "yesterday",
        "this week",
        "current",
        "breaking",
        "what's new",
        "whats new",
        "newsletter",
        "news",
        "updates",
        "update",
        "developments",
    )
    topic_markers = (
        "ai",
        "artificial intelligence",
        "llm",
        "model",
        "tech",
        "technology",
        "deep view",
        "deepview",
        "openai",
        "anthropic",
        "google",
        "meta",
        "apple",
        "microsoft",
    )
    has_time = any(m in q for m in time_markers)
    has_topic = any(m in q for m in topic_markers)
    has_news_phrase = any(
        phrase in q
        for phrase in (
            "what is the latest",
            "what are the latest",
            "what's the latest",
            "whats the latest",
            "latest news",
            "recent news",
            "recent updates",
            "latest updates",
            "ai news",
            "news in ai",
            "updates on ai",
            "deep view",
            "deepview",
        )
    )
    return has_news_phrase or (has_time and has_topic)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError("Non-JSON numeric constant")


def _parse_json_response(text: str) -> dict[str, str]:
    """Parse a complete JSON envelope without recovering partial tool calls."""
    result = {"thought": "", "action": "", "action_input": "", "final_answer": ""}
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        if not isinstance(payload, dict):
            raise ValueError("Expected one JSON object")
        fields = {}
        for key, value in payload.items():
            name = key.strip().lower().replace(" ", "_")
            if name in result:
                if name in fields:
                    raise ValueError("Ambiguous ReAct JSON fields")
                fields[name] = value
        # Ordinary JSON answers without ReAct fields remain plain answers.
        if not fields:
            return result
        for name in ("thought", "action", "final_answer"):
            value = fields.get(name)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError(f"{name} must be a string")
                result[name] = value.strip()
        if result["action"] and result["final_answer"]:
            raise ValueError("Expected an action or a final answer, not both")
        if result["final_answer"]:
            return result
        if not result["action"]:
            raise ValueError("ReAct JSON needs an action or a final answer")
        arguments = fields.get("action_input", {})
        if isinstance(arguments, str):
            arguments = json.loads(
                arguments,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        if not isinstance(arguments, dict):
            raise ValueError("action_input must be a JSON object")
        result["action_input"] = json.dumps(
            arguments, ensure_ascii=False, allow_nan=False
        )
        return result
    except (ValueError, RecursionError):
        # Do not expose JSON decoder messages or salvage a partial action: the
        # response may contain tool arguments, secrets, or quoted action markers.
        return {
            "thought": "",
            "action": "",
            "action_input": "",
            "final_answer": "",
            "parse_error": (
                "Expected one JSON object with either a nonempty action and "
                "object action_input, or a nonempty final_answer."
            ),
        }


@AgentRegistry.register("native_react")
class NativeReActAgent(ToolUsingAgent):
    """ReAct agent: Thought -> Action -> Observation loop."""

    agent_id = "native_react"
    _default_temperature = 0.7
    _default_max_tokens = 1024
    _default_max_turns = 10

    def __init__(
        self,
        engine: InferenceEngine,
        model: str,
        *,
        tools: Optional[List[BaseTool]] = None,
        bus: Optional[EventBus] = None,
        max_turns: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        interactive: bool = False,
        confirm_callback=None,
        skill_few_shot_examples: Optional[List[str]] = None,
        capability_policy: Optional[Any] = None,
        agent_id: Optional[str] = None,
        rate_limiter: Optional[Any] = None,
    ) -> None:
        super().__init__(
            engine,
            model,
            tools=tools,
            bus=bus,
            max_turns=max_turns,
            temperature=temperature,
            max_tokens=max_tokens,
            interactive=interactive,
            confirm_callback=confirm_callback,
            skill_few_shot_examples=skill_few_shot_examples,
            capability_policy=capability_policy,
            agent_id=agent_id,
            rate_limiter=rate_limiter,
        )

    def _parse_response(self, text: str) -> dict:
        """Parse ReAct structured output."""
        stripped = text.strip()
        fenced = re.fullmatch(
            r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```",
            stripped,
            re.DOTALL | re.IGNORECASE,
        )
        if fenced and fenced.group(1).lstrip().startswith(("{", "[")):
            return _parse_json_response(fenced.group(1))
        if (
            stripped.startswith("{")
            or re.match(r"\[\s*\{", stripped)
            or stripped.lower().startswith("```json")
            or re.match(r"```[ \t]*\r?\n\s*[\{\[]", stripped)
        ):
            return _parse_json_response(stripped)

        result = {"thought": "", "action": "", "action_input": "", "final_answer": ""}

        # Extract Thought
        thought_match = re.search(
            r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        # Check for Final Answer
        final_match = re.search(
            r"Final Answer:\s*(.+)", text, re.DOTALL | re.IGNORECASE
        )
        if final_match:
            result["final_answer"] = final_match.group(1).strip()
            return result

        # Extract Action and Action Input
        action_match = re.search(r"Action:\s*(.+)", text, re.IGNORECASE)
        if action_match:
            result["action"] = action_match.group(1).strip()

        input_match = re.search(
            r"Action Input:\s*(.+?)(?=\n\n|\nThought:|\Z)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if input_match:
            result["action_input"] = input_match.group(1).strip()

        return result

    def run(
        self,
        input: str,
        context: Optional[AgentContext] = None,
        **kwargs: Any,
    ) -> AgentResult:
        self._emit_turn_start(input)

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        current_time_str = now.strftime("%A, %B %d, %Y, %I:%M %p UTC")
        current_date_str = now.strftime("%B %d, %Y")
        current_time_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        current_day = now.strftime("%A")

        # Build system prompt with rich tool descriptions
        tool_desc = build_tool_descriptions(self._tools)
        # Plan 2B I3: render optimized few-shot skill examples as a section
        # before the tool descriptions. Empty string when not present.
        if self._skill_few_shot_examples:
            skill_examples_block = (
                "## Skill Examples\n\n"
                + "\n\n".join(self._skill_few_shot_examples)
                + "\n\n"
            )
        else:
            skill_examples_block = ""
        # Respect $OPENJARVIS_HOME override for the base template (M2+ work).
        prompt_template = (
            load_system_prompt_override("native_react") or REACT_SYSTEM_PROMPT
        )
        format_kwargs = {
            "tool_descriptions": tool_desc,
            "skill_examples": skill_examples_block,
            "current_time_str": current_time_str,
            "current_date_str": current_date_str,
            "current_time_utc": current_time_utc,
            "current_day": current_day,
        }
        try:
            system_prompt = prompt_template.format(**format_kwargs)
        except KeyError:
            safe_kwargs = {
                k: v for k, v in format_kwargs.items() if f"{{{k}}}" in prompt_template
            }
            system_prompt = prompt_template.format(**safe_kwargs)
            if skill_examples_block and "{skill_examples}" not in prompt_template:
                system_prompt = system_prompt + "\n\n" + skill_examples_block
            if "{current_time_str}" not in prompt_template:
                system_prompt = (
                    f"Current Date and Time: {current_time_str}\n\n" + system_prompt
                )

        messages = self._build_messages(input, context, system_prompt=system_prompt)

        # Inject few-shot exemplars before the user input
        for ex in load_few_shot_exemplars("native_react"):
            if ex.get("input") and ex.get("output"):
                messages.insert(-1, Message(role=Role.USER, content=ex["input"]))
                messages.insert(-1, Message(role=Role.ASSISTANT, content=ex["output"]))

        # Forward tool specifications to inference engine if supported
        tools_openai = [
            t.to_openai_function()
            for t in self._tools
            if hasattr(t, "to_openai_function")
        ]
        extra_gen_kwargs: dict[str, Any] = {}
        if tools_openai:
            extra_gen_kwargs["tools"] = tools_openai

        all_tool_results: list[ToolResult] = []
        turns = 0
        total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for _turn in range(self._max_turns):
            turns += 1

            if self._loop_guard:
                messages = self._loop_guard.compress_context(messages)

            result = self._generate(messages, **extra_gen_kwargs)
            usage = result.get("usage", {})
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)

            content = result.get("content", "")
            parsed = self._parse_response(content)

            # Check if engine returned native tool_calls (or recovered failed_generation)
            if not parsed.get("action") and result.get("tool_calls"):
                raw_tc = result["tool_calls"][0]
                parsed["action"] = raw_tc.get("name", "")
                parsed["action_input"] = raw_tc.get("arguments", "{}")

            # Turn 1 Temporal Retrieval Guard: If query asks for latest/recent news,
            # never allow ungrounded memory hallucinations on turn 1. Force deep_view_search.
            if turns == 1 and not all_tool_results and _is_time_sensitive_news_query(input):
                has_dv = any(
                    getattr(t, "spec", None) and t.spec.name == "deep_view_search"
                    for t in self._tools
                )
                if has_dv and not parsed.get("action"):
                    logger.info("Enforcing Deep View retrieval guard for time-sensitive query: %r", input)
                    parsed["action"] = "deep_view_search"
                    q_lower = input.lower()
                    if any(kw in q_lower for kw in ("robot", "agent", "chip", "quantum", "voice", "search", "vision")):
                        parsed["action_input"] = json.dumps({"action": "search", "query": input, "limit": 10})
                    else:
                        parsed["action_input"] = json.dumps({"action": "latest_newsletters", "limit": 10})
                    parsed["final_answer"] = ""

            if parsed.get("parse_error"):
                self._emit_turn_end(turns=turns)
                return AgentResult(
                    content="Could not parse the model's ReAct response. "
                    + parsed["parse_error"],
                    tool_results=all_tool_results,
                    turns=turns,
                    metadata={
                        **total_usage,
                        "response_parse_error": parsed["parse_error"],
                        "messages": [_message_to_dict(m) for m in messages],
                    },
                )

            # Final answer?
            if parsed["final_answer"]:
                self._emit_turn_end(turns=turns)
                msg_dicts = [_message_to_dict(m) for m in messages]
                return AgentResult(
                    content=parsed["final_answer"],
                    tool_results=all_tool_results,
                    turns=turns,
                    metadata={**total_usage, "messages": msg_dicts},
                )

            # No action? Treat content as final answer
            if not parsed["action"]:
                cleaned_content = content
                thought_prefix = re.match(
                    r"^Thought:\s*(.+?)\n\n+(.+)$", content, re.DOTALL | re.IGNORECASE
                )
                if thought_prefix:
                    cleaned_content = thought_prefix.group(2).strip()
                self._emit_turn_end(turns=turns)
                msg_dicts = [_message_to_dict(m) for m in messages]
                return AgentResult(
                    content=cleaned_content,
                    tool_results=all_tool_results,
                    turns=turns,
                    metadata={**total_usage, "messages": msg_dicts},
                )

            # Execute action
            if not content or "Action:" not in content:
                content = (
                    f"Thought: I will search The Deep View for AI news.\n"
                    f"Action: {parsed['action']}\n"
                    f"Action Input: {parsed['action_input']}"
                )
            messages.append(Message(role=Role.ASSISTANT, content=content))

            tool_call = ToolCall(
                id=f"react_{turns}",
                name=parsed["action"],
                arguments=parsed["action_input"] or "{}",
            )

            # Loop guard check before execution
            if self._loop_guard:
                verdict = self._loop_guard.check_call(
                    tool_call.name,
                    tool_call.arguments,
                )
                if verdict.blocked:
                    tool_result = ToolResult(
                        tool_name=tool_call.name,
                        content=f"Loop guard: {verdict.reason}",
                        success=False,
                    )
                    all_tool_results.append(tool_result)
                    observation = f"Observation: {tool_result.content}"
                    messages.append(Message(role=Role.USER, content=observation))
                    continue

            tool_result = self._executor.execute(tool_call)
            all_tool_results.append(tool_result)

            observation = f"Observation: {tool_result.content}"
            if (
                tool_call.name in ("deep_view_search", "deep_view_research")
                and tool_result.success
            ):
                observation += (
                    "\n\n[Synthesize the substantive news stories, model/product announcements, "
                    "key takeaways, dates, and URLs from the above observation into your Final Answer. "
                    "Do not output meta-commentary or placeholders.]"
                )
            messages.append(Message(role=Role.USER, content=observation))

        # Max turns exceeded
        msg_dicts = [_message_to_dict(m) for m in messages]
        return self._max_turns_result(
            all_tool_results,
            turns,
            metadata={**total_usage, "messages": msg_dicts},
        )


__all__ = ["NativeReActAgent", "REACT_SYSTEM_PROMPT"]
