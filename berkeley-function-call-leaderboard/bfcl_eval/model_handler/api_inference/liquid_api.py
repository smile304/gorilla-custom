import json
import re
from typing import Any, List, Dict
import os

from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.model_handler.utils import convert_to_function_call
from openai import OpenAI
from overrides import override


class LiquidFCAPIHandler(OpenAICompletionsHandler):
    """
    API handler for Liquid function-calling models using OpenAI-compatible endpoints.

    This handler formats prompts according to Liquid's chat template:
    - System messages with tool injection via <|tool_list_start|> and <|tool_list_end|>
    - Tool calls expected in format: <|tool_call_start|>[func_name(args)]<|tool_call_end|>
    """

    def __init__(self, model_name, temperature, min_p=0.0) -> None:
        super().__init__(model_name, temperature, min_p)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS

        # Configure OpenAI client - adjust base_url and api_key as needed
        self.client = OpenAI(
            base_url=os.getenv("LIQUID_API_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("LIQUID_API_KEY", "none"),
        )

        # Enable FC path (mirrors Qwen)
        self.is_fc_model = True

    # ----------------------------
    # Prompt formatting (template-compatible)
    # ----------------------------
    def _format_liquid_prompt(self, messages: List[Dict], tools: List[Dict] = None) -> List[Dict]:
        """
        Format messages according to Liquid's chat template requirements.
        Returns properly formatted messages for the API call.
        """
        if not messages:
            return []

        formatted_messages = []
        msgs = list(messages)

        # Extract system message if present
        system_prompt = ""
        if msgs and msgs[0].get("role") == "system":
            system_prompt = msgs[0].get("content", "")
            msgs = msgs[1:]

        # Build system message with tools if provided
        if tools:
            tool_items = [json.dumps(tool) for tool in tools]
            tool_list_block = (
                "List of tools: <|tool_list_start|>["
                + ", ".join(tool_items)
                + "]<|tool_list_end|>"
            )
            guidance = " ".join([
                "If a tool is needed, respond with a tool call using the following format:",
                "<|tool_call_start|>[tool_function_call_1, tool_function_call_2, ...]<|tool_call_end|>.",
                'Each tool function call should use Python-like syntax, e.g., speak("Hello"), random_number(min=1, max=10).',
                "If no tool is needed, you should answer the user directly without calling any tools.",
                "Always use the most relevant tool(s) for the user's request.",
                "If a tool returns an error, explain the error to the user.",
                "Be concise and helpful."
            ])

            if system_prompt:
                system_content = system_prompt + "\n" + tool_list_block + "\n" + guidance
            else:
                system_content = (
                    "You are an AI assistant that has access to the following external functions:\n" +
                    tool_list_block + "\n" + guidance
                )
        else:
            system_content = system_prompt or "You are a helpful AI assistant."

        # Add system message
        formatted_messages.append({"role": "system", "content": system_content})

        # Add remaining messages; only wrap tool content here (on the way INTO the model)
        for msg in msgs:
            role = msg.get("role", "")
            if role == "tool":
                tool_msg: Dict[str, Any] = {"role": "tool"}

                content = msg.get("content", "")
                # Wrap content for Liquid template; raw remains in history elsewhere
                if isinstance(content, str):
                    wrapped = f"<|tool_response_start|>{content}<|tool_response_end|>"
                else:
                    wrapped = f"<|tool_response_start|>{json.dumps(content)}<|tool_response_end|>"
                tool_msg["content"] = wrapped

                # Preserve OpenAI-compatible linkage fields for the server
                if "tool_call_id" in msg:
                    tool_msg["tool_call_id"] = msg["tool_call_id"]
                if "name" in msg:
                    tool_msg["name"] = msg["name"]

                formatted_messages.append(tool_msg)
            else:
                # passthrough for user/assistant/system
                formatted_messages.append({k: v for k, v in msg.items()})

        return formatted_messages

    # ----------------------------
    # Tool call extraction
    # ----------------------------
    @staticmethod
    def _extract_tool_calls(text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from Liquid format:
          - <|tool_call_start|>[func_name(arg1="val", arg2=123)]<|tool_call_end|>
        Returns a list of {"name": str, "arguments": dict}.
        """
        if not text:
            return []

        calls: List[Dict[str, Any]] = []

        # Extract Liquid-specific blocks
        liquid_blocks = re.findall(r"<\|tool_call_start\|>\s*(.*?)\s*<\|tool_call_end\|>", text, re.DOTALL)

        for block in liquid_blocks:
            block = block.strip()

            # Remove outer brackets if present: [func(args)] -> func(args)
            if block.startswith("[") and block.endswith("]"):
                block = block[1:-1].strip()

            # Parse function call: func_name(arg1="val", arg2=123)
            func_match = re.match(r"([A-Za-z_][\w]*)\s*\((.*?)\)\s*$", block, re.DOTALL)
            if not func_match:
                continue

            func_name = func_match.group(1)
            args_str = func_match.group(2).strip()

            # Parse arguments into Python objects
            arguments = {}
            if args_str:
                arguments = LiquidFCAPIHandler._parse_function_arguments(args_str)

            calls.append({"name": func_name, "arguments": arguments})

        # Deduplicate while preserving order
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for c in calls:
            sig = (c.get("name"), json.dumps(c.get("arguments", {}), sort_keys=True, ensure_ascii=False))
            if sig not in seen:
                seen.add(sig)
                deduped.append(c)

        return deduped

    @staticmethod
    def _parse_function_arguments(args_str: str) -> Dict[str, Any]:
        """
        Parse function arguments from string format: arg1="val", arg2=123, arg3=["a", "b"]
        """
        args: Dict[str, Any] = {}

        def split_top_level_commas(s: str) -> List[str]:
            parts, buf, depth, quote = [], [], 0, None
            for ch in s:
                if quote:
                    buf.append(ch)
                    if ch == quote and (not buf or buf[-2] != "\\"):
                        quote = None
                    continue
                if ch in ("'", '"'):
                    quote = ch
                    buf.append(ch)
                    continue
                if ch in "[{(":
                    depth += 1
                    buf.append(ch)
                    continue
                if ch in "]})":
                    depth -= 1
                    buf.append(ch)
                    continue
                if ch == "," and depth == 0:
                    parts.append("".join(buf).strip())
                    buf = []
                else:
                    buf.append(ch)
            if buf:
                parts.append("".join(buf).strip())
            return parts

        for part in split_top_level_commas(args_str):
            if not part or "=" not in part:
                continue
            k, v = part.split("=", 1)
            key = k.strip()
            val = v.strip()

            parsed = None

            # Try JSON for {} or []
            if (val.startswith("{") and val.endswith("}")) or (val.startswith("[") and val.endswith("]")):
                try:
                    parsed = json.loads(val)
                except Exception:
                    parsed = None

            if parsed is None:
                # Handle quoted strings
                if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
                    parsed = val[1:-1]
                # Handle boolean and null values
                elif val.lower() in ("true", "false", "null", "none"):
                    parsed = {"true": True, "false": False, "null": None, "none": None}[val.lower()]
                # Handle integers
                elif re.fullmatch(r"-?\d+", val):
                    parsed = int(val)
                # Handle floats, incl. scientific notation
                elif re.fullmatch(r"-?(?:\d+\.\d*|\.\d+|\d+\.)(?:[eE][+-]?\d+)?", val) or re.fullmatch(r"-?\d+(?:[eE][+-]?\d+)", val):
                    try:
                        parsed = float(val)
                    except Exception:
                        parsed = val
                else:
                    parsed = val  # raw string fallback

            args[key] = parsed

        return args

    @staticmethod
    def _is_tool_call_response_format(items: list) -> bool:
        """Check if the response is in the expected tool call format."""
        if not isinstance(items, list) or not items:
            return False
        for it in items:
            if not isinstance(it, dict):
                return False
            if set(it.keys()) != {"name", "arguments"}:
                return False
        return True

    # ---------- NEW: pythonic call string builder ----------
    @staticmethod
    def _to_python_call_str(name: str, arguments: Dict[str, Any]) -> str:
        """
        Build a pythonic function call string, e.g.:
          name(a=1, b='x', c=[1, 2], d={'k': True}, e=None)
        Uses repr() so booleans/None render as True/False/None.
        """
        if not arguments:
            return f"{name}()"
        parts = []
        for k, v in arguments.items():
            parts.append(f"{k}={repr(v)}")
        return f"{name}({', '.join(parts)})"

    # ----------------------------
    # Function Calling (FC) methods
    # ----------------------------
    @override
    def _query_FC(self, inference_data: dict):
        """Query the model in function calling mode."""
        messages = inference_data["message"]
        tools = inference_data.get("tools", [])

        # Format messages according to Liquid template
        formatted_messages = self._format_liquid_prompt(messages, tools)

        inference_data["inference_input_log"] = {
            "message": repr(formatted_messages),
            "tools": tools
        }

        return self.generate_with_backoff(
            messages=formatted_messages,
            model=self.model_name,
            temperature=self.temperature,
            min_p=self.min_p,
            # tools are embedded in system message; do not pass `tools=` here
        )

    # --- in LiquidFCAPIHandler._parse_query_response_FC ---

    @override
    def _parse_query_response_FC(self, api_response: Any) -> dict:
        """Parse the response from function calling mode."""
        try:
            content = api_response.choices[0].message.content or ""
        except Exception:
            content = ""

        # Extract tool calls from the response
        extracted_tool_calls = self._extract_tool_calls(content)

        if extracted_tool_calls and self._is_tool_call_response_format(extracted_tool_calls):
            # Build OpenAI-compatible tool_calls for the chat history,
            # but DO NOT transform the user-visible result.
            tool_calls_for_history = []
            tool_call_ids = []
            for i, call in enumerate(extracted_tool_calls):
                tool_id = f"call_{i}"
                tool_call_ids.append(tool_id)
                args_json_str = json.dumps(call["arguments"], ensure_ascii=False)
                tool_calls_for_history.append({
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": args_json_str
                    }
                })

            # Preserve raw model text (with Liquid tags or [func(...)]), so the caller's "result"
            # is exactly what the model produced.
            model_responses = content

            model_response_message_for_chat_history = {
                "role": "assistant",
                "content": None,  # keep None for FC flow compatibility
                "tool_calls": tool_calls_for_history,
            }
        else:
            # No valid tool calls; treat as normal text
            model_responses = content
            model_response_message_for_chat_history = {
                "role": "assistant",
                "content": content,
            }
            tool_call_ids = []

        # Token usage best-effort
        usage = getattr(api_response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)
        else:
            input_tokens = output_tokens = 0

        return {
            "model_responses": model_responses,  # <-- now raw model text, not a list of dicts
            "model_responses_message_for_chat_history": model_response_message_for_chat_history,
            "tool_call_ids": tool_call_ids if extracted_tool_calls else [],
            "input_token": input_tokens,
            "output_token": output_tokens,
            # Optional: surface decoded calls if your caller wants them
            "model_responses_decoded": extracted_tool_calls or [],
        }


    # ----------------------------
    # Prompting methods
    # ----------------------------
    @override
    def _query_prompting(self, inference_data: dict):
        """Query the model in prompting mode."""
        messages = inference_data["message"]
        function_list = inference_data.get("function", [])

        # Format messages according to Liquid template
        formatted_messages = self._format_liquid_prompt(messages, function_list)

        inference_data["inference_input_log"] = {"message": repr(formatted_messages)}

        return self.generate_with_backoff(
            messages=formatted_messages,
            model=self.model_name,
            temperature=self.temperature,
            min_p=self.min_p,
        )

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """Parse the response from prompting mode."""
        try:
            content = api_response.choices[0].message.content or ""
        except Exception:
            content = ""

        # Extract tool calls from the response (rare in prompting mode for Liquid)
        extracted_tool_calls = self._extract_tool_calls(content)

        if extracted_tool_calls and self._is_tool_call_response_format(extracted_tool_calls):
            hist_msg = {
                "role": "assistant",
                "content": content,
                "tool_calls": extracted_tool_calls
            }
        else:
            hist_msg = {"role": "assistant", "content": content}

        # Token usage
        usage = getattr(api_response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)
        else:
            input_tokens = output_tokens = 0

        return {
            "model_responses": content,
            "model_responses_message_for_chat_history": hist_msg,
            "model_responses_decoded": extracted_tool_calls,
            "input_token": input_tokens,
            "output_token": output_tokens,
        }

    @override
    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        """Add assistant message to the conversation history in prompting mode."""
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: List[str], model_response_data: dict
    ) -> dict:
        """
        Add execution results to the conversation history in prompting mode.
        Keep raw content; Liquid wrapping is applied in _format_liquid_prompt when sending.
        """
        payload = []
        decoded = model_response_data.get("model_responses_decoded", [])

        for exec_res, d in zip(execution_results, decoded or execution_results):
            name = d.get("name") if isinstance(d, dict) else str(d)
            payload.append({"name": name, "execution_result": exec_res})

        tool_content = json.dumps(payload, indent=2)
        # No tool_call_id linkage in prompting path
        inference_data["message"].append({"role": "tool", "content": tool_content})
        return inference_data

    # ----------------------------
    # FC: Append execution results with tool_call_id linkage (Qwen-compatible)
    # ----------------------------
    def _add_execution_results_FC(
        self, inference_data: dict, execution_results: List[str], model_response_data: dict
    ) -> dict:
        """
        Append OpenAI-compatible tool messages so the framework can continue the FC loop.
        Keep RAW content here for compatibility; Liquid wrapping happens in _format_liquid_prompt.
        """
        assistant_msg = model_response_data.get("model_responses_message_for_chat_history", {})
        tool_calls = assistant_msg.get("tool_calls", [])
        if not tool_calls:
            return inference_data

        tool_messages = []
        for exec_res, tool_call in zip(execution_results, tool_calls):
            tool_id = tool_call.get("id")
            tool_name = tool_call.get("function", {}).get("name")

            raw_content = exec_res if isinstance(exec_res, str) else json.dumps(exec_res)

            tool_messages.append({
                "role": "tool",
                "tool_call_id": tool_id,   # critical for chaining
                "name": tool_name,
                "content": raw_content,     # keep raw (Qwen-compat)
            })

        inference_data["message"].extend(tool_messages)
        return inference_data

    # ----------------------------
    # Decoder methods for compatibility
    # ----------------------------
    def decode_ast(self, result, language="python", has_tool_call_tag=True):
        """Decode AST from model response."""
        tool_calls = self._extract_tool_calls(result)
        if not self._is_tool_call_response_format(tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [{c["name"]: dict(c.get("arguments", {}))} for c in tool_calls]

    def decode_execute(self, result, has_tool_call_tag=True):
        """Decode for execution from model response."""
        tool_calls = self._extract_tool_calls(result)
        if not self._is_tool_call_response_format(tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        # convert_to_function_call expects dict values, so pass parsed args here
        return convert_to_function_call([{c["name"]: c.get("arguments", {})} for c in tool_calls])
