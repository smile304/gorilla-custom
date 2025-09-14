import json
import re
from typing import Any, List, Dict

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override
import time


class LiquidFCHandler(OSSHandler):
    """
    Liquid function-calling handler aligned with your chat template:

    - Begin with: <|im_start|>system\n ... <|im_end|>\n  (NO BOS)
    - Tools injected as: "List of tools: <|tool_list_start|>[...JSON tools...]<|tool_list_end|>"
    - Replay messages as <|im_start|>{role}\n{content}<|im_end|>\n
      (Do NOT add <|tool_response_*|> tags here; the outer template wraps role='tool'.)
    - Assistant cue: <|im_start|>assistant\n

    The model is instructed to emit ONLY tool calls as:
      <|tool_call_start|>{"name": "<tool-name>", "arguments": {...}}<|tool_call_end|>
    """

    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.is_fc_model = False

    # ----------------------------
    # Prompt formatting (template-compatible)
    # ----------------------------
    @override
    def _format_prompt(self, messages: Any, function: Any) -> str:
        """
        "bos_token": "<|startoftext|>",
        "chat_template":
        {%- set system_prompt = "" -%}
        {%- set ns = namespace(system_prompt="") -%}
        {%- if messages[0]["role"] == "system" -%}
            {%- set ns.system_prompt = messages[0]["content"] -%}
            {%- set messages = messages[1:] -%}
        {%- endif -%}
        {%- if tools -%}
            {%- set ns.system_prompt = ns.system_prompt + ("\n" if ns.system_prompt else "") + "List of tools: <|tool_list_start|>[" -%}
            {%- for tool in tools -%}
                {%- if tool is not string -%}
                    {%- set tool = tool | tojson -%}
                {%- endif -%}
                {%- set ns.system_prompt = ns.system_prompt + tool -%}
                {%- if not loop.last -%}
                    {%- set ns.system_prompt = ns.system_prompt + ", " -%}
                {%- endif -%}
            {%- endfor -%}
            {%- set ns.system_prompt = ns.system_prompt + "]<|tool_list_end|>" -%}
        {%- endif -%}
        {%- if ns.system_prompt -%}
            {{- "<|im_start|>system\n" + ns.system_prompt + "<|im_end|>\n" -}}
        {%- endif -%}
        {%- for message in messages -%}
            {{- "<|im_start|>" + message["role"] + "\n" -}}
            {%- set content = message["content"] -%}
            {%- if content is not string -%}
                {%- set content = content | tojson -%}
            {%- endif -%}
            {%- if message["role"] == "tool" -%}
                {%- set content = "<|tool_response_start|>" + content + "<|tool_response_end|>" -%}
            {%- endif -%}
            {{- content + "<|im_end|>\n" -}}
        {%- endfor -%}
        {%- if add_generation_prompt -%}
            {{- "<|im_start|>assistant\n" -}}
        {%- endif -%}
        """
        msgs = list(messages or [])
        tools = list(function or [])
        out_parts: List[str] = []

        # Pull optional system message content
        system_prompt = ""
        if msgs and msgs[0].get("role") == "system":
            system_prompt = msgs[0].get("content") or ""
            msgs = msgs[1:]

        # Build system body with tool list and guidance
        if tools:
            tool_items = [t if isinstance(t, str) else json.dumps(t) for t in tools]
            tool_list_block = (
                "<|tool_list_start|>\n["
                + ", ".join(tool_items)
                + "]\n<|tool_list_end|>"
            )
            guidance = " ".join([
                "If a tool is needed, respond with a tool call using the following format: ",
                "<|tool_call_start|>[tool_function_call_1, tool_function_call_2, ...]<|tool_call_end|>.",
                'Each tool function call should use Python-like syntax, e.g., speak("Hello"), random_number(min=1, max=10).',
                "If no tool is needed, you should answer the user directly without calling any tools.",
                "Always use the most relevant tool(s) for the user's request.",
                "If a tool returns an error, explain the error to the user.",
                "Be concise and helpful."
            ])
            
            if system_prompt:
                system_body = (system_prompt + "\n" + tool_list_block + "\n" + guidance)
            else:
                system_body = ("You are an AI assistant that has access to the following external functions: \n" + 
                             tool_list_block + "\n" + guidance)
        else:
            system_body = system_prompt

        if system_body:
            out_parts.append(f"<|im_start|>system\n{system_body}<|im_end|>\n")

        # Emit remaining chat history (no manual tool-response tags)
        for m in msgs:
            role = m.get("role", "")
            content = m.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content)
            out_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        # Assistant generation cue
        out_parts.append("<|im_start|>assistant\n")
        output = "".join(out_parts)
        # print(f"#######################%%%%%%%%%%%%%#########################\noutput: {output}", flush=True)
        # time.sleep(1000)
        return output

    # ----------------------------
    # Decoding helpers
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

        # Extract Liquid-specific format: <|tool_call_start|>[func(args)]<|tool_call_end|>
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
            
            # Parse arguments
            arguments = {}
            if args_str:
                arguments = LiquidFCHandler._parse_function_arguments(args_str)
            
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
                # Handle floats
                elif re.fullmatch(r"-?(?:\d+\.\d*|\.\d+|\d+\.)(?:[eE][+-]?\d+)?", val) or re.fullmatch(r"-?\d+(?:[eE][+-]?\d+)", val):
                    try:
                        parsed = float(val)
                    except Exception:
                        parsed = val
                else:
                    parsed = val  # last resort: raw string
            
            args[key] = parsed
        
        return args

    @staticmethod
    def _is_tool_call_response_format(items: list) -> bool:
        if not isinstance(items, list) or not items:
            return False
        for it in items:
            if not isinstance(it, dict):
                return False
            if set(it.keys()) != {"name", "arguments"}:
                return False
        return True

    # ----------------------------
    # AST / Execute decoders
    # ----------------------------
    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if not self._is_tool_call_response_format(tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [{c["name"]: dict(c.get("arguments", {}))} for c in tool_calls]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if not self._is_tool_call_response_format(tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return convert_to_function_call([{c["name"]: c.get("arguments", {})} for c in tool_calls])

    # ----------------------------
    # Pre/post hooks
    # ----------------------------
    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        """
        Respect the chat template: pass through messages and tools.
        """
        return {
            "message": test_entry.get("message", []),
            "function": test_entry.get("function", []),
        }

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        """
        Extract raw text; if tool calls are present, provide them in chat history and decoded for execution pairing.
        """
        try:
            text = api_response.choices[0].text
        except Exception:
            text = getattr(getattr(api_response.choices[0], "message", {}), "content", "") or ""
        
        # print(f"#######################%%%%%%%%%%%%%#########################\ntext: {api_response}", flush=True)
        # time.sleep(1000)
        extracted = self._extract_tool_calls(text)

        if self._is_tool_call_response_format(extracted):
            hist_msg = {"role": "assistant", "content": "", "tool_calls": extracted}
        else:
            hist_msg = {"role": "assistant", "content": text}

        usage = getattr(api_response, "usage", None) or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()

        return {
            "model_responses": text,  # keep raw; decode_* uses the extractor
            "model_responses_message_for_chat_history": hist_msg,
            "model_responses_decoded": extracted,  # for name alignment below
            "input_token": getattr(usage, "prompt_tokens", 0),
            "output_token": getattr(usage, "completion_tokens", 0),
        }

    @override
    def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
        inference_data["message"].append(model_response_data["model_responses_message_for_chat_history"])
        return inference_data

    @override
    def _add_execution_results_prompting(
        self, inference_data: dict, execution_results: List[str], model_response_data: dict
    ) -> dict:
        """
        Append a tool message WITHOUT adding any template tags; your outer template will wrap role='tool'.
        Uses model_responses_decoded to align execution results with tool names.
        """
        payload = []
        decoded = model_response_data.get("model_responses_decoded") or []
        for exec_res, d in zip(execution_results, decoded or execution_results):
            name = d.get("name") if isinstance(d, dict) else str(d)
            payload.append({"name": name, "execution_result": exec_res})

        inference_data["message"].append({"role": "tool", "content": json.dumps(payload, indent=2)})
        return inference_data
