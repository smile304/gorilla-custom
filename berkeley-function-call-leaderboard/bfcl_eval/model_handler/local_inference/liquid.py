import json
import re
from typing import Any, List, Dict

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override


class LiquidFCHandler(OSSHandler):
    bos_token = "<|startoftext|>"

    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.is_fc_model = True  # function-calling style

    # ----------------------------
    # Prompt formatting (matches your Liquid template)
    # ----------------------------
    @override
    def _format_prompt(self, messages: List[Dict[str, Any]], function: List[Dict[str, Any]]) -> str:
        msgs = list(messages or [])
        tools = list(function or [])
        out = [self.bos_token]

        # Pull system if present
        system_prompt = ""
        if msgs and msgs[0].get("role") == "system":
            system_prompt = msgs[0].get("content", "") or ""
            msgs = msgs[1:]

        # Append tool list + explicit Liquid tool-call instruction
        if tools:
            # Tool list block (exactly as template expects)
            tool_items = [t if isinstance(t, str) else json.dumps(t) for t in tools]
            tool_list_block = (
                "List of tools: <|tool_list_start|>["
                + ", ".join(tool_items)
                + "]<|tool_list_end|>"
            )
            guidance = (
                "\nYou may call one or more tools to assist with the user query.\n"
                "For each tool call, respond with **only** one block in this exact format:\n"
                "<|tool_call_start|>{\"name\": \"<tool-name>\", \"arguments\": { ... }}<|tool_call_end|>\n"
                "Do not include any extra commentary before or after the tool-call block.\n"
                "After tool responses arrive (they will be wrapped as <|tool_response_start|>…<|tool_response_end|>), "
                "either make more tool calls or provide the final answer without tool-call tags."
            )
            system_prompt = (system_prompt + ("\n" if system_prompt else "") + tool_list_block + guidance)

        if system_prompt:
            out.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")

        # Emit chat history (NO manual tool-response tags here; template will wrap)
        for m in msgs:
            role = m.get("role", "")
            content = m.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content)
            out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        out.append("<|im_start|>assistant\n")
        return "".join(out)

    # ----------------------------
    # Decoding helpers
    # ----------------------------
    @staticmethod
    def _extract_tool_calls(text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        blocks = []
        # Liquid markers (primary)
        blocks += re.findall(r"<\|tool_call_start\|>\s*(.*?)\s*<\|tool_call_end\|>", text, re.DOTALL)
        # Fallback: XML style
        blocks += re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)

        calls: List[Dict[str, Any]] = []
        for blob in blocks:
            parsed = None
            for attempt in (blob, None):
                try:
                    parsed = json.loads(blob if attempt is None else json.loads(blob))
                    break
                except Exception:
                    pass
            if isinstance(parsed, dict):
                if "name" in parsed and "arguments" in parsed:
                    calls.append(parsed)
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "name" in item and "arguments" in item:
                        calls.append(item)
        return calls

    # ----------------------------
    # AST / Execute decoders
    # ----------------------------
    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if not isinstance(tool_calls, list) or any(not isinstance(x, dict) for x in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [{c["name"]: dict(c.get("arguments", {}))} for c in tool_calls]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if not isinstance(tool_calls, list) or any(not isinstance(x, dict) for x in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return convert_to_function_call([{c["name"]: c.get("arguments", {})} for c in tool_calls])

    # ----------------------------
    # Pre/post hooks
    # ----------------------------
    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        # Liquid template handles system + tool list; don't mutate user messages here.
        return {"message": [], "function": test_entry.get("function", [])}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        # Prefer text completions; fall back to message.content
        text = ""
        try:
            text = api_response.choices[0].text
        except Exception:
            text = getattr(getattr(api_response.choices[0], "message", {}), "content", "") or ""

        tool_calls = self._extract_tool_calls(text)

        # Optional <think>…</think> support
        reasoning, cleaned = "", text
        if "</think>" in text:
            pre, post = text.split("</think>", 1)
            reasoning = pre.rsplit("<think>", 1)[-1].strip("\n")
            cleaned = post.lstrip("\n")

        if tool_calls:
            hist_msg = {"role": "assistant", "content": "", "tool_calls": tool_calls}
        else:
            hist_msg = {"role": "assistant", "content": cleaned}
        hist_msg["reasoning_content"] = reasoning

        usage = getattr(api_response, "usage", None) or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()
        return {
            "model_responses": cleaned,
            "reasoning_content": reasoning,
            "model_responses_message_for_chat_history": hist_msg,
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
        Append a tool message WITHOUT manual <|tool_response_*|> tags.
        The Liquid template will wrap role='tool' content automatically.
        """
        payload = []
        decoded = model_response_data.get("model_responses_decoded") or []
        for exec_res, d in zip(execution_results, decoded or execution_results):
            name = d.get("name") if isinstance(d, dict) else str(d)
            payload.append({"name": name, "execution_result": exec_res})

        inference_data["message"].append({"role": "tool", "content": json.dumps(payload, indent=2)})
        return inference_data
