from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import (
    combine_consecutive_user_prompts,
    system_prompt_pre_processing_chat_model,
)
from overrides import override

import time
from typing import Any

def parse_liquid_response(response: str | None) -> str:
    """
    Parse the response from LiquidAI and return the function call.
    """
    if response is None:
        return "No Response"
    if not isinstance(response, str):
        try:
            response = str(response)
        except Exception as e:
            return ""
    import re
    match = re.search(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", response, re.DOTALL)
    answer = ""
    if match:
        answer = match.group(1)
    else:
        answer = response
    answer = re.sub(r"<\|tool_call_start\|>|<\|tool_call_end\|>", "", answer).strip()   
    return answer

class LiquidHandler(OSSHandler):
    tool_response_template = """<|tool_response_start|>{content}<|tool_response_end|>"""
    tool_call_start_template = """<|tool_call_start|>{content}<|tool_call_end|>"""
    tool_list_template = """<|tool_list_start|>\n{tools}\n<|tool_list_end|>"""
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)

    @override
    def _format_prompt(self, messages, function) -> str:
        """
        "bos_token": "<|startoftext|>",
        "chat_template": "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%}\n    {%- if messages[0]['content'] is string -%}\n        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}\n    {%- else -%}\n        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}\n    {%- endif -%}\n    {%- set loop_messages = messages[1:] -%}\n{%- else -%}\n    {%- set first_user_prefix = \"\" -%}\n    {%- set loop_messages = messages -%}\n{%- endif -%}\n{%- for message in loop_messages -%}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}\n        {{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}\n    {%- endif -%}\n    {%- if (message['role'] == 'assistant') -%}\n        {%- set role = \"model\" -%}\n    {%- else -%}\n        {%- set role = message['role'] -%}\n    {%- endif -%}\n    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else \"\") }}\n    {%- if message['content'] is string -%}\n        {{ message['content'] | trim }}\n    {%- elif message['content'] is iterable -%}\n        {%- for item in message['content'] -%}\n            {%- if item['type'] == 'image' -%}\n                {{ '<start_of_image>' }}\n            {%- elif item['type'] == 'text' -%}\n                {{ item['text'] | trim }}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- else -%}\n        {{ raise_exception(\"Invalid content type\") }}\n    {%- endif -%}\n    {{ '<end_of_turn>\n' }}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{'<start_of_turn>model\n'}}\n{%- endif -%}\n",
        """
        return ""

    @override
    def _query_prompting(self, inference_data: dict) -> Any:
        # We use the OpenAI Completions API
        import json
        import re

        function: list[dict] = inference_data["function"]
        message: list[dict] = inference_data["message"]
        for i in range(len(function)):
            if "name" in function[i]:
                function[i] = {"type": "function", "function": function[i]}
        json_string = json.dumps(function, indent=2)
        formatted_prompt: str = self._format_prompt(message, function)
        inference_data["inference_input_log"] = {"formatted_prompt": formatted_prompt}

        # prompt formatting
        # setup the client and system prompt
        model_name_lfm = "lfm2_1b"
        port = 3000
        gpu_number = "001"
        self.client.base_url = f"http://liquid-gpu-{gpu_number}:{port}/v1"
        start_time = time.time()

        api_response = self.client.chat.completions.create(
            model=model_name_lfm,
            messages = message,
            temperature=0.3,
            tools = function
        )
        end_time = time.time()


        content_tobe_parsed = api_response.choices[0].message.content
        if content_tobe_parsed is None:
            cutted_message = []
            if len(message) > 1:
                cutted_message.append(message[0])
                cutted_message.append(message[-1])
            else:
                cutted_message.append(message[0])
            start_time = time.time()
            api_response = self.client.chat.completions.create(
                model=model_name_lfm,
                messages = cutted_message,
                temperature=0.3,
                tools = function
            )
            end_time = time.time()
            content_tobe_parsed = api_response.choices[0].message.content
        api_response = parse_liquid_response(content_tobe_parsed)
        #response_content = re.sub(r"<\|tool_call_start\|>\s*(.*?)\s*<\|tool_call_end\|>", r"\1", api_response, flags=re.DOTALL)

        return api_response, end_time - start_time


    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        return {
            "model_responses": api_response,
            "input_token": 50,
            "output_token": 50
        }


    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                test_entry["question"][round_idx]
            )

        data = {"message": [], "function": functions}
        if "initial_config" in test_entry:
            data["initial_config"] = test_entry["initial_config"]

        return data

    @override
    def _add_execution_results_prompting(self, inference_data: dict, execution_results: list[str], model_response_data: dict) -> dict:
        import json
        response_message = []
        for execution_result, decoded_model_response in zip(
            execution_results, model_response_data["model_responses_decoded"]
        ):
            forming_message = {'name': decoded_model_response, 'execution_result': execution_result}
            response_message.append(forming_message)
        
        response_message_str = json.dumps(response_message, indent=2)
        inference_data["message"].append(
            {
                "role": "tool",
                "content": self.tool_response_template.format(content=response_message_str),
            }
        )
        return inference_data
