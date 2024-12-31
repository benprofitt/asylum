from config import *
from openai import OpenAI
import openai, tiktoken
import json
from pydantic import BaseModel


class TokenUsageTracker:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TokenUsageTracker, cls).__new__(cls, *args, **kwargs)
            cls._instance.tokens_in = 0  # Initialize the token in usage attribute
            cls._instance.tokens_out = 0  # Initialize the token out usage attribute
        return cls._instance

    def add_tokens(self, tokens_in: int, tokens_out: int):
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out

    def get_usage(self):
        return self.tokens_in, self.tokens_out


class GPTResponse(BaseModel):
    answer: str
    reasoning: str

class ResponseString(BaseModel):
    response: str
    sufficient_data_present: bool
    additional_reasoning: str

class ResponseStringList(BaseModel):
    response: list[str]
    sufficient_data_present: bool
    additional_reasoning: str

class ResponseBoolean(BaseModel):
    response: bool
    sufficient_data_present: bool
    additional_reasoning: str

class ResponseBooleanList(BaseModel):
    response: list[bool]
    sufficient_data_present: bool
    additional_reasoning: str

class Item(BaseModel):
    name: str
    value: str

    def to_dict(self):
        return {
            self.name: self.value,
        }

class ResponseJSON(BaseModel):
    response: Item
    sufficient_data_present: bool
    additional_reasoning: str

class ResponseJSONList(BaseModel):
    response: list[Item]
    sufficient_data_present: bool
    additional_reasoning: str

    def to_dict(self):
        d = {}
        for item in self.response:
            d[item.name] = item.value
        return {
            "response": d,
            "sufficient_data_present": self.sufficient_data_present,
            "additional_reasoning": self.additional_reasoning,
        }
    
    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)
    

def count_tokens(message: str):
    model = "gpt-4"
    encoding = tiktoken.encoding_for_model(model)
    token_count = len(encoding.encode(message))

    return int(token_count)


def call_gpt_formatted(
    messages, format, model="gpt-4o-2024-08-06", temp=0.1, verbose=False, max_tokens=4069
):
    client = OpenAI()

    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=temp,
        max_tokens=max_tokens,
        response_format=format,
    )

    message = response.choices[0].message
    tokens_in = response.usage.prompt_tokens
    tokens_out = response.usage.completion_tokens

    TokenUsageTracker().add_tokens(tokens_in, tokens_out)

    if verbose and False:
        print("User: ")
        print(messages[-1]["content"][:200])
        print("GPT: ")
        print(message)
        print()

    if message.parsed:
        return message.parsed, tokens_in, tokens_out, None
    else:
        return "", tokens_in, tokens_out, "Refusal"


def call_gpt(
    messages, model="gpt-4o", temp=0.1, tools=[], verbose=False, max_tokens=4069
):

    client = OpenAI()

    if len(tools) == 0:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temp, max_tokens=max_tokens
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temp,
            parallel_tool_calls=False,
            tools=tools,
            tool_choice="required",
        )

    tokens_in = response.usage.prompt_tokens
    tokens_out = response.usage.completion_tokens
    response = response.choices[0].message

    content = response.content

    if len(tools) == 0:
        if verbose:
            print(content)
        return content, tokens_in, tokens_out, None

    tool_calls = response.tool_calls

    if verbose:
        print(content)

    return content, tokens_in, tokens_out, tool_calls


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(
            obj, openai.types.chat.chat_completion_message_tool_call.Function
        ):
            return obj.to_dict()
        return super().default(obj)

    def custom_json_serializer(obj):
        return json.dumps(obj, cls=CustomJSONEncoder)


def make_message(
    role, text=None, tool_calls=[], images=[], tool_call_id=None, verbose=False
):

    message = {"role": role}

    if tool_call_id is not None:
        message["tool_call_id"] = tool_call_id

    if tool_calls:
        message["tool_calls"] = tool_calls

    if len(images) > 0:
        content = [{"type": "text", "text": text}]
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image,
                    },
                }
            )
        message["content"] = content

    elif text is not None:
        message["content"] = text

    if verbose:
        print(json.dumps(message, indent=2, cls=CustomJSONEncoder), flush=True)

    return message


def make_tool(tool_name, description, type, params: list[list], required=[]):

    properties = {}
    for p in params:
        if p[1] == "array":
            properties[p[0]] = {
                "type": "array",
                "items": {"type": p[2], "description": p[3]},
                "description": p[3],
            }
        else:
            properties[p[0]] = {"type": p[1], "description": p[2]}

    tool = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": {
                "type": type,
                "properties": properties,
                "required": required,
            },
        },
    }

    return tool

if __name__ == "__main__":
    # test the json list type

    messages = [
        make_message("system", "Your job is to make five examples in a list of what a datasource might look like that would contain an answer to a given query."),
        make_message("user", "query: What is the capital of France?")
    ]

    response, _, _, refusal = call_gpt_formatted(messages, ResponseJSONList, temp=0.4, verbose=True)

    if refusal is not None:
        print("Refusal: ", refusal)

    print("Response: ", response)
