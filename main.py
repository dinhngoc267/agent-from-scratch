import inspect
import json
from time import time
import traceback
import types

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, ConfigDict, create_model

from dotenv import load_dotenv
load_dotenv()


class Goal(BaseModel):
    name: str
    description: str



class Tool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    tags: Optional[List[str]] = None
    terminal: bool = False


from enum import Enum
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class Memory(BaseModel):
    """M in GAME: the agent's conversational record — the running list of
    Messages the LLM sees each turn. A thin, swappable store so strategies
    (windowing, summarization, persistence) can vary later without touching
    Agent.run; for now it is just an append-and-read list."""

    messages: List["Message"] = []

    def add(self, message: "Message") -> None:
        self.messages.append(message)

    def get(self) -> List["Message"]:
        return self.messages
    

JSON_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def get_json_type(python_type: Any) -> str:
    origin = get_origin(python_type)

    if python_type in (list, List) or origin in (list, List):
        return "array"

    if python_type in (dict, Dict) or origin in (dict, Dict):
        return "object"

    if origin in (Union, types.UnionType):
        non_none_types = [arg for arg in get_args(python_type) if arg is not type(None)]
        return get_json_type(non_none_types[0]) if non_none_types else "string"

    return JSON_TYPE_MAP.get(python_type, "string")


def build_tool(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters_override: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Tool:
    name = name or func.__name__
    description = description or (func.__doc__.strip() if func.__doc__ else "")

    if parameters_override is None:
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)

        args_schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        # The injected-context param (by name OR ActionContext-subclass annotation)
        # is filled by the Environment from the hidden ActionContext.
        ctx = context_param(func)
        ctx_param_name = ctx[0] if ctx else None

        for param_name, param in signature.parameters.items():
            if param_name == ctx_param_name:
                continue

            param_type = type_hints.get(param_name, str)
            param_schema = {
                "type": get_json_type(param_type)
            }

            args_schema["properties"][param_name] = param_schema
            if param.default == inspect.Parameter.empty:
                args_schema["required"].append(param_name)
    else:
        args_schema = parameters_override

    return Tool(
        name=name,
        description=description,
        parameters=args_schema,
        function=func,
        tags=tags or []
    )


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters_override: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
):
    """Marks a function as an agent tool without registering it.

    Note: `terminal` is an agent-level decision, not a tool property — set it at
    registry.register(func, terminal=True), not here.
    """

    def decorator(func: Callable):
        func._tool_metadata = build_tool(
            func=func,
            name=name,
            description=description,
            parameters_override=parameters_override,
            tags=tags
        )
        return func

    return decorator


class ActionRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def register(self, action: Tool | Callable, terminal: bool = False):
        if isinstance(action, Tool):
            tool = action
        else:
            tool = getattr(action, "_tool_metadata", None)
            if tool is None:
                raise TypeError(
                    f"register() expects a Tool or a @tool-decorated function, got "
                    f"{type(action).__name__}. Decorate it with @tool (or pass "
                    f"build_tool(func)) before registering."
                )

        if terminal != tool.terminal:
            tool = tool.model_copy(update={"terminal": terminal})

        self.register_tool(tool)

    def get_tool(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Return tool schemas in the OpenAI/litellm function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools.values()
        ]



def generate_response(memory: "Memory", tools: List[str]=None) -> Any:
    from litellm import completion

    dict_messages = [message.model_dump(exclude_none=True) for message in memory.get()]

    response = completion(
        model="openai/gpt-4o",
        messages=dict_messages,
        max_tokens=1024,
        tools=tools,
        tool_choice="required",
        parallel_tool_calls=False
    )
    
    return response 

def has_named_parameter(func: Callable, name: str) -> bool:
    """True if `func`'s signature declares a parameter called `name`."""
    return name in inspect.signature(func).parameters


def context_param(func: Callable) -> Optional[tuple[str, type]]:
    """Find the injected-context parameter of a tool and the ActionContext type it
    requires, or None if the tool declares no context.
    """
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}
    for name in inspect.signature(func).parameters:
        ann = hints.get(name)
        if isinstance(ann, type) and issubclass(ann, ActionContext):
            return name, ann
    return None


class ActionContext(BaseModel):
    """The hidden context the Environment injects into tools that declare it.

    Holds everything we want HIDDEN from the LLM — connection strings, clients,
    secrets, the agent itself.

    We can declare subclass `ActionContext` as well. 
        class UserContext(ActionContext):
            db: dict                            
            def get_user(self, uid: int) -> dict:
                return self.db[uid]

    `arbitrary_types_allowed` lets fields hold live objects (db clients, sockets,
    the agent) that aren't JSON-serializable.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Environment:
    """The execution site for tools.

    """

    def execute_action(
        self,
        action: Tool,
        args: dict,
        action_context: Optional[ActionContext] = None,
    ) -> Any:
        """Run `action` with the LLM-provided `args`, injecting the hidden context
        into whichever param the tool declares for it (by name or subclass type)."""
        try:
            call_args = dict(args)
            ctx = context_param(action.function)
            if ctx:
                ctx_param_name, _ = ctx
                call_args[ctx_param_name] = action_context or ActionContext()
            return action.function(**call_args)
        except Exception as e:
            return {"error": str(e)}



OutputSchema = Union[type[str], type[BaseModel]]


class Agent:
    TERMINATE_TOOL_NAME = "terminate"
    _TERMINATE_STR_FIELD = "result"

    def __init__(
        self,
        goal: str,
        action_registry: ActionRegistry,
        output_schema: Optional[OutputSchema] = None,
        environment: Optional["Environment"] = None,
        action_context: Optional[ActionContext] = None,
    ):
        self.goal = goal
        self.action_registry = action_registry
        self.output_schema = output_schema
        self.result: Any = None

        self.environment = environment or Environment()
        self.action_context = action_context or ActionContext()


        self._validate_context()

        # Tools the registry marked terminal (excluding the framework's terminate tool).
        terminal_tools = [
            t.name for t in action_registry.tools.values()
            if t.terminal and t.name != self.TERMINATE_TOOL_NAME
        ]

        # No exit declared at all -> default to a text answer (output_schema=str).
        if output_schema is None and not terminal_tools:
            output_schema = self.output_schema = str
        
        # Because in my design, Agent can stop the loop after a specific tool call. (to save tokens, LLm do not need to generate terminal tool with the same parameters again)
        # This choice make it a little bit complicated when output schema defined in Agent different from the tool's output schema. 
        if output_schema is not None:
            if terminal_tools:
                raise ValueError(
                    f"output_schema={output_schema.__name__} conflicts with terminal "
                    f"tool(s) {terminal_tools}: their output does not match the schema. "
                    f"Drop output_schema or unmark those tools as terminal."
                )
            action_registry.register_tool(self._build_terminate_tool(output_schema))
        elif len(terminal_tools) > 1:
            raise ValueError(
                f"Multiple terminal tools {terminal_tools}; an agent must have exactly "
                f"one exit so result has a well-defined type."
            )

        self.memory = Memory()
        self.memory.add(Message(role=Role.SYSTEM, content=self.goal))

    def _validate_context(self) -> None:
        """Verify that the agent's action_context satisfies every tool's declared"""
        for tool in self.action_registry.tools.values():
            ctx = context_param(tool.function)
            if ctx is None:
                continue  # tool declares no context — nothing to satisfy
            _, required = ctx
            if not isinstance(self.action_context, required):
                raise TypeError(
                    f"Tool {tool.name!r} requires action_context: {required.__name__}, "
                    f"but this agent was given {type(self.action_context).__name__}. "
                    f"Pass a context that is-a {required.__name__} (combine schemas via "
                    f"subclassing if multiple tools need different types), or split the "
                    f"tools across agents."
                )

    @classmethod
    def _build_terminate_tool(cls, output_schema: OutputSchema) -> Tool:
        """Build the terminate tool that ends the agent loop and carries the final result.

        str is normalized to a one-field model {result: str} and unwrapped on return,
        so a single code path handles both: the LLM-facing schema always comes from
        pydantic's model_json_schema(), and result is the string (for str) or
        the schema instance (for a BaseModel).
        """
        unwrap_field = None
        if output_schema is str:
            output_schema = create_model("StrResult", **{cls._TERMINATE_STR_FIELD: (str, ...)})
            unwrap_field = cls._TERMINATE_STR_FIELD

        def terminate(**kwargs: Any) -> Any:
            """Provide the final result and end the task."""
            obj = output_schema(**kwargs)
            return getattr(obj, unwrap_field) if unwrap_field else obj

        terminate_tool = build_tool(
            func=terminate,
            name=cls.TERMINATE_TOOL_NAME,
            description=f"Provide the final result ({output_schema.__name__}) and end the task.",
            parameters_override=output_schema.model_json_schema(),
        )
        terminate_tool.terminal = True
        return terminate_tool

    def add_memory(self, message: Message):
        self.memory.add(message)

    @staticmethod
    def _to_content(result: Any) -> str:
        """Render a tool's return value as text for the LLM. """
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        if isinstance(result, (dict, list)):
            return json.dumps(result, ensure_ascii=False)
        return str(result)

    def run(self, user_input: str, max_iterations: int = 15):
        self.add_memory(Message(role=Role.USER, content=user_input))

        for i in range(max_iterations):
            response = generate_response(self.memory, tools=self.action_registry.get_tools_schema())
            assistant_msg = response.choices[0].message

            # Record the assistant turn (may carry content and/or tool_calls).
            self.add_memory(
                Message(
                    role=Role.ASSISTANT,
                    content=assistant_msg.content,
                    tool_calls=[tc.model_dump() for tc in assistant_msg.tool_calls] 
                                    if assistant_msg.tool_calls else None,
                )
            )

            if assistant_msg.tool_calls:
                for tool_call in assistant_msg.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments

                    tool = self.action_registry.get_tool(tool_name)

                    if not tool:
                        raise ValueError(f"Tool '{tool_name}' not found.")

                    # Execute the tool in the Environment (which injects the hidden
                    # action_context and catches exceptions), then record its result
                    # as text for the LLM.
                    result = self.environment.execute_action(
                        tool, json.loads(tool_args), self.action_context
                    )
                    self.add_memory(
                        Message(
                            role=Role.TOOL,
                            tool_call_id=tool_call.id,
                            name=tool_name,
                            content=self._to_content(result)
                        )
                    )

                    # A terminal tool ends the loop — but only on success. If it
                    # raised, the Environment returned {"error": ...}; feed that back
                    # so the LLM can correct and re-call rather than ending on an error.
                    errored = isinstance(result, dict) and "error" in result
                    if tool.terminal and not errored:
                        self.result = result
                        return self.result

        return self.result


@tool()
def add(a: float, b: float) -> float:
    """Adds two floats."""
    return a + b

@tool()
def multiply(a: float, b: float) -> float:
    """Multiplies two floats."""
    return a * b



class UserContext(ActionContext):
    """Context for user-store tools. Holds the hidden DB the LLM must never see."""
    db: dict
    def get_user(self, user_id: int) -> dict:
        return self.db[user_id]


class AuditContext(ActionContext):
    """Context for audit tools — a DIFFERENT capability set."""
    audit_sink: list
    def log(self, event: str) -> str:
        self.audit_sink.append(event)
        return f"logged: {event}"


@tool()
def lookup_user(user_id: int, action_context: UserContext) -> dict:
    """Look up a user by id."""
    # LLM only ever sees lookup_user(user_id: int); the store stays hidden.
    return action_context.get_user(user_id)


@tool()
def audit(event: str, action_context: AuditContext) -> str:
    """Record an audit event."""
    return action_context.log(event)


def chat_loop():

    registry = ActionRegistry()
    registry.register(add)
    registry.register(multiply)

    agent = Agent(
        goal=(
            "You are a helpful conversational assistant. Chat naturally with the "
            "user. When a calculation is needed, use the add and multiply tools. "
            "When you have an answer for the user's latest message, call terminate "
            "with your reply."
        ),
        action_registry=registry,
    )

    print("Chat with the agent (tools: add, multiply). Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("bye!")
            break

        reply = agent.run(user_input)
        print(json.dumps([m.model_dump(exclude_none=True) for m in agent.memory.get()], indent=4))
        print(f"agent> {reply}\n")


if __name__ == "__main__":
    import sys

    if "--demo" not in sys.argv:
        chat_loop()
        sys.exit(0)

    print("Starting Agent...")

    # output_schema=str: the LLM synthesizes a text answer via terminate(result: str).
    registry = ActionRegistry()
    registry.register(add)
    registry.register(multiply)

    agent = Agent(goal="Perform arithmetic operations.", action_registry=registry)
    
    result = agent.run("Please add 200 and 10 and then multiply the result by 2.")
    print("\n[str] result:", result)
    print(json.dumps([m.model_dump(exclude_none=True) for m in agent.memory.get()], indent=4))

    # output_schema=BaseModel  a structured result; the model fills each field.
    class FinalAnswer(BaseModel):
        answer: str
        computation: str

    registry_b = ActionRegistry()
    registry_b.register(add)
    registry_b.register(multiply)

    agent_b = Agent(
        goal="Perform arithmetic operations and summarize the result for the user.",
        action_registry=registry_b,
        output_schema=FinalAnswer,
    )
    result_b = agent_b.run("Please add 200 and 10, then explain the result.")

    print("\n[BaseModel] result:", result_b)
    print(json.dumps([m.model_dump(exclude_none=True) for m in agent_b.memory.get()], indent=4))

    fake_db = {7: {"id": 7, "name": "Ada", "email": "ada@example.com"}}

    registry_c = ActionRegistry()
    registry_c.register(lookup_user)

    agent_c = Agent(
        goal="Look up users and report their name.",
        action_registry=registry_c,
        action_context=UserContext(db=fake_db),
    )

    lookup_schema = next(
        t for t in registry_c.get_tools_schema() if t["function"]["name"] == "lookup_user"
    )
    assert "action_context" not in lookup_schema["function"]["parameters"]["properties"], \
        "action_context leaked into the LLM-facing schema!"
    print("\n[context] lookup_user schema (no action_context):",
          json.dumps(lookup_schema["function"]["parameters"], indent=2))

    result_c = agent_c.run("Who is user 7?")
    print("\n[context] result:", result_c)


    try:
        Agent(
            goal="...",
            action_registry=registry_c,
            action_context=AuditContext(audit_sink=[]),  # not a UserContext
        )
        print("\n[validate] ERROR: mismatch was not caught!")
    except TypeError as e:
        print("\n[validate] mismatch caught at init:", e)

    # Two tools needing DIFFERENT context types: only a context that is-a BOTH works.
    class AppContext(UserContext, AuditContext):
        """Combines both capability sets — is-a UserContext AND is-a AuditContext."""

    registry_d = ActionRegistry()
    registry_d.register(lookup_user)
    registry_d.register(audit)

    sink: list = []
    agent_d = Agent(
        goal="Look up users and audit lookups.",
        action_registry=registry_d,
        action_context=AppContext(db=fake_db, audit_sink=sink),  # satisfies both
    )
    print("\n[validate] combined AppContext satisfies both tools: OK")


