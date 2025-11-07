"""
Gemini LLM service implementation.

This module provides an implementation of the LlmService interface backed by
Google's Gemini API (google-generativeai). It supports non-streaming responses
and streaming of text content. Tool/function calling is supported via Gemini's
function calling capabilities.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from vanna.core.llm import (
    LlmService,
    LlmRequest,
    LlmResponse,
    LlmStreamChunk,
)
from vanna.core.tool import ToolCall, ToolSchema

logger = logging.getLogger(__name__)

class GeminiLlmService(LlmService):
    """Google Gemini-backed LLM service.

    Args:
        model: Gemini model name (e.g., "gemini-1.5-pro", "gemini-pro").
            Defaults to "gemini-1.5-pro". Can also be set via GEMINI_MODEL env var.
        api_key: API key; falls back to env `GOOGLE_API_KEY` or `GEMINI_API_KEY`.
        temperature: Sampling temperature; defaults to 0.7.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        **extra_kwargs: Any,
    ) -> None:
        try:
            import google.generativeai as genai
            self.genai = genai
        except Exception as e:
            raise ImportError(
                "google-generativeai package is required. Install with: pip install 'vanna[gemini]'"
            ) from e

        # Get API key from parameter, environment variables, or .env file
        api_key = (
            api_key
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        
        if not api_key:
            raise ValueError(
                "API key is required. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable, "
                "or pass api_key parameter."
            )

        # Configure Gemini
        genai.configure(api_key=api_key)

        # Model selection - default to gemini-2.5-flash if available, otherwise try gemini-2.5-pro
        default_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.model = model or default_model
        self.temperature = temperature
        self.extra_kwargs = extra_kwargs
        self.api_key = api_key

        # Base model - we'll create models with system_instruction dynamically if needed
        # Remove 'models/' prefix if present
        self._base_model_name = self.model.replace("models/", "")

    async def send_request(self, request: LlmRequest) -> LlmResponse:
        """Send a non-streaming request to Gemini and return the response."""
        payload = self._build_payload(request)
        
        # Debug: Log what we're sending to Gemini
        logger.debug(f"[Gemini] Sending request with {len(payload.get('contents', []))} messages")
        for i, msg in enumerate(payload.get('contents', [])):
            role = msg.get('role', 'unknown')
            if role == 'function':
                func_resp = msg.get('parts', [{}])[0].get('function_response', {})
                logger.debug(f"[Gemini] Message {i}: role={role}, function_name={func_resp.get('name')}, response_keys={list(func_resp.get('response', {}).keys()) if isinstance(func_resp.get('response'), dict) else 'not_dict'}")
            elif role in ('user', 'model'):
                parts = msg.get('parts', [])
                text_parts = [p.get('text', '')[:100] for p in parts if 'text' in p]
                func_calls = [p.get('function_call', {}).get('name') for p in parts if 'function_call' in p]
                logger.debug(f"[Gemini] Message {i}: role={role}, text_preview={text_parts}, function_calls={func_calls}")
            else:
                logger.debug(f"[Gemini] Message {i}: role={role}, parts={msg.get('parts', [])}")

        try:
            # Gemini's generate_content is synchronous, but we're in async context
            # Unpack the payload correctly for Gemini API
            contents = payload.pop("contents", [])
            generation_config = payload.pop("generation_config", {})
            system_instruction = payload.pop("system_instruction", None)
            tools = payload.pop("tools", None)
            
            # Get or create model - if system_instruction is provided, try to create model with it
            # Otherwise use base model
            if system_instruction:
                try:
                    # Try to create model with system_instruction (newer Gemini API)
                    model = self.genai.GenerativeModel(
                        self._base_model_name,
                        system_instruction=system_instruction
                    )
                except TypeError:
                    # Fallback: Add system instruction to contents if model creation doesn't support it
                    if contents and contents[0].get("role") == "user":
                        first_text = contents[0]["parts"][0].get("text", "")
                        contents[0]["parts"][0]["text"] = f"{system_instruction}\n\n{first_text}"
                    else:
                        contents.insert(0, {
                            "role": "user",
                            "parts": [{"text": system_instruction}]
                        })
                    model = self.genai.GenerativeModel(self._base_model_name)
            else:
                model = self.genai.GenerativeModel(self._base_model_name)
            
            # Call with proper parameters
            call_kwargs = {
                "contents": contents,
                "generation_config": generation_config,
            }
            if tools:
                call_kwargs["tools"] = tools
            
            resp = model.generate_content(**call_kwargs)
        except Exception as e:
            raise RuntimeError(f"Gemini request failed: {str(e)}") from e

        # Extract content and tool calls
        content = self._extract_content(resp)
        tool_calls = self._extract_tool_calls(resp)
        finish_reason = self._get_finish_reason(resp)
        
        # Debug: Log what Gemini returned
        logger.debug(f"[Gemini] Response: content_length={len(content) if content else 0}, tool_calls={len(tool_calls) if tool_calls else 0}, finish_reason={finish_reason}")
        if tool_calls:
            logger.debug(f"[Gemini] Tool calls: {[tc.name for tc in tool_calls]}")
        if content:
            logger.debug(f"[Gemini] Content preview: {content[:200]}")
        
        # If there are tool calls, ensure finish_reason indicates tool_calls
        if tool_calls:
            finish_reason = "tool_calls"

        # Extract usage if available
        usage: Optional[Dict[str, int]] = None
        if hasattr(resp, "usage_metadata"):
            try:
                usage = {
                    "prompt_tokens": getattr(resp.usage_metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(resp.usage_metadata, "completion_token_count", 0),
                    "total_tokens": (
                        getattr(resp.usage_metadata, "prompt_token_count", 0) +
                        getattr(resp.usage_metadata, "completion_token_count", 0)
                    ),
                }
            except Exception:
                pass

        return LlmResponse(
            content=content,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def stream_request(
        self, request: LlmRequest
    ) -> AsyncGenerator[LlmStreamChunk, None]:
        """Stream a request to Gemini.

        Emits `LlmStreamChunk` for textual deltas as they arrive. Tool calls are
        accumulated and emitted at the end of the stream.
        """
        payload = self._build_payload(request)

        try:
            # Gemini streaming
            # Unpack the payload correctly for Gemini API
            contents = payload.pop("contents", [])
            generation_config = payload.pop("generation_config", {})
            system_instruction = payload.pop("system_instruction", None)
            tools = payload.pop("tools", None)
            
            # Get or create model - if system_instruction is provided, try to create model with it
            if system_instruction:
                try:
                    # Try to create model with system_instruction (newer Gemini API)
                    model = self.genai.GenerativeModel(
                        self._base_model_name,
                        system_instruction=system_instruction
                    )
                except TypeError:
                    # Fallback: Add system instruction to contents if model creation doesn't support it
                    if contents and contents[0].get("role") == "user":
                        first_text = contents[0]["parts"][0].get("text", "")
                        contents[0]["parts"][0]["text"] = f"{system_instruction}\n\n{first_text}"
                    else:
                        contents.insert(0, {
                            "role": "user",
                            "parts": [{"text": system_instruction}]
                        })
                    model = self.genai.GenerativeModel(self._base_model_name)
            else:
                model = self.genai.GenerativeModel(self._base_model_name)
            
            # Call with proper parameters for streaming
            call_kwargs = {
                "contents": contents,
                "generation_config": generation_config,
                "stream": True,
            }
            if tools:
                call_kwargs["tools"] = tools
            
            stream = model.generate_content(**call_kwargs)
        except Exception as e:
            raise RuntimeError(f"Gemini streaming request failed: {str(e)}") from e

        accumulated_tool_calls: List[ToolCall] = []
        accumulated_content = ""
        last_finish: Optional[str] = None

        for chunk in stream:
            # Yield text content as it arrives
            chunk_text = self._extract_content_from_chunk(chunk)
            if chunk_text:
                accumulated_content += chunk_text
                yield LlmStreamChunk(content=chunk_text)

            # Accumulate tool calls
            chunk_tool_calls = self._extract_tool_calls_from_chunk(chunk)
            if chunk_tool_calls:
                accumulated_tool_calls.extend(chunk_tool_calls)

            # Check if this is the final chunk
            if hasattr(chunk, "candidates") and chunk.candidates:
                finish_reason = self._get_finish_reason_from_chunk(chunk)
                if finish_reason:
                    last_finish = finish_reason

        # Emit final chunk with tool calls if any
        if accumulated_tool_calls:
            yield LlmStreamChunk(
                tool_calls=accumulated_tool_calls,
                finish_reason="tool_calls"  # Explicitly set tool_calls when there are tool calls
            )
        else:
            # Emit terminal chunk to signal completion
            yield LlmStreamChunk(finish_reason=last_finish or "stop")

    async def validate_tools(self, tools: List[ToolSchema]) -> List[str]:
        """Validate tool schemas. Returns a list of error messages."""
        errors: List[str] = []
        # Basic validation
        for t in tools:
            if not t.name:
                errors.append("Tool name is required")
            if not t.description:
                errors.append(f"Tool '{t.name}' should have a description")
        return errors

    # Internal helpers
    def _build_payload(self, request: LlmRequest) -> Dict[str, Any]:
        """Build the Gemini API payload from LlmRequest."""
        # Convert messages to Gemini format
        contents: List[Dict[str, Any]] = []
        
        # Handle system prompt
        system_instruction = request.system_prompt or ""

        # Convert Vanna messages to Gemini format
        # Gemini expects Content objects, but we can pass dicts which the SDK converts
        for m in request.messages:
            if m.role == "user":
                # User message - simple text content
                contents.append({
                    "role": "user",
                    "parts": [{"text": m.content or ""}]
                })
            elif m.role == "assistant":
                # Assistant message - can have text and/or function calls
                parts = []
                if m.content:
                    parts.append({"text": m.content})
                # Handle tool calls in assistant messages
                if m.tool_calls:
                    for tc in m.tool_calls:
                        parts.append({
                            "function_call": {
                                "name": tc.name,
                                "args": tc.arguments if isinstance(tc.arguments, dict) else json.loads(tc.arguments) if isinstance(tc.arguments, str) else {}
                            }
                        })
                if parts:  # Only add if there are parts
                    contents.append({
                        "role": "model",
                        "parts": parts
                    })
            elif m.role == "tool":
                # Gemini uses function_response for tool results
                # The response must be a dict/JSON object, not a string
                # Extract the function name from the tool_call_id (format: function_name-uuid)
                tool_call_id = m.tool_call_id or "unknown"
                # Extract function name from tool_call_id (remove the uuid part)
                function_name = tool_call_id.split("-")[0] if "-" in tool_call_id else tool_call_id
                
                logger.debug(f"[Gemini] Processing tool response: tool_call_id={tool_call_id}, extracted_function_name={function_name}, content_type={type(m.content).__name__}")
                
                response_data = m.content
                if isinstance(response_data, str):
                    # Try to parse as JSON
                    try:
                        response_data = json.loads(response_data)
                        logger.debug(f"[Gemini] Parsed tool response as JSON: keys={list(response_data.keys()) if isinstance(response_data, dict) else 'not_dict'}")
                    except (json.JSONDecodeError, ValueError):
                        # If not valid JSON, wrap in a dict
                        response_data = {"result": response_data}
                        logger.debug(f"[Gemini] Wrapped tool response as dict: {response_data}")
                elif not isinstance(response_data, dict):
                    # If it's not a dict, wrap it
                    response_data = {"result": response_data}
                    logger.debug(f"[Gemini] Wrapped non-dict tool response: {response_data}")
                
                function_response = {
                    "name": function_name,
                    "response": response_data
                }
                logger.debug(f"[Gemini] Function response structure: {json.dumps(function_response, default=str, indent=2)[:500]}")
                
                contents.append({
                    "role": "function",
                    "parts": [{
                        "function_response": function_response
                    }]
                })

        # Build tools/function declarations if provided
        tools_config: Optional[Dict[str, Any]] = None
        if request.tools:
            function_declarations = []
            for t in request.tools:
                # Clean parameters schema to remove fields Gemini doesn't support
                cleaned_params = self._clean_schema_for_gemini(t.parameters)
                function_declarations.append({
                    "name": t.name,
                    "description": t.description,
                    "parameters": cleaned_params
                })
            
            tools_config = {
                "function_declarations": function_declarations
            }

        payload: Dict[str, Any] = {
            "contents": contents,
        }

        if system_instruction:
            payload["system_instruction"] = system_instruction

        if tools_config:
            payload["tools"] = [tools_config]

        # Add generation config
        generation_config = {
            "temperature": request.temperature if request.temperature is not None else self.temperature,
        }
        if request.max_tokens is not None:
            generation_config["max_output_tokens"] = request.max_tokens
        
        payload["generation_config"] = generation_config

        return payload

    def _clean_schema_for_gemini(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Clean JSON Schema to remove fields that Gemini doesn't support.
        
        Gemini's Schema doesn't support fields like 'title', 'default', 'anyOf', etc.
        This recursively removes unsupported fields and resolves 'anyOf' to the first non-null type.
        """
        if not isinstance(schema, dict):
            return schema
        
        # Handle anyOf/oneOf - Gemini doesn't support these, so extract the first non-null type
        if "anyOf" in schema:
            any_of = schema.get("anyOf", [])
            # Find first non-null entry (usually Optional[T] is [{"type": "null"}, {"type": "..."}])
            for option in any_of:
                if isinstance(option, dict):
                    option_type = option.get("type")
                    if option_type != "null":
                        # Use this option, but clean it recursively
                        cleaned_option = self._clean_schema_for_gemini(option)
                        # Merge with other fields from parent (like description), but exclude unsupported fields
                        unsupported_fields = {"title", "default", "$schema", "additionalProperties", "anyOf", "oneOf"}
                        result = {k: v for k, v in schema.items() if k not in unsupported_fields}
                        result.update(cleaned_option)
                        # Clean again to ensure no unsupported fields made it through
                        return self._clean_schema_for_gemini(result)
            # If all are null or we can't find a good option, return first one cleaned
            if any_of:
                cleaned_option = self._clean_schema_for_gemini(any_of[0])
                unsupported_fields = {"title", "default", "$schema", "additionalProperties", "anyOf", "oneOf"}
                result = {k: v for k, v in schema.items() if k not in unsupported_fields}
                result.update(cleaned_option)
                return self._clean_schema_for_gemini(result)
        
        if "oneOf" in schema:
            one_of = schema.get("oneOf", [])
            # Similar handling for oneOf
            for option in one_of:
                if isinstance(option, dict):
                    option_type = option.get("type")
                    if option_type != "null":
                        cleaned_option = self._clean_schema_for_gemini(option)
                        unsupported_fields = {"title", "default", "$schema", "additionalProperties", "anyOf", "oneOf"}
                        result = {k: v for k, v in schema.items() if k not in unsupported_fields}
                        result.update(cleaned_option)
                        return self._clean_schema_for_gemini(result)
            if one_of:
                cleaned_option = self._clean_schema_for_gemini(one_of[0])
                unsupported_fields = {"title", "default", "$schema", "additionalProperties", "anyOf", "oneOf"}
                result = {k: v for k, v in schema.items() if k not in unsupported_fields}
                result.update(cleaned_option)
                return self._clean_schema_for_gemini(result)
        
        # Fields that Gemini Schema doesn't support (remove these)
        unsupported_fields = {"title", "default", "$schema", "additionalProperties", "anyOf", "oneOf"}
        
        cleaned = {}
        for key, value in schema.items():
            # Skip unsupported fields
            if key in unsupported_fields:
                continue
            
            # Recursively process nested structures
            if isinstance(value, dict):
                cleaned[key] = self._clean_schema_for_gemini(value)
            elif isinstance(value, list):
                cleaned[key] = [
                    self._clean_schema_for_gemini(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                cleaned[key] = value
        
        return cleaned

    def _extract_content(self, response: Any) -> Optional[str]:
        """Extract text content from Gemini response."""
        if not hasattr(response, "candidates") or not response.candidates:
            return None

        candidate = response.candidates[0]
        if not hasattr(candidate, "content") or not candidate.content:
            return None

        parts = candidate.content.parts
        text_parts = [part.text for part in parts if hasattr(part, "text") and part.text]
        return "".join(text_parts) if text_parts else None

    def _extract_content_from_chunk(self, chunk: Any) -> Optional[str]:
        """Extract text content from streaming chunk."""
        if not hasattr(chunk, "candidates") or not chunk.candidates:
            return None

        candidate = chunk.candidates[0]
        if not hasattr(candidate, "content") or not candidate.content:
            return None

        parts = candidate.content.parts
        text_parts = [part.text for part in parts if hasattr(part, "text") and part.text]
        return "".join(text_parts) if text_parts else None

    def _extract_tool_calls(self, response: Any) -> List[ToolCall]:
        """Extract tool calls from Gemini response."""
        tool_calls: List[ToolCall] = []

        if not hasattr(response, "candidates") or not response.candidates:
            return tool_calls

        candidate = response.candidates[0]
        if not hasattr(candidate, "content") or not candidate.content:
            return tool_calls

        parts = candidate.content.parts
        for part in parts:
            if hasattr(part, "function_call"):
                fc = part.function_call
                name = getattr(fc, "name", None)
                if not name:
                    continue

                # Parse arguments - Gemini returns MapComposite (protobuf) which needs conversion
                args = getattr(fc, "args", {})
                
                # Convert MapComposite or other protobuf objects to dict
                # MapComposite behaves like dict but needs explicit conversion
                if isinstance(args, dict):
                    # Already a dict, use as-is
                    pass
                elif hasattr(args, "to_dict"):
                    try:
                        args = args.to_dict()
                    except Exception:
                        # Fallback: try dict() constructor which works on MapComposite
                        try:
                            args = dict(args.items()) if hasattr(args, 'items') else dict(args)
                        except Exception:
                            args = {"_raw": str(args)}
                elif isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"_raw": args}
                else:
                    # Try to convert protobuf MapComposite or similar to dict
                    try:
                        # MapComposite supports .items() iteration
                        if hasattr(args, 'items'):
                            args = dict(args.items())
                        elif hasattr(args, '__iter__') and not isinstance(args, (str, bytes)):
                            args = dict(args)
                        else:
                            args = {"_raw": str(args)}
                    except Exception:
                        args = {"_raw": str(args)}

                # Generate unique ID for each tool call since Gemini doesn't provide one
                # Use format: function_name-uuid to ensure uniqueness while keeping name reference
                unique_id = f"{name}-{uuid.uuid4().hex[:8]}"

                tool_calls.append(
                    ToolCall(
                        id=unique_id,
                        name=name,
                        arguments=args,
                    )
                )

        return tool_calls

    def _extract_tool_calls_from_chunk(self, chunk: Any) -> List[ToolCall]:
        """Extract tool calls from streaming chunk."""
        tool_calls: List[ToolCall] = []

        if not hasattr(chunk, "candidates") or not chunk.candidates:
            return tool_calls

        candidate = chunk.candidates[0]
        if not hasattr(candidate, "content") or not candidate.content:
            return tool_calls

        parts = candidate.content.parts
        for part in parts:
            if hasattr(part, "function_call"):
                fc = part.function_call
                name = getattr(fc, "name", None)
                if not name:
                    continue

                # Parse arguments - Gemini returns MapComposite (protobuf) which needs conversion
                args = getattr(fc, "args", {})
                
                # Convert MapComposite or other protobuf objects to dict
                # MapComposite behaves like dict but needs explicit conversion
                if isinstance(args, dict):
                    # Already a dict, use as-is
                    pass
                elif hasattr(args, "to_dict"):
                    try:
                        args = args.to_dict()
                    except Exception:
                        # Fallback: try dict() constructor which works on MapComposite
                        try:
                            args = dict(args.items()) if hasattr(args, 'items') else dict(args)
                        except Exception:
                            args = {"_raw": str(args)}
                elif isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"_raw": args}
                else:
                    # Try to convert protobuf MapComposite or similar to dict
                    try:
                        # MapComposite supports .items() iteration
                        if hasattr(args, 'items'):
                            args = dict(args.items())
                        elif hasattr(args, '__iter__') and not isinstance(args, (str, bytes)):
                            args = dict(args)
                        else:
                            args = {"_raw": str(args)}
                    except Exception:
                        args = {"_raw": str(args)}

                # Generate unique ID for each tool call
                unique_id = f"{name}-{uuid.uuid4().hex[:8]}"

                tool_calls.append(
                    ToolCall(
                        id=unique_id,
                        name=name,
                        arguments=args,
                    )
                )

        return tool_calls

    def _get_finish_reason(self, response: Any) -> Optional[str]:
        """Extract finish reason from Gemini response."""
        if not hasattr(response, "candidates") or not response.candidates:
            return None

        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        
        # Gemini's finish_reason is an enum, convert to string or int
        if finish_reason is not None:
            # Try to get the enum value (int) or name (string)
            if hasattr(finish_reason, 'value'):
                finish_reason_value = finish_reason.value
            elif hasattr(finish_reason, 'name'):
                # Map enum name to standard format
                name_map = {
                    "STOP": "stop",
                    "MAX_TOKENS": "length",
                    "SAFETY": "safety",
                    "RECITATION": "recitation",
                    "OTHER": "other",
                }
                return name_map.get(finish_reason.name, "stop")
            else:
                finish_reason_value = int(finish_reason) if isinstance(finish_reason, (int, str)) else 0
            
            # Map numeric finish reasons to standard format
            finish_reason_map = {
                0: "stop",  # STOP
                1: "length",  # MAX_TOKENS
                2: "safety",  # SAFETY
                3: "recitation",  # RECITATION
                4: "other",  # OTHER
            }
            return finish_reason_map.get(finish_reason_value, "stop")
        
        return "stop"

    def _get_finish_reason_from_chunk(self, chunk: Any) -> Optional[str]:
        """Extract finish reason from streaming chunk."""
        if not hasattr(chunk, "candidates") or not chunk.candidates:
            return None

        candidate = chunk.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        
        if finish_reason is not None:
            # Handle enum or int/string
            if hasattr(finish_reason, 'value'):
                finish_reason_value = finish_reason.value
            elif hasattr(finish_reason, 'name'):
                name_map = {
                    "STOP": "stop",
                    "MAX_TOKENS": "length",
                    "SAFETY": "safety",
                    "RECITATION": "recitation",
                    "OTHER": "other",
                }
                return name_map.get(finish_reason.name, "stop")
            else:
                finish_reason_value = int(finish_reason) if isinstance(finish_reason, (int, str)) else 0
            
            finish_reason_map = {
                0: "stop",
                1: "length",
                2: "safety",
                3: "recitation",
                4: "other",
            }
            return finish_reason_map.get(finish_reason_value, "stop")
        
        return None

