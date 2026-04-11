"""Main chat engine with direct agent import and fallback."""

from __future__ import annotations

import sys
import threading
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

from .models import (
    ChatMessage,
    ChatSession,
    ComposerState,
    MessageRole,
    StreamingEvent,
    ToolCall,
)
from .streamer import ChatStreamer
from .fallback_tmux import TmuxChatFallback


class ChatNotAvailableError(Exception):
    """Raised when chat functionality is not available."""

    pass


class ChatEngine:
    """Main chat engine - tries direct import, falls back to TMUX."""

    _instance: Optional["ChatEngine"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ChatEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._sessions: dict[str, ChatSession] = {}
        self._streamers: dict[str, ChatStreamer] = {}
        self._fallbacks: dict[str, TmuxChatFallback] = {}
        self._initialized = True
        self._direct_import_available = self._check_direct_import()

    @staticmethod
    def _check_direct_import() -> bool:
        """Check if hermes-agent can be imported."""
        try:
            # Try multiple import paths
            try:
                from run_agent import AIAgent

                return True
            except ImportError:
                pass

            try:
                sys.path.insert(
                    0, str(Path.home() / ".local" / "share" / "hermes" / "src")
                )
                from run_agent import AIAgent

                return True
            except ImportError:
                pass

            # Check common installation locations
            for path in [
                "/usr/local/lib/hermes/src",
                "/opt/hermes/src",
                str(Path.home() / "hermes" / "src"),
                str(Path.home() / "projects" / "hermes" / "src"),
            ]:
                if Path(path).exists():
                    sys.path.insert(0, path)
                    try:
                        from run_agent import AIAgent

                        return True
                    except ImportError:
                        sys.path.pop(0)

            return False
        except Exception:
            return False

    def create_session(
        self, profile: Optional[str] = None, model: Optional[str] = None
    ) -> ChatSession:
        """Create a new chat session."""
        session_id = str(uuid.uuid4())[:8]

        # Determine backend type
        if self._direct_import_available:
            backend_type = "direct"
        elif TmuxChatFallback.is_available():
            backend_type = "tmux"
        else:
            raise ChatNotAvailableError(
                "Chat not available. Install hermes-agent or ensure tmux is running with Hermes."
            )

        session = ChatSession(
            id=session_id,
            profile=profile,
            model=model,
            title=f"Chat {session_id}",
            backend_type=backend_type,
        )

        self._sessions[session_id] = session

        # Initialize appropriate backend
        if backend_type == "tmux":
            fallback = TmuxChatFallback(session_id)
            if fallback.find_hermes_pane():
                self._fallbacks[session_id] = fallback
            else:
                # TMUX available but no Hermes pane found
                raise ChatNotAvailableError(
                    "TMUX available but no Hermes pane found. Start Hermes CLI in a tmux session first."
                )

        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[ChatSession]:
        """List all active sessions."""
        return list(self._sessions.values())

    def end_session(self, session_id: str) -> bool:
        """End a chat session."""
        if session_id in self._sessions:
            self._sessions[session_id].is_active = False

            # Cleanup streamer
            if session_id in self._streamers:
                self._streamers[session_id].stop()
                del self._streamers[session_id]

            # Cleanup fallback
            if session_id in self._fallbacks:
                del self._fallbacks[session_id]

            return True
        return False

    def send_message(
        self,
        session_id: str,
        content: str,
        on_token: Optional[Callable[[str], None]] = None,
        on_tool_start: Optional[Callable[[str, str, dict], None]] = None,
        on_tool_end: Optional[Callable[[str, Any, Optional[str]], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
    ) -> ChatStreamer:
        """Send a message and return streamer for responses."""
        session = self._sessions.get(session_id)
        if not session:
            raise ChatNotAvailableError(f"Session {session_id} not found")

        if not session.is_active:
            raise ChatNotAvailableError(f"Session {session_id} is inactive")

        streamer = ChatStreamer()
        self._streamers[session_id] = streamer

        # Update session stats
        session.message_count += 1
        session.last_activity = datetime.now()

        # Route to appropriate backend
        if session.backend_type == "direct" and self._direct_import_available:
            self._send_direct(
                session_id,
                content,
                streamer,
                on_token,
                on_tool_start,
                on_tool_end,
                on_reasoning,
            )
        elif session.backend_type == "tmux" and session_id in self._fallbacks:
            self._send_tmux(session_id, content, streamer)
        else:
            streamer.emit_error("No backend available for this session")

        return streamer

    def _send_direct(
        self,
        session_id: str,
        content: str,
        streamer: ChatStreamer,
        on_token: Optional[Callable[[str], None]] = None,
        on_tool_start: Optional[Callable[[str, str, dict], None]] = None,
        on_tool_end: Optional[Callable[[str, Any, Optional[str]], None]] = None,
        on_reasoning: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Send via direct agent import (threaded)."""

        def run_agent():
            try:
                from run_agent import AIAgent

                # Callbacks that emit to streamer
                def token_callback(token: str):
                    streamer.emit_token(token)
                    if on_token:
                        on_token(token)

                def tool_start_callback(tool_id: str, name: str, args: dict):
                    streamer.emit_tool_start(tool_id, name, args)
                    if on_tool_start:
                        on_tool_start(tool_id, name, args)

                def tool_end_callback(
                    tool_id: str, result: Any, error: Optional[str] = None
                ):
                    streamer.emit_tool_end(tool_id, result, error)
                    if on_tool_end:
                        on_tool_end(tool_id, result, error)

                def reasoning_callback(content: str):
                    streamer.emit_reasoning(content)
                    if on_reasoning:
                        on_reasoning(content)

                # Create and run agent
                session = self._sessions[session_id]
                agent = AIAgent(
                    session_id=session_id,
                    profile=session.profile,
                    model=session.model,
                    on_token=token_callback,
                    on_tool_start=tool_start_callback,
                    on_tool_end=tool_end_callback,
                    on_reasoning=reasoning_callback,
                )

                # Run the conversation turn
                agent.send_message(content)

                streamer.emit_done()

            except Exception as e:
                streamer.emit_error(f"Agent error: {str(e)}")

        # Spawn daemon thread
        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()

    def _send_tmux(
        self,
        session_id: str,
        content: str,
        streamer: ChatStreamer,
    ) -> None:
        """Send via TMUX fallback."""
        fallback = self._fallbacks.get(session_id)
        if not fallback:
            streamer.emit_error("TMUX fallback not available")
            return

        # TMUX can't truly stream, so emit info and wait for DB
        streamer.emit(
            StreamingEvent(
                type="info",
                data={
                    "message": "TMUX mode: Message sent to CLI. Response will appear when available."
                },
            )
        )

        if fallback.send_message(content):
            # In TMUX mode, we rely on file watcher to detect DB changes
            # The frontend will poll or use WebSocket for updates
            streamer.emit_done()
        else:
            streamer.emit_error("Failed to send message via TMUX")

    def get_composer_state(self, session_id: str) -> ComposerState:
        """Get current composer state for UI."""
        session = self._sessions.get(session_id)
        if not session:
            return ComposerState(model="unknown")

        return ComposerState(
            model=session.model or "claude-4-sonnet",  # Default fallback
            is_streaming=session_id in self._streamers
            and session_id not in self._fallbacks,
            current_tool=None,  # Would need to track current tool from streamer
        )


# Global engine instance
chat_engine = ChatEngine()
