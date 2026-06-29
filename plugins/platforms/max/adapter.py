"""MAX messenger (МАКС) platform adapter for Hermes Agent.

A Hermes platform plugin that connects to MAX Bot API (https://botapi.max.ru)
using httpx (already a Hermes dependency) — no external MAX SDK required.

Supports two update delivery modes:

- **Long polling** (default): GET /updates with a ``marker`` cursor.
- **Webhook**: Starts a local HTTP server that receives POST updates from
  MAX servers. Requires a publicly reachable URL.

Configuration in config.yaml::

    platforms:
      max:
        enabled: true
        extra:
          token: "your-bot-token"
          mode: "polling"            # or "webhook"
          webhook_url: "https://your-domain.com/webhook/max"
          webhook_port: 8765        # local webhook server port
          allowed_users: "user1,user2"
          allow_all_users: false
          home_channel: "chat_id"
          home_channel_name: "My Chat"
          markdown: true

Environment variables (env wins over config.yaml ``extra``):

    MAX_BOT_TOKEN             Bot token (required)
    MAX_MODE                  "polling" or "webhook"
    MAX_WEBHOOK_URL           Public webhook URL (webhook mode)
    MAX_WEBHOOK_PORT          Local port for webhook server
    MAX_ALLOWED_USERS         Comma-separated user IDs allowlist
    MAX_ALLOW_ALL_USERS       "true" to disable allowlist (dev only)
    MAX_HOME_CHANNEL          Default chat ID for cron / notification delivery
    MAX_HOME_CHANNEL_NAME     Human label for the home channel
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_API_BASE = "https://botapi.max.ru"
MAX_MESSAGE_LENGTH = 4000  # MAX per-message text limit
DEDUP_WINDOW_SECONDS = 600
DEDUP_MAX_SIZE = 2000
RECONNECT_BACKOFF = [2, 5, 10, 30, 60]
DEFAULT_POLL_TIMEOUT = 30
DEFAULT_POLL_LIMIT = 100

# Attachment type -> MessageType mapping
_ATTACHMENT_TYPE_MAP: Dict[str, MessageType] = {
    "image": MessageType.PHOTO,
    "video": MessageType.VIDEO,
    "audio": MessageType.AUDIO,
    "file": MessageType.DOCUMENT,
    "sticker": MessageType.STICKER,
}


def check_requirements() -> bool:
    """Check whether the MAX adapter is minimally configured."""
    if not HTTPX_AVAILABLE:
        return False
    token = os.getenv("MAX_BOT_TOKEN", "").strip()
    return bool(token)


def validate_config(config: PlatformConfig) -> bool:
    """Validate that the MAX platform has a token configured."""
    extra = getattr(config, "extra", {}) or {}
    token = extra.get("token") or os.getenv("MAX_BOT_TOKEN", "")
    return bool(token)


def is_connected(config: PlatformConfig) -> bool:
    """Check whether MAX is configured."""
    extra = getattr(config, "extra", {}) or {}
    token = os.getenv("MAX_BOT_TOKEN") or extra.get("token", "")
    return bool(token)


class MaxAdapter(BasePlatformAdapter):
    """MAX messenger platform adapter.

    Connects to the MAX Bot API and implements the Hermes
    BasePlatformAdapter interface for bidirectional message relay.
    """

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig) -> None:
        platform = Platform("max")
        super().__init__(config=config, platform=platform)

        extra = config.extra or {}

        # Token is the only hard requirement
        self._token: str = extra.get("token") or os.getenv("MAX_BOT_TOKEN", "")

        # Mode: "polling" (default) or "webhook"
        self._mode: str = (
            extra.get("mode") or os.getenv("MAX_MODE", "polling")
        ).strip().lower()

        # Webhook config
        self._webhook_url: str = (
            extra.get("webhook_url") or os.getenv("MAX_WEBHOOK_URL", "")
        )
        self._webhook_port: int = int(
            extra.get("webhook_port") or os.getenv("MAX_WEBHOOK_PORT", "8765")
        )

        # Markdown formatting for outbound messages
        self._markdown: bool = bool(extra.get("markdown"))

        # Internal state
        self._http_client: Optional[httpx.AsyncClient] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._webhook_runner: Optional[asyncio.Task] = None
        self._webhook_server: Optional[asyncio.AbstractServer] = None
        self._last_update_id: int = 0

        # Message deduplication: message_id -> timestamp
        self._seen_messages: Dict[str, float] = {}

        # Bot info (fetched on connect)
        self._bot_user_id: Optional[int] = None
        self._bot_name: str = "MAX Bot"

    # -- API helpers --------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        """Build Authorization headers for MAX API calls.
        
        Note: MAX Bot API accepts the token directly in Authorization header,
        no Bearer prefix required.
        """
        return {
            "Authorization": self._token,
            "Content-Type": "application/json",
        }

    async def _api_request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """Execute a MAX API request and return parsed JSON response.

        Returns None on failure after logging.
        """
        url = f"{MAX_API_BASE}{path}"
        try:
            resp = await self._http_client.request(
                method,
                url,
                headers=self._headers(),
                params=params,
                json=json_body,
                timeout=timeout,
            )
            if resp.status_code >= 400:
                logger.warning(
                    "[%s] MAX API %s %s -> %d: %s",
                    self.name, method, path, resp.status_code, resp.text[:300],
                )
                return None
            return resp.json()
        except httpx.TimeoutException:
            logger.warning("[%s] MAX API timeout: %s %s", self.name, method, path)
            return None
        except Exception as exc:
            logger.error("[%s] MAX API error: %s %s - %s", self.name, method, path, exc)
            return None

    async def _api_upload(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Upload a file to MAX and return the attachment token.

        Uses POST /uploads with multipart/form-data.
        """
        url = f"{MAX_API_BASE}/uploads"
        try:
            with open(file_path, "rb") as f:
                resp = await self._http_client.post(
                    url,
                    headers={"Authorization": self._token},
                    files={"file": (os.path.basename(file_path), f)},
                    timeout=60.0,
                )
            if resp.status_code >= 400:
                logger.warning(
                    "[%s] Upload failed %d: %s", self.name, resp.status_code, resp.text[:200]
                )
                return None
            return resp.json()
        except Exception as exc:
            logger.error("[%s] Upload error: %s", self.name, exc)
            return None

    # -- Connection lifecycle -----------------------------------------------

    async def connect(self) -> bool:
        """Connect to MAX by verifying the bot token and starting updates."""
        if not HTTPX_AVAILABLE:
            logger.warning("[%s] httpx not installed. Run: pip install httpx", self.name)
            return False
        if not self._token:
            logger.warning("[%s] MAX_BOT_TOKEN not configured", self.name)
            return False

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=60.0, write=15.0, pool=15.0)
        )

        # Verify bot token by fetching bot info
        me = await self._api_request("GET", "/me")
        if not me:
            logger.error("[%s] Failed to verify bot token - /me returned nothing", self.name)
            await self._http_client.aclose()
            self._http_client = None
            return False

        self._bot_user_id = me.get("user_id")
        self._bot_name = me.get("name", "MAX Bot")
        logger.info("[%s] Bot authenticated: %s (id=%s)", self.name, self._bot_name, self._bot_user_id)

        if self._mode == "webhook":
            success = await self._start_webhook()
        else:
            success = await self._start_polling()

        if success:
            self._mark_connected()
            logger.info(
                "[%s] Connected - %s mode, bot=%s",
                self.name, self._mode, self._bot_name,
            )
        return success

    async def _start_polling(self) -> bool:
        """Start the long-polling task."""
        self._poll_task = asyncio.create_task(self._run_polling_loop())
        return True

    async def _start_webhook(self) -> bool:
        """Start the local webhook server and subscribe with MAX."""
        if not self._webhook_url:
            logger.error("[%s] MAX_WEBHOOK_URL required for webhook mode", self.name)
            return False

        try:
            self._webhook_server = await asyncio.start_server(
                self._handle_webhook_connection,
                "0.0.0.0",
                self._webhook_port,
            )
        except OSError as exc:
            logger.error("[%s] Cannot start webhook server on port %d: %s", self.name, self._webhook_port, exc)
            return False

        # Register webhook with MAX
        subscribe_result = await self._api_request(
            "POST",
            "/subscriptions",
            json_body={"url": self._webhook_url},
        )
        if subscribe_result is None:
            logger.warning(
                "[%s] Failed to subscribe webhook at MAX - will still listen locally",
                self.name,
            )
        else:
            logger.info("[%s] Webhook subscribed: %s", self.name, self._webhook_url)

        self._webhook_runner = asyncio.create_task(
            self._serve_webhook_forever()
        )
        return True

    async def _serve_webhook_forever(self) -> None:
        """Keep the webhook server running until disconnect."""
        if self._webhook_server:
            async with self._webhook_server:
                await self._webhook_server.serve_forever()

    async def _handle_webhook_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle an incoming HTTP connection for webhook mode."""
        try:
            raw = await asyncio.wait_for(reader.read(65536), timeout=10.0)
            if not raw:
                writer.close()
                return

            # Parse the HTTP request manually (lightweight, no framework needed)
            text = raw.decode("utf-8", errors="replace")
            parts = text.split("\r\n\r\n", 1)
            header_section = parts[0]
            body = parts[1] if len(parts) > 1 else ""

            # Extract X-Max-Signature for verification
            signature = ""
            for line in header_section.split("\r\n"):
                if line.lower().startswith("x-max-signature:"):
                    signature = line.split(":", 1)[1].strip()

            # Verify HMAC-SHA256 signature if present
            if signature and self._token:
                expected = hmac.new(
                    self._token.encode(), body.encode(), hashlib.sha256
                ).hexdigest()
                if not hmac.compare_digest(signature, expected):
                    logger.warning("[%s] Webhook signature mismatch", self.name)
                    response = b"HTTP/1.1 403 Forbidden\r\n\r\n"
                    writer.write(response)
                    await writer.drain()
                    writer.close()
                    return

            # Parse the update
            if body.strip():
                try:
                    update = json.loads(body)
                    asyncio.create_task(self._process_update(update))
                except json.JSONDecodeError:
                    logger.debug("[%s] Webhook: invalid JSON body", self.name)

            # Respond 200 OK
            response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK"
            writer.write(response)
            await writer.drain()
            writer.close()
        except asyncio.TimeoutError:
            writer.close()
        except Exception as exc:
            logger.error("[%s] Webhook handler error: %s", self.name, exc)
            try:
                writer.close()
            except Exception:
                pass

    async def disconnect(self) -> None:
        """Disconnect from MAX."""
        self._running = False
        self._mark_disconnected()

        # Cancel polling task
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        # Unsubscribe webhook if applicable
        if self._mode == "webhook" and self._http_client:
            try:
                await self._api_request("DELETE", "/subscriptions")
            except Exception:
                pass

        # Stop webhook server
        if self._webhook_server:
            self._webhook_server.close()
            try:
                await self._webhook_server.wait_closed()
            except Exception:
                pass
            self._webhook_server = None

        if self._webhook_runner:
            self._webhook_runner.cancel()
            try:
                await self._webhook_runner
            except asyncio.CancelledError:
                pass
            self._webhook_runner = None

        # Close HTTP client
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._seen_messages.clear()
        logger.info("[%s] Disconnected", self.name)

    # -- Long-polling loop --------------------------------------------------

    async def _run_polling_loop(self) -> None:
        """Long-polling loop with automatic reconnection and backoff."""
        backoff_idx = 0
        poll_start: float = 0.0

        while self._running:
            try:
                poll_start = time.monotonic()
                await self._poll_once()
                backoff_idx = 0  # Reset on success
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if not self._running:
                    return
                logger.warning("[%s] Polling error: %s", self.name, exc)

                # Reset backoff if the poll stayed alive for a while
                if time.monotonic() - poll_start >= 60.0:
                    backoff_idx = 0
                delay = RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)]
                logger.info("[%s] Reconnecting in %ds...", self.name, delay)
                await asyncio.sleep(delay)
                backoff_idx += 1

    async def _poll_once(self) -> None:
        """Execute a single long-poll cycle.
        
        MAX returns: {"updates": [...], "marker": <int>}
        We save the marker and pass it on the next poll.
        """
        params: Dict[str, Any] = {
            "limit": DEFAULT_POLL_LIMIT,
        }
        if self._last_update_id:
            params["marker"] = self._last_update_id

        data = await self._api_request(
            "GET",
            "/updates",
            params=params,
            timeout=float(DEFAULT_POLL_TIMEOUT + 10),
        )

        if data is None:
            # Transient failure - short sleep before retry
            await asyncio.sleep(2)
            return

        updates = data if isinstance(data, list) else data.get("updates", [])
        marker = data.get("marker") if isinstance(data, dict) else None
        if marker:
            self._last_update_id = marker

        for update in updates:
            if not self._running:
                return
            await self._process_update(update)

    # -- Update processing --------------------------------------------------

    async def _process_update(self, update: Dict[str, Any]) -> None:
        """Route an incoming MAX update to the appropriate handler."""
        update_type = update.get("update_type", "")

        if update_type == "message_created":
            await self._on_message_created(update)
        elif update_type == "message_callback":
            await self._on_message_callback(update)
        elif update_type == "message_edited":
            # Optionally handle edited messages
            pass
        elif update_type in (
            "bot_added", "bot_removed", "user_added", "user_removed",
            "message_chat_created", "chat_title_changed",
            "message_deleted",
        ):
            logger.debug("[%s] Ignoring update type: %s", self.name, update_type)
        else:
            logger.debug("[%s] Unknown update type: %s", self.name, update_type)

    async def _on_message_created(self, update: Dict[str, Any]) -> None:
        """Process a message_created update."""
        message = update.get("message", {})
        body = message.get("body", {})
        mid = str(body.get("mid", "") or message.get("mid", "") or uuid.uuid4().hex)

        # Deduplication
        if self._is_duplicate(mid):
            logger.debug("[%s] Duplicate message %s, skipping", self.name, mid)
            return

        # Extract text
        text = (body.get("text") or "").strip()

        # Extract sender
        sender = message.get("sender", {})
        sender_id = str(sender.get("user_id", "") or message.get("sender_id", ""))
        sender_name = sender.get("name", "") or f"user_{sender_id}"

        # Chat identification: prefer chat_id, fall back to recipient
        chat_id = str(
            message.get("chat_id", "")
            or message.get("recipient_id", "")
            or sender_id
        )
        chat_type = "dm"

        # Determine if it's a group chat
        if message.get("chat_id") and str(message.get("chat_id")) != sender_id:
            chat_type = "group"

        # Handle attachments (images, files, voice, stickers, etc.)
        attachments = body.get("attachments", [])
        media_urls: List[str] = []
        media_types: List[str] = []
        message_type = MessageType.TEXT

        for att in attachments:
            att_type = att.get("type", "")
            payload = att.get("payload", {})

            if att_type in _ATTACHMENT_TYPE_MAP:
                message_type = _ATTACHMENT_TYPE_MAP[att_type]
                media_url = payload.get("url", "")
                if media_url:
                    media_urls.append(media_url)
                    media_types.append(att_type)
                # For stickers, also store the code in raw_message
                if att_type == "sticker":
                    sticker_code = payload.get("code", "")
                    if sticker_code:
                        if "raw_message" not in update:
                            update["raw_message"] = {}
                        update["raw_message"]["sticker_code"] = sticker_code

        # If no text but has attachments, still process
        if not text and not media_urls:
            logger.debug("[%s] Empty message body with no attachments, skipping", self.name)
            return

        # If text is present and has attachments, keep TEXT type
        # (attachments become media_urls on the event)
        if text:
            message_type = MessageType.TEXT

        # Build source
        source = self.build_source(
            chat_id=chat_id,
            chat_name=sender_name if chat_type == "dm" else None,
            chat_type=chat_type,
            user_id=sender_id,
            user_name=sender_name,
        )

        # Parse timestamp
        msg_timestamp = message.get("timestamp")
        try:
            timestamp = (
                datetime.fromtimestamp(int(msg_timestamp) / 1000, tz=timezone.utc)
                if msg_timestamp else datetime.now(tz=timezone.utc)
            )
        except (ValueError, OSError, TypeError):
            timestamp = datetime.now(tz=timezone.utc)

        message_event = MessageEvent(
            text=text,
            message_type=message_type,
            source=source,
            message_id=mid,
            raw_message=update,
            timestamp=timestamp,
            media_urls=media_urls,
            media_types=media_types,
        )

        logger.debug(
            "[%s] Message from %s in %s: %s",
            self.name, sender_name, chat_id, text[:80] if text else f"[{message_type.value}]",
        )
        await self.handle_message(message_event)

    async def _on_message_callback(self, update: Dict[str, Any]) -> None:
        """Process a message_callback (inline keyboard button press)."""
        callback_id = update.get("callback_id", "")
        payload = update.get("payload", {})
        callback_payload = payload.get("payload", "")

        message = update.get("message", {})
        body = message.get("body", {})
        mid = str(body.get("mid", "") or uuid.uuid4().hex)

        sender = message.get("sender", {})
        sender_id = str(sender.get("user_id", "") or message.get("sender_id", ""))
        sender_name = sender.get("name", "") or f"user_{sender_id}"

        chat_id = str(
            message.get("chat_id", "")
            or message.get("recipient_id", "")
            or sender_id
        )

        # Treat callback as a text message with the payload as text
        source = self.build_source(
            chat_id=chat_id,
            chat_type="dm",
            user_id=sender_id,
            user_name=sender_name,
        )

        message_event = MessageEvent(
            text=callback_payload or f"[callback:{callback_id}]",
            message_type=MessageType.COMMAND,
            source=source,
            message_id=mid,
            raw_message=update,
            timestamp=datetime.now(tz=timezone.utc),
        )

        logger.debug("[%s] Callback from %s: %s", self.name, sender_name, callback_payload[:80])
        await self.handle_message(message_event)

        # Auto-answer the callback query
        await self._answer_callback(callback_id)

    async def _answer_callback(self, callback_id: str) -> None:
        """Answer a callback query to dismiss the loading indicator."""
        await self._api_request(
            "POST",
            "/answers",
            json_body={"callback_id": callback_id},
        )

    # -- Deduplication ------------------------------------------------------

    def _is_duplicate(self, msg_id: str) -> bool:
        """Return True if this message ID was already seen within the dedup window."""
        now = time.time()
        if len(self._seen_messages) > DEDUP_MAX_SIZE:
            cutoff = now - DEDUP_WINDOW_SECONDS
            self._seen_messages = {
                k: v for k, v in self._seen_messages.items() if v > cutoff
            }
        if msg_id in self._seen_messages:
            return True
        self._seen_messages[msg_id] = now
        return False

    # -- Outbound messaging -------------------------------------------------

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message to a MAX chat.

        Args:
            chat_id: Target user_id or chat_id.
            content: Message text (may be markdown if configured).
            reply_to: Optional message ID to reply to.
            metadata: Additional platform-specific options.
                      Supported: sticker_code (str), attachments (list)

        Returns:
            SendResult with success status and message ID.
        """
        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")

        metadata = metadata or {}

        body: Dict[str, Any] = {}

        # Target: use chat_id if it looks like a chat, otherwise user_id
        if chat_id.startswith("-") or len(chat_id) > 12:
            body["chat_id"] = chat_id
        else:
            body["user_id"] = chat_id

        # Check for sticker in metadata
        sticker_code = metadata.get("sticker_code")
        attachments = metadata.get("attachments", [])

        if sticker_code:
            # Send sticker (must have empty text)
            body["text"] = ""
            body["attachments"] = [{
                "type": "sticker",
                "payload": {"code": sticker_code},
            }]
        elif attachments:
            body["text"] = content[:self.MAX_MESSAGE_LENGTH] if content else ""
            body["attachments"] = attachments
        else:
            # Text-only message
            if len(content) > self.MAX_MESSAGE_LENGTH:
                logger.warning(
                    "[%s] Message truncated from %d to %d chars",
                    self.name, len(content), self.MAX_MESSAGE_LENGTH,
                )
            body["text"] = content[:self.MAX_MESSAGE_LENGTH]

        # Format
        if self._markdown and not sticker_code:
            body["format"] = "markdown"

        # Reply to
        if reply_to:
            body["reply_to"] = reply_to

        # Notify (default true)
        body["notify"] = True

        result = await self._api_request("POST", "/messages", json_body=body)

        if result is not None:
            returned_mid = str(result.get("message_id", "") or result.get("mid", ""))
            return SendResult(success=True, message_id=returned_mid or uuid.uuid4().hex[:12])

        return SendResult(success=False, error="MAX API returned no result")

    async def send_sticker(
        self,
        chat_id: str,
        sticker_code: str,
    ) -> SendResult:
        """Send a sticker by its code.

        Args:
            chat_id: Target user_id or chat_id.
            sticker_code: Sticker hex code (e.g. "c122ffbb").

        Returns:
            SendResult with success status.
        """
        return await self.send(
            chat_id=chat_id,
            content="",
            metadata={"sticker_code": sticker_code},
        )

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Send a typing indicator to a MAX chat.

        MAX API does not have an explicit typing endpoint, so this is a no-op.
        """
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info about a MAX chat."""
        result = await self._api_request("GET", f"/chats/{chat_id}")
        if result:
            return {
                "name": result.get("title", chat_id),
                "type": "group" if result.get("type") == "group" else "dm",
            }
        return {"name": chat_id, "type": "dm"}

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        """Edit a previously sent message."""
        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")

        body: Dict[str, Any] = {"text": content[:self.MAX_MESSAGE_LENGTH]}
        if self._markdown:
            body["format"] = "markdown"

        result = await self._api_request(
            "POST",
            f"/messages/{message_id}/edit",
            json_body=body,
        )

        if result is not None:
            return SendResult(success=True, message_id=message_id)
        return SendResult(success=False, error="Edit failed")

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a message."""
        if not self._http_client:
            return False

        result = await self._api_request(
            "POST",
            f"/messages/{message_id}/delete",
        )
        return result is not None

    async def send_media(
        self,
        chat_id: str,
        file_path: str,
        media_type: str = "file",
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        """Upload and send a media file.

        Args:
            chat_id: Target chat or user ID.
            file_path: Local file path to upload.
            media_type: One of "image", "video", "audio", "file".
            caption: Optional caption text.
            reply_to: Optional message ID to reply to.

        Returns:
            SendResult with success status.
        """
        if not self._http_client:
            return SendResult(success=False, error="HTTP client not initialized")

        # Upload the file first
        upload_result = await self._api_upload(file_path)
        if not upload_result:
            return SendResult(success=False, error="File upload failed")

        # Build attachment
        attachment: Dict[str, Any] = {
            "type": media_type,
            "payload": upload_result,
        }

        body: Dict[str, Any] = {}

        if chat_id.startswith("-") or len(chat_id) > 12:
            body["chat_id"] = chat_id
        else:
            body["user_id"] = chat_id

        if caption:
            body["text"] = caption[:self.MAX_MESSAGE_LENGTH]
            if self._markdown:
                body["format"] = "markdown"

        body["attachments"] = [attachment]
        body["notify"] = True

        if reply_to:
            body["reply_to"] = reply_to

        result = await self._api_request("POST", "/messages", json_body=body)

        if result is not None:
            returned_mid = str(result.get("message_id", "") or result.get("mid", ""))
            return SendResult(success=True, message_id=returned_mid or uuid.uuid4().hex[:12])

        return SendResult(success=False, error="Failed to send media message")


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def _env_enablement() -> Optional[dict]:
    """Seed PlatformConfig.extra from env vars during gateway config load.

    Called by the platform registry before adapter construction, so
    ``hermes gateway status`` and ``get_connected_platforms()`` reflect
    env-only configuration without instantiating the HTTP client.
    Returns None when MAX is not minimally configured.
    """
    token = os.getenv("MAX_BOT_TOKEN", "").strip()
    if not token:
        return None

    seed: dict = {"token": token}

    # Only explicit env vars should override config.yaml.  If MAX_MODE is
    # absent, leave the YAML mode intact; the adapter constructor already
    # defaults to polling when neither YAML nor env config sets a mode.
    mode = os.getenv("MAX_MODE", "").strip().lower()
    if mode:
        seed["mode"] = mode

    webhook_url = os.getenv("MAX_WEBHOOK_URL", "").strip()
    if webhook_url:
        seed["webhook_url"] = webhook_url

    webhook_port = os.getenv("MAX_WEBHOOK_PORT", "").strip()
    if webhook_port:
        seed["webhook_port"] = int(webhook_port)

    home = os.getenv("MAX_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("MAX_HOME_CHANNEL_NAME", home),
        }

    return seed


async def _standalone_send(
    pconfig: PlatformConfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Out-of-process send for cron / send_message_tool fallbacks.

    Used when the gateway runner is not in this process (e.g. ``hermes cron``
    running standalone). Without this hook, ``deliver=max`` cron jobs fail
    with ``No live adapter for platform``.
    """
    if not HTTPX_AVAILABLE:
        return {"error": "MAX standalone send: httpx not installed"}

    extra = getattr(pconfig, "extra", {}) or {}
    token = extra.get("token") or os.getenv("MAX_BOT_TOKEN", "")
    if not token:
        return {"error": "MAX standalone send: MAX_BOT_TOKEN not configured"}

    markdown = bool(extra.get("markdown"))

    body: Dict[str, Any] = {"text": message[:MAX_MESSAGE_LENGTH], "notify": True}
    if markdown:
        body["format"] = "markdown"

    if chat_id.startswith("-") or len(chat_id) > 12:
        body["chat_id"] = chat_id
    else:
        body["user_id"] = chat_id

    # Handle media files if provided
    attachments: List[Dict[str, Any]] = []
    if media_files:
        async with httpx.AsyncClient(timeout=60.0) as client:
            for file_path in media_files:
                if not os.path.exists(file_path):
                    continue
                try:
                    with open(file_path, "rb") as f:
                        upload_resp = await client.post(
                            f"{MAX_API_BASE}/uploads",
                            headers={"Authorization": token},
                            files={"file": (os.path.basename(file_path), f)},
                        )
                    if upload_resp.status_code < 300:
                        attachments.append({
                            "type": "file",
                            "payload": upload_resp.json(),
                        })
                except Exception as exc:
                    logger.warning("MAX standalone upload error: %s", exc)

    if attachments:
        body["attachments"] = attachments

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{MAX_API_BASE}/messages",
                headers={
                    "Authorization": token,
                    "Content-Type": "application/json",
                },
                json=body,
            )
        if resp.status_code >= 300:
            return {"error": f"MAX HTTP {resp.status_code}: {resp.text[:200]}"}
        data = resp.json()
        mid = str(data.get("message_id", "") or data.get("mid", ""))
        return {"success": True, "platform": "max", "chat_id": chat_id, "message_id": mid}
    except Exception as exc:
        return {"error": f"MAX standalone send failed: {exc}"}


def register(ctx) -> None:
    """Plugin entry point - called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="max",
        label="MAX",
        adapter_factory=lambda cfg: MaxAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["MAX_BOT_TOKEN"],
        install_hint="pip install httpx   # already a Hermes dependency",
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="MAX_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="MAX_ALLOWED_USERS",
        allow_all_env="MAX_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji=":speech_balloon:",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are communicating via MAX messenger (MAKC by VK). "
            "MAX supports markdown formatting. "
            "Keep responses within the 4000-character limit. "
            "MAX is a Russian messenger - respond in the user's language."
        ),
    )
