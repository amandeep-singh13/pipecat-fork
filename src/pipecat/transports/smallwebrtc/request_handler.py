#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""SmallWebRTC request handler for managing peer connections.

This module provides a client for handling web requests and managing WebRTC connections.
"""

import asyncio
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

from aiortc.sdp import candidate_from_sdp
from fastapi import HTTPException
from loguru import logger

from pydantic import BaseModel, Field
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection


class SmallWebRTCRequest(BaseModel):
    """Small WebRTC transport session arguments for the runner."""

    sdp: str
    type: str
    pc_id: Optional[str] = Field(default=None, alias="pcId")
    restart_pc: Optional[bool] = Field(default=None, alias="restartPc")
    request_data: Optional[Any] = Field(default=None, alias="requestData")

    model_config = {"populate_by_name": True, "extra": "allow"}


class IceCandidate(BaseModel):
    """The remote ice candidate object received from the peer connection."""

    candidate: str
    sdp_mid: Optional[str] = Field(default=None, alias="sdpMid")
    sdp_mline_index: Optional[int] = Field(default=None, alias="sdpMLineIndex")

    model_config = {"populate_by_name": True, "extra": "allow"}


class SmallWebRTCPatchRequest(BaseModel):
    """Small WebRTC transport session arguments for the runner."""

    pc_id: str = Field(alias="pcId")
    candidate: IceCandidate

    model_config = {"populate_by_name": True, "extra": "allow"}


class ConnectionMode(Enum):
    """Enum defining the connection handling modes."""

    SINGLE = "single"  # Only one active connection allowed
    MULTIPLE = "multiple"  # Multiple simultaneous connections allowed


class SmallWebRTCRequestHandler:
    """SmallWebRTC request handler for managing peer connections.

    This class is responsible for:
      - Handling incoming SmallWebRTC requests.
      - Creating and managing WebRTC peer connections.
      - Supporting ESP32-specific SDP munging if enabled.
      - Invoking callbacks for newly initialized connections.
      - Supporting both single and multiple connection modes.
    """

    def __init__(
        self,
        ice_servers: Optional[List[IceServer]] = None,
        esp32_mode: bool = False,
        host: Optional[str] = None,
        connection_mode: ConnectionMode = ConnectionMode.MULTIPLE,
    ) -> None:
        """Initialize a SmallWebRTC request handler.

        Args:
            ice_servers (Optional[List[IceServer]]): List of ICE servers to use for WebRTC
                connections.
            esp32_mode (bool): If True, enables ESP32-specific SDP munging.
            host (Optional[str]): Host address used for SDP munging in ESP32 mode.
                Ignored if `esp32_mode` is False.
            connection_mode (ConnectionMode): Mode of operation for handling connections.
                SINGLE allows only one active connection, MULTIPLE allows several.
        """
        self._ice_servers = ice_servers
        self._esp32_mode = esp32_mode
        self._host = host
        self._connection_mode = connection_mode

        # Store connections by pc_id
        self._pcs_map: Dict[str, SmallWebRTCConnection] = {}

    def _check_single_connection_constraints(self, pc_id: Optional[str]) -> None:
        """Check if the connection request satisfies single connection mode constraints.

        Args:
            pc_id: The peer connection ID from the request

        Raises:
            HTTPException: If constraints are violated in single connection mode
        """
        if self._connection_mode != ConnectionMode.SINGLE:
            return

        if not self._pcs_map:  # No existing connections
            return

        # Get the existing connection (should be only one in single mode)
        existing_connection = next(iter(self._pcs_map.values()))

        if existing_connection.pc_id != pc_id and pc_id:
            logger.warning(
                f"Connection pc_id mismatch: existing={existing_connection.pc_id}, received={pc_id}"
            )
            raise HTTPException(status_code=400, detail="PC ID mismatch with existing connection")

        if not pc_id:
            logger.warning(
                "Cannot create new connection: existing connection found but no pc_id received"
            )
            raise HTTPException(
                status_code=400,
                detail="Cannot create new connection with existing connection active",
            )

    def update_ice_servers(self, ice_servers: Optional[List[IceServer]] = None):
        """Update the list of ICE servers used for WebRTC connections."""
        self._ice_servers = ice_servers

    async def handle_web_request(
        self,
        request: SmallWebRTCRequest,
        webrtc_connection_callback: Callable[[Any], Awaitable[None]],
    ) -> Optional[Dict[str, str]]:
        """Handle a SmallWebRTC request and resolve the pending answer.

        This method will:
          - Reuse an existing WebRTC connection if `pc_id` exists.
          - Otherwise, create a new `SmallWebRTCConnection`.
          - Invoke the provided callback with the connection.
          - Manage ESP32-specific munging if enabled.
          - Enforce single/multiple connection mode constraints.

        Args:
            request (SmallWebRTCRequest): The incoming WebRTC request, containing
                SDP, type, and optionally a `pc_id`.
            webrtc_connection_callback (Callable[[Any], Awaitable[None]]): An
                asynchronous callback function that is invoked with the WebRTC connection.

        Returns:
            Dictionary containing SDP answer, type, and peer connection ID,
            or None if no answer is available.

        Raises:
            HTTPException: If connection mode constraints are violated
            Exception: Any exception raised during request handling or callback execution
                will be logged and propagated.
        """
        try:
            pc_id = request.pc_id

            # Check connection mode constraints first
            self._check_single_connection_constraints(pc_id)

            # After constraints are satisfied, get the existing connection if any
            existing_connection = self._pcs_map.get(pc_id) if pc_id else None

            if existing_connection:
                pipecat_connection = existing_connection
                logger.info(f"Reusing existing connection for pc_id: {pc_id}")
                await pipecat_connection.renegotiate(
                    sdp=request.sdp,
                    type=request.type,
                    restart_pc=request.restart_pc or False,
                )
            else:
                pipecat_connection = SmallWebRTCConnection(ice_servers=self._ice_servers)
                await pipecat_connection.initialize(sdp=request.sdp, type=request.type)

                @pipecat_connection.event_handler("closed")
                async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
                    logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
                    self._pcs_map.pop(webrtc_connection.pc_id, None)

                # Invoke callback provided in runner arguments
                try:
                    await webrtc_connection_callback(pipecat_connection)
                    logger.debug(
                        f"webrtc_connection_callback executed successfully for peer: {pipecat_connection.pc_id}"
                    )
                except Exception as callback_error:
                    logger.error(
                        f"webrtc_connection_callback failed for peer {pipecat_connection.pc_id}: {callback_error}"
                    )

            answer = pipecat_connection.get_answer()

            if self._esp32_mode:
                from pipecat.runner.utils import smallwebrtc_sdp_munging

                answer["sdp"] = smallwebrtc_sdp_munging(answer["sdp"], self._host)

            self._pcs_map[answer["pc_id"]] = pipecat_connection

            return answer
        except Exception as e:
            logger.error(f"Error processing SmallWebRTC request: {e}")
            logger.debug(f"SmallWebRTC request details: {request}")
            raise

    async def handle_patch_request(self, request: SmallWebRTCPatchRequest):
        """Handle a SmallWebRTC patch candidate request."""
        peer_connection = self._pcs_map.get(request.pc_id)

        if not peer_connection:
            raise HTTPException(status_code=404, detail="Peer connection not found")

        c = request.candidate
        candidate = candidate_from_sdp(c.candidate)
        candidate.sdpMid = c.sdp_mid
        candidate.sdpMLineIndex = c.sdp_mline_index
        await peer_connection.add_ice_candidate(candidate)

    async def close(self):
        """Clear the connection map."""
        coros = [pc.disconnect() for pc in self._pcs_map.values()]
        await asyncio.gather(*coros)
        self._pcs_map.clear()
