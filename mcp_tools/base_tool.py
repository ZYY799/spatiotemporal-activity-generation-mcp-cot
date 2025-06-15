#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : base_tool.py
# @Time    : 2025/6/15 15:06
# @Desc    : Base class for all MCP tools implementing the Model Context Protocol


from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from datetime import datetime


@dataclass
class MCPMessage:
    """Unified message format for MCP communication"""
    message_type: str  # 'query', 'response', 'action', 'feedback'
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    message_id: str


class BaseMCPTool(ABC):
    """Base class for all MCP tools"""

    def __init__(self, tool_name: str, memory_manager=None):
        self.tool_name = tool_name
        self.memory_manager = memory_manager
        self.capabilities = []
        self.session_history = []

    @abstractmethod
    def process_query(self, query: MCPMessage) -> MCPMessage:
        """Process a query message and return response"""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of tool capabilities"""
        pass

    def _get_personal_preference(self, preference_key: str, default_value: Any = None) -> Any:
        """Get personal preference from memory manager"""
        if self.memory_manager:
            return self.memory_manager.get_preference(preference_key, default_value)
        return default_value

    def _create_response(self, data: Dict[str, Any], status: str = "success") -> MCPMessage:
        """Create standardized response message"""
        return MCPMessage(
            message_type="response",
            data=data,
            metadata={"tool": self.tool_name, "status": status},
            timestamp=datetime.now(),
            message_id=f"{self.tool_name}_{datetime.now().timestamp()}"
        )

    def _log_interaction(self, query: MCPMessage, response: MCPMessage) -> None:
        """Log tool interaction for session history"""
        self.session_history.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now()
        })
