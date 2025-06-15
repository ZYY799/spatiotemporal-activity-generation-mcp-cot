#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : zy312
# @File    : tool_manager.py
# @Time    : 2025/6/15 15:34
# @Desc    : MCP Tools Manager for coordinating all MCP tools


from typing import Dict, List, Any, Optional
from .base_tool import BaseMCPTool, MCPMessage
from .temporal_tools import TemporalManagementTool
from .spatial_tools import SpatialNavigationTool
from .environmental_tools import EnvironmentalPerceptionTool
from .social_tools import SocialCollaborationTool
from .evaluation_tools import ExperienceEvaluationTool
from datetime import datetime


class MCPToolManager:
    """Central manager for coordinating all MCP tools"""

    def __init__(self, memory_manager=None):
        self.memory_manager = memory_manager

        # Initialize specialized tools
        self.tools = {
            "temporal": TemporalManagementTool(memory_manager),
            "spatial": SpatialNavigationTool(memory_manager),
            "environmental": EnvironmentalPerceptionTool(memory_manager),
            "social": SocialCollaborationTool(memory_manager),
            "evaluation": ExperienceEvaluationTool(memory_manager)
        }

        self.tool_routing = self._build_tool_routing()

    def process_query(self, tool_name: str, query: MCPMessage) -> MCPMessage:
        """Route and execute query on specified tool"""

        if tool_name not in self.tools:
            return self._create_error_response(f"Unknown tool: {tool_name}")

        try:
            tool = self.tools[tool_name]
            response = tool.process_query(query)
            self._log_tool_interaction(tool_name, query, response)
            return response

        except Exception as e:
            return self._create_error_response(f"Tool execution error: {str(e)}")

    def get_available_tools(self) -> Dict[str, List[str]]:
        """Return all tools and their capabilities"""
        tools_info = {}
        for tool_name, tool in self.tools.items():
            tools_info[tool_name] = tool.get_capabilities()
        return tools_info

    def suggest_tool_for_query(self, query_text: str) -> List[str]:
        """Suggest appropriate tools based on query keywords"""
        query_lower = query_text.lower()
        suggested_tools = []

        for tool_name, keywords in self.tool_routing.items():
            if any(keyword in query_lower for keyword in keywords):
                suggested_tools.append(tool_name)

        return suggested_tools

    def _build_tool_routing(self) -> Dict[str, List[str]]:
        """Build keyword-based routing table for tool suggestions"""
        return {
            "temporal": ["time", "schedule", "duration", "when", "timing", "calendar", "deadline"],
            "spatial": ["location", "place", "route", "distance", "where", "navigation", "poi"],
            "environmental": ["weather", "crowd", "event", "condition", "environment", "climate"],
            "social": ["friend", "group", "social", "companion", "together", "contact", "meeting"],
            "evaluation": ["rate", "score", "evaluate", "assess", "compare", "satisfaction", "quality"]
        }

    def _create_error_response(self, error_message: str) -> MCPMessage:
        """Generate standardized error response"""
        return MCPMessage(
            message_type="response",
            data={"error": error_message},
            metadata={"status": "error", "tool": "tool_manager"},
            timestamp=datetime.now(),
            message_id=f"error_{datetime.now().timestamp()}"
        )

    def _log_tool_interaction(self, tool_name: str, query: MCPMessage, response: MCPMessage) -> None:
        """Log tool interactions for monitoring and debugging"""
        print(f"Tool: {tool_name}, Query: {query.data.get('query_type', 'unknown')}, "
              f"Status: {response.metadata.get('status', 'unknown')}")