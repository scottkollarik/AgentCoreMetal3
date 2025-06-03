import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any
import aiohttp
from redis.asyncio import Redis

from .monitoring.monitoring_service import MonitoringService
from .monitoring.resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)

class ToolConfig:
    """Configuration for a tool container."""
    def __init__(
        self,
        name: str,
        container: str,
        endpoint: str,
        health_check_endpoint: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0
    ):
        self.name = name
        self.container = container
        self.endpoint = endpoint
        self.health_check_endpoint = health_check_endpoint or f"{endpoint}/health"
        self.max_retries = max_retries
        self.timeout = timeout

class ToolOrchestrator:
    """Orchestrates communication between agent and tool containers."""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        monitoring_service: Optional[MonitoringService] = None
    ):
        """Initialize the tool orchestrator.
        
        Args:
            redis_url: URL for Redis connection
            monitoring_service: Optional monitoring service instance
        """
        self.redis = Redis.from_url(redis_url)
        self.tools: Dict[str, ToolConfig] = {}
        self.monitoring_service = monitoring_service or MonitoringService()
        self.resource_monitor = ResourceMonitor()
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def start(self):
        """Start the orchestrator and monitoring service."""
        self._session = aiohttp.ClientSession()
        await self.monitoring_service.start()
        logger.info("Tool orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator and monitoring service."""
        if self._session:
            await self._session.close()
        await self.monitoring_service.stop()
        await self.redis.close()
        logger.info("Tool orchestrator stopped")
    
    def register_tool(self, tool_config: ToolConfig):
        """Register a tool with the orchestrator."""
        self.tools[tool_config.name] = tool_config
        logger.info(f"Registered tool: {tool_config.name}")
    
    async def check_tool_health(self, tool_name: str) -> bool:
        """Check if a tool is healthy."""
        if tool_name not in self.tools:
            logger.warning(f"Tool {tool_name} not registered")
            return False
            
        tool_config = self.tools[tool_name]
        
        try:
            async with self._session.get(
                tool_config.health_check_endpoint,
                timeout=tool_config.timeout
            ) as response:
                is_healthy = response.status == 200
                
                # Record health check in monitoring
                await self.monitoring_service.record_tool_execution(
                    tool_name=tool_name,
                    execution_time=0.0,  # Health check time is negligible
                    success=is_healthy,
                    error=None if is_healthy else f"Health check failed with status {response.status}"
                )
                
                return is_healthy
        except Exception as e:
            logger.error(f"Error checking health of {tool_name}: {e}")
            
            # Record failed health check
            await self.monitoring_service.record_tool_execution(
                tool_name=tool_name,
                execution_time=0.0,
                success=False,
                error=str(e)
            )
            
            return False
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute a tool with the given parameters."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not registered")
            
        tool_config = self.tools[tool_name]
        timeout = timeout or tool_config.timeout
        
        # Get initial resource usage
        initial_resources = self.resource_monitor.get_system_resources()
        
        start_time = time.time()
        last_error = None
        
        for attempt in range(tool_config.max_retries):
            try:
                async with self._session.post(
                    f"{tool_config.endpoint}/execute",
                    json=parameters,
                    timeout=timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        execution_time = time.time() - start_time
                        
                        # Get final resource usage
                        final_resources = self.resource_monitor.get_system_resources()
                        
                        # Record successful execution
                        await self.monitoring_service.record_tool_execution(
                            tool_name=tool_name,
                            execution_time=execution_time,
                            success=True,
                            cpu_usage=final_resources.get("cpu_percent", 0.0),
                            memory_usage=final_resources.get("memory_gb", 0.0),
                            gpu_usage=final_resources.get("gpu_percent")
                        )
                        
                        return result
                    else:
                        last_error = f"Tool returned status {response.status}"
            except asyncio.TimeoutError:
                last_error = "Tool execution timed out"
            except Exception as e:
                last_error = str(e)
            
            if attempt < tool_config.max_retries - 1:
                await asyncio.sleep(1)  # Wait before retry
        
        # Record failed execution
        execution_time = time.time() - start_time
        await self.monitoring_service.record_tool_execution(
            tool_name=tool_name,
            execution_time=execution_time,
            success=False,
            error=last_error
        )
        
        raise RuntimeError(f"Tool {tool_name} failed after {tool_config.max_retries} attempts: {last_error}")
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools with their health status."""
        tools_info = []
        
        for tool_name, tool_config in self.tools.items():
            is_healthy = await self.check_tool_health(tool_name)
            health_info = await self.monitoring_service.get_tool_health(tool_name)
            
            tools_info.append({
                "name": tool_name,
                "container": tool_config.container,
                "endpoint": tool_config.endpoint,
                "is_healthy": is_healthy,
                "health_metrics": health_info
            })
        
        return tools_info

# Usage example:
# async def main():
#     orchestrator = ToolOrchestrator()
#     try:
#         result = await orchestrator.execute_tool(
#             "chart_generator",
#             {"data": data, "type": "bar"}
#         )
#         print(result)
#     finally:
#         await orchestrator.close()
#
# asyncio.run(main()) 