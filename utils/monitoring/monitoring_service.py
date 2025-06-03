import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import os
from pathlib import Path

from .metrics import ToolMetrics, MetricType

logger = logging.getLogger(__name__)

class MonitoringService:
    """Service for collecting and managing monitoring metrics."""
    
    def __init__(self, storage_path: str = "logs/metrics"):
        """Initialize the monitoring service.
        
        Args:
            storage_path: Path to store metric data
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.tool_metrics: Dict[str, ToolMetrics] = {}
        self._cleanup_task = None
    
    async def start(self):
        """Start the monitoring service."""
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info("Monitoring service started")
    
    async def stop(self):
        """Stop the monitoring service."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Monitoring service stopped")
    
    def get_tool_metrics(self, tool_name: str) -> ToolMetrics:
        """Get or create metrics for a tool."""
        if tool_name not in self.tool_metrics:
            self.tool_metrics[tool_name] = ToolMetrics(tool_name)
        return self.tool_metrics[tool_name]
    
    async def record_tool_execution(
        self,
        tool_name: str,
        execution_time: float,
        success: bool,
        error: Optional[str] = None,
        cpu_usage: Optional[float] = None,
        memory_usage: Optional[float] = None,
        gpu_usage: Optional[float] = None
    ):
        """Record a tool execution with metrics."""
        metrics = self.get_tool_metrics(tool_name)
        
        # Record execution metrics
        metrics.record_execution(execution_time, success, error)
        
        # Record resource usage if available
        if cpu_usage is not None and memory_usage is not None:
            metrics.record_resource_usage(cpu_usage, memory_usage, gpu_usage)
        
        # Save metrics to disk
        await self._save_metrics(tool_name)
    
    async def get_tool_health(self, tool_name: str) -> Dict[str, Any]:
        """Get health metrics for a tool."""
        metrics = self.get_tool_metrics(tool_name)
        
        # Calculate success rate
        success_count = sum(
            point.value for point in metrics.metrics[MetricType.SUCCESS_COUNT.value].points
        )
        error_count = sum(
            point.value for point in metrics.metrics[MetricType.ERROR_COUNT.value].points
        )
        total_count = success_count + error_count
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        
        # Calculate average execution time
        execution_times = [
            point.value for point in metrics.metrics[MetricType.EXECUTION_TIME.value].points
        ]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "tool_name": tool_name,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "total_executions": total_count,
            "last_updated": metrics.last_updated.isoformat()
        }
    
    async def _save_metrics(self, tool_name: str):
        """Save metrics to disk."""
        metrics = self.tool_metrics[tool_name]
        file_path = self.storage_path / f"{tool_name}_metrics.json"
        
        try:
            async with asyncio.Lock():
                # Read existing data if any
                existing_data = {}
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        existing_data = json.load(f)
                
                # Update with new metrics
                existing_data.update(metrics.to_dict())
                
                # Write back to file
                with open(file_path, 'w') as f:
                    json.dump(existing_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics for {tool_name}: {e}")
    
    async def _periodic_cleanup(self):
        """Periodically clean up old metrics data."""
        while True:
            try:
                # Keep last 7 days of data
                cutoff_date = datetime.now() - timedelta(days=7)
                
                for file_path in self.storage_path.glob("*_metrics.json"):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        # Filter out old data points
                        for metric_name, metric_data in data.get("metrics", {}).items():
                            metric_data["points"] = [
                                point for point in metric_data["points"]
                                if datetime.fromisoformat(point["timestamp"]) > cutoff_date
                            ]
                        
                        # Write back filtered data
                        with open(file_path, 'w') as f:
                            json.dump(data, f, indent=2)
                    except Exception as e:
                        logger.error(f"Error cleaning up {file_path}: {e}")
                
                await asyncio.sleep(3600)  # Run cleanup every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics cleanup: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying 