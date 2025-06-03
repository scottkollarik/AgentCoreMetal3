from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import time
from enum import Enum

class MetricType(Enum):
    """Types of metrics that can be collected."""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    GPU_USAGE = "gpu_usage"
    ERROR_COUNT = "error_count"
    SUCCESS_COUNT = "success_count"
    REQUEST_COUNT = "request_count"
    RESPONSE_TIME = "response_time"
    TOOL_HEALTH = "tool_health"

@dataclass
class MetricPoint:
    """A single data point for a metric."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Metric:
    """A collection of metric points for a specific metric type."""
    name: str
    type: MetricType
    points: List[MetricPoint] = field(default_factory=list)
    description: str = ""
    
    def add_point(self, value: float, labels: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Add a new data point to the metric."""
        self.points.append(MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels or {},
            metadata=metadata or {}
        ))

@dataclass
class ToolMetrics:
    """Metrics for a specific tool."""
    tool_name: str
    metrics: Dict[str, Metric] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Initialize default metrics for the tool."""
        for metric_type in MetricType:
            self.metrics[metric_type.value] = Metric(
                name=f"{self.tool_name}_{metric_type.value}",
                type=metric_type,
                description=f"{metric_type.value} for {self.tool_name}"
            )
    
    def record_execution(self, execution_time: float, success: bool, error: Optional[str] = None):
        """Record a tool execution."""
        self.last_updated = datetime.now()
        
        # Record execution time
        self.metrics[MetricType.EXECUTION_TIME.value].add_point(
            value=execution_time,
            metadata={"success": success, "error": error}
        )
        
        # Record success/failure
        if success:
            self.metrics[MetricType.SUCCESS_COUNT.value].add_point(value=1.0)
        else:
            self.metrics[MetricType.ERROR_COUNT.value].add_point(
                value=1.0,
                metadata={"error": error}
            )
    
    def record_resource_usage(self, cpu_usage: float, memory_usage: float, gpu_usage: Optional[float] = None):
        """Record resource usage metrics."""
        self.metrics[MetricType.CPU_USAGE.value].add_point(value=cpu_usage)
        self.metrics[MetricType.MEMORY_USAGE.value].add_point(value=memory_usage)
        if gpu_usage is not None:
            self.metrics[MetricType.GPU_USAGE.value].add_point(value=gpu_usage)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "tool_name": self.tool_name,
            "last_updated": self.last_updated.isoformat(),
            "metrics": {
                name: {
                    "type": metric.type.value,
                    "description": metric.description,
                    "points": [
                        {
                            "timestamp": point.timestamp.isoformat(),
                            "value": point.value,
                            "labels": point.labels,
                            "metadata": point.metadata
                        }
                        for point in metric.points
                    ]
                }
                for name, metric in self.metrics.items()
            }
        }
    
    def to_json(self) -> str:
        """Convert metrics to JSON string."""
        return json.dumps(self.to_dict(), indent=2) 