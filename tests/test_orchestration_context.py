import unittest
from datetime import datetime
from orchestrators.component_context import ComponentContext, ComponentType, ComponentData

class TestComponentContext(unittest.TestCase):
    """Test suite for ComponentContext, the building block for component state and relationship tracking."""
    
    def setUp(self):
        """Set up test cases with a basic component context"""
        self.basic_context = ComponentContext(
            component_type=ComponentType.AGENT,
            name="test-agent"
        )
        
    def test_basic_initialization(self):
        """Test basic component context initialization"""
        self.assertEqual(self.basic_context.component_type, ComponentType.AGENT)
        self.assertEqual(self.basic_context.name, "test-agent")
        self.assertEqual(self.basic_context.version, "1.0.0")
        self.assertIsNotNone(self.basic_context.flow_id)
        self.assertIsNotNone(self.basic_context.run_id)
        self.assertIsNotNone(self.basic_context.component_id)
        
    def test_full_id_format(self):
        """Test the format of the full component identifier"""
        full_id = self.basic_context.full_id
        parts = full_id.split(":")
        self.assertEqual(len(parts), 3)  # flow_id:run_id:component_id
        self.assertEqual(parts[0], self.basic_context.flow_id)
        self.assertEqual(parts[1], self.basic_context.run_id)
        self.assertEqual(parts[2], self.basic_context.component_id)
        
    def test_create_child_context(self):
        """Test child component context creation and inheritance"""
        child = self.basic_context.create_child_context(
            component_type=ComponentType.TOOL,
            name="child-tool"
        )
        
        # Test inheritance
        self.assertEqual(child.flow_id, self.basic_context.flow_id)
        self.assertEqual(child.run_id, self.basic_context.run_id)
        self.assertEqual(child.parent_id, self.basic_context.component_id)
        
        # Test prior component tracking
        self.assertIn(self.basic_context.component_id, child.prior_component_ids)
        
    def test_component_data_handling(self):
        """Test component data operations and validation"""
        # Test setting component data
        self.basic_context.update_component_data(
            input_data={"task": "test"},
            metrics={"start_time": 1234567890.0}
        )
        
        # Test getting component data
        self.assertEqual(
            self.basic_context.get_component_data("input_data"),
            {"task": "test"}
        )
        self.assertEqual(
            self.basic_context.get_component_data("metrics"),
            {"start_time": 1234567890.0}
        )
        
        # Test invalid data field
        with self.assertRaises(ValueError):
            self.basic_context.update_component_data(invalid_field="value")
            
        with self.assertRaises(ValueError):
            self.basic_context.get_component_data("invalid_field")
            
    def test_json_serialization(self):
        """Test JSON serialization and deserialization of component context"""
        # Add some data to the context
        self.basic_context.update_component_data(
            input_data={"task": "test"},
            metrics={"start_time": 1234567890.0}
        )
        
        # Convert to JSON
        json_data = self.basic_context.to_json()
        
        # Create new context from JSON
        new_context = ComponentContext.from_json(json_data)
        
        # Verify data integrity
        self.assertEqual(new_context.component_type, self.basic_context.component_type)
        self.assertEqual(new_context.name, self.basic_context.name)
        self.assertEqual(new_context.flow_id, self.basic_context.flow_id)
        self.assertEqual(new_context.run_id, self.basic_context.run_id)
        self.assertEqual(new_context.component_id, self.basic_context.component_id)
        self.assertEqual(
            new_context.get_component_data("input_data"),
            self.basic_context.get_component_data("input_data")
        )
        
    def test_execution_chain(self):
        """Test execution chain tracking across components"""
        # Create a chain of contexts
        child1 = self.basic_context.create_child_context(
            component_type=ComponentType.TOOL,
            name="tool-1"
        )
        child2 = child1.create_child_context(
            component_type=ComponentType.AGENT,
            name="agent-1"
        )
        
        # Test execution chain
        chain = child2.get_execution_chain()
        self.assertEqual(len(chain), 3)
        self.assertEqual(chain[0], self.basic_context.component_id)
        self.assertEqual(chain[1], child1.component_id)
        self.assertEqual(chain[2], child2.component_id)
        
    def test_intent_executor_tracking(self):
        """Test intent executor tracking and duplicate prevention"""
        # Add intent executors
        self.basic_context.add_intent_executor("executor-1")
        self.basic_context.add_intent_executor("executor-2")
        
        # Test executor list
        self.assertIn("executor-1", self.basic_context.intent_executor_ids)
        self.assertIn("executor-2", self.basic_context.intent_executor_ids)
        
        # Test duplicate prevention
        self.basic_context.add_intent_executor("executor-1")
        self.assertEqual(
            self.basic_context.intent_executor_ids.count("executor-1"),
            1
        )

    def test_planning_agent_context(self):
        """Test context creation for planning agent based on notebook usage"""
        planning_context = ComponentContext(
            component_type=ComponentType.AGENT,
            name="planning-agent",
            version="1.0.0"
        )
        
        # Add planning-specific data
        planning_context.update_component_data(
            input_data={"task": "Create a summary of AI trends"},
            metrics={
                "model_name": "llama2",
                "temperature": 0.7,
                "max_tokens": 1000
            }
        )
        
        # Verify planning data
        self.assertEqual(
            planning_context.get_component_data("input_data")["task"],
            "Create a summary of AI trends"
        )
        self.assertEqual(
            planning_context.get_component_data("metrics")["model_name"],
            "llama2"
        )

    def test_execution_agent_context(self):
        """Test context creation for execution agent based on notebook usage"""
        execution_context = ComponentContext(
            component_type=ComponentType.AGENT,
            name="execution-agent"
        )
        
        # Add execution-specific data
        execution_context.update_component_data(
            input_data={
                "step": "Research AI trends",
                "tool_names": "web_search",
                "tools": "- web_search: Search the web for information"
            },
            metrics={
                "model_name": "llama2",
                "temperature": 0.7,
                "max_tokens": 1000
            }
        )
        
        # Verify execution data
        self.assertEqual(
            execution_context.get_component_data("input_data")["step"],
            "Research AI trends"
        )
        self.assertIn(
            "web_search",
            execution_context.get_component_data("input_data")["tool_names"]
        )

    def test_vector_memory_context(self):
        """Test context creation for vector memory based on notebook usage"""
        memory_context = ComponentContext(
            component_type=ComponentType.MEMORY,
            name="vector-memory"
        )
        
        # Add memory-specific data
        memory_context.update_component_data(
            input_data={
                "embedding_model": "ollama:mxbai-embed-large",
                "index_name": "agent_memory",
                "dimension": 1024
            },
            metrics={
                "document_count": 0,
                "last_update": datetime.now().isoformat()
            }
        )
        
        # Verify memory data
        self.assertEqual(
            memory_context.get_component_data("input_data")["embedding_model"],
            "ollama:mxbai-embed-large"
        )
        self.assertEqual(
            memory_context.get_component_data("input_data")["dimension"],
            1024
        )

    def test_error_handling(self):
        """Test error handling in component context"""
        error_context = ComponentContext(
            component_type=ComponentType.AGENT,
            name="error-test-agent"
        )
        
        # Add error data
        error_context.update_component_data(
            error={
                "message": "Failed to execute step",
                "type": "ExecutionError",
                "context": {
                    "step": "Research AI trends",
                    "agent": "execution_agent"
                }
            }
        )
        
        # Verify error data
        error_data = error_context.get_component_data("error")
        self.assertEqual(error_data["message"], "Failed to execute step")
        self.assertEqual(error_data["type"], "ExecutionError")
        self.assertEqual(
            error_data["context"]["step"],
            "Research AI trends"
        )

    def test_component_state_tracking(self):
        """Test tracking of component state changes"""
        state_context = ComponentContext(
            component_type=ComponentType.AGENT,
            name="state-test-agent"
        )
        
        # Add state data
        state_context.update_component_data(
            state={
                "status": "running",
                "current_step": 1,
                "total_steps": 5,
                "progress": 0.2
            }
        )
        
        # Update state
        state_context.update_component_data(
            state={
                "status": "completed",
                "current_step": 5,
                "total_steps": 5,
                "progress": 1.0
            }
        )
        
        # Verify final state
        final_state = state_context.get_component_data("state")
        self.assertEqual(final_state["status"], "completed")
        self.assertEqual(final_state["progress"], 1.0)

if __name__ == '__main__':
    unittest.main() 