import unittest
import os
from datetime import datetime
from orchestrators.component_context import ComponentContext, ComponentType
from utils.visualization import ComponentVisualizer, visualize_component_flow

class TestComponentVisualizer(unittest.TestCase):
    """Test suite for component visualization utilities."""
    
    def setUp(self):
        """Set up test cases with a sample component hierarchy."""
        # Create a sample component hierarchy
        self.root = ComponentContext(
            component_type=ComponentType.AGENT,
            name="root-agent"
        )
        
        self.child1 = self.root.create_child_context(
            component_type=ComponentType.TOOL,
            name="tool-1"
        )
        
        self.child2 = self.root.create_child_context(
            component_type=ComponentType.MEMORY,
            name="memory-1"
        )
        
        self.grandchild = self.child1.create_child_context(
            component_type=ComponentType.MODEL,
            name="model-1"
        )
        
        self.contexts = [self.root, self.child1, self.child2, self.grandchild]
        self.visualizer = ComponentVisualizer()
        
    def tearDown(self):
        """Clean up generated files after tests."""
        for ext in ['.png', '.svg', '.pdf', '.html']:
            if os.path.exists(f"component_graph{ext}"):
                os.remove(f"component_graph{ext}")
            if os.path.exists(f"test_graph{ext}"):
                os.remove(f"test_graph{ext}")
                
    def test_create_component_graph(self):
        """Test creation of static component graphs."""
        # Test PNG output
        output_path = self.visualizer.create_component_graph(
            self.contexts,
            output_format="png",
            filename="test_graph"
        )
        self.assertTrue(os.path.exists(output_path))
        
        # Test SVG output
        output_path = self.visualizer.create_component_graph(
            self.contexts,
            output_format="svg",
            filename="test_graph"
        )
        self.assertTrue(os.path.exists(output_path))
        
    def test_create_interactive_graph(self):
        """Test creation of interactive component graphs."""
        output_path = self.visualizer.create_interactive_graph(
            self.contexts,
            filename="test_graph.html"
        )
        self.assertTrue(os.path.exists(output_path))
        
    def test_node_levels_calculation(self):
        """Test calculation of node levels in the hierarchy."""
        levels = self.visualizer._calculate_node_levels(self.contexts)
        
        # Root should be at level 0
        self.assertEqual(levels[self.root.component_id], 0)
        
        # Children should be at level 1
        self.assertEqual(levels[self.child1.component_id], 1)
        self.assertEqual(levels[self.child2.component_id], 1)
        
        # Grandchild should be at level 2
        self.assertEqual(levels[self.grandchild.component_id], 2)
        
    def test_visualize_component_flow(self):
        """Test the convenience function for visualization."""
        # Test static visualization
        output_path = visualize_component_flow(
            self.contexts,
            output_format="png",
            interactive=False
        )
        self.assertTrue(os.path.exists(output_path))
        
        # Test interactive visualization
        output_path = visualize_component_flow(
            self.contexts,
            interactive=True
        )
        self.assertTrue(os.path.exists(output_path))
        
    def test_complex_hierarchy(self):
        """Test visualization of a more complex component hierarchy."""
        # Create a more complex hierarchy
        agent1 = ComponentContext(
            component_type=ComponentType.AGENT,
            name="agent-1"
        )
        
        tool1 = agent1.create_child_context(
            component_type=ComponentType.TOOL,
            name="tool-1"
        )
        
        tool2 = agent1.create_child_context(
            component_type=ComponentType.TOOL,
            name="tool-2"
        )
        
        memory1 = tool1.create_child_context(
            component_type=ComponentType.MEMORY,
            name="memory-1"
        )
        
        model1 = tool2.create_child_context(
            component_type=ComponentType.MODEL,
            name="model-1"
        )
        
        # Add cross-connections
        memory1.add_prior_component(tool2.component_id)
        model1.add_prior_component(memory1.component_id)
        
        complex_contexts = [agent1, tool1, tool2, memory1, model1]
        
        # Test both visualization types
        static_path = visualize_component_flow(
            complex_contexts,
            output_format="png",
            interactive=False
        )
        self.assertTrue(os.path.exists(static_path))
        
        interactive_path = visualize_component_flow(
            complex_contexts,
            interactive=True
        )
        self.assertTrue(os.path.exists(interactive_path))

if __name__ == '__main__':
    unittest.main() 