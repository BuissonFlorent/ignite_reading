import ast
import os
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import json
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set

class DependencyTracker:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.graph = nx.DiGraph()
        self.method_definitions: Dict[str, List[str]] = defaultdict(list)
        
    def parse_imports(self, file_path):
        """Parse import statements from a Python file"""
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
            
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module if node.module else ''
                for name in node.names:
                    imports.append(f"{module}.{name.name}")
                    
        return imports
    
    def find_method_definitions(self, file_path: str) -> Dict[str, List[int]]:
        """Find all method definitions and their line numbers in a file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Dictionary mapping method names to lists of line numbers
        """
        with open(file_path, 'r') as f:
            tree = ast.parse(f.read())
            
        methods = defaultdict(list)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                methods[node.name].append(node.lineno)
                
        return methods
    
    def build_graph(self):
        """Build dependency graph from project files"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'modules': {},
            'dependencies': {},
            'duplicate_methods': {}
        }
        
        for path in self.project_root.rglob('*.py'):
            if '__pycache__' in str(path):
                continue
                
            # Convert Windows path to module format
            relative_path = path.relative_to(self.project_root)
            module_path = str(relative_path).replace('\\', '.').replace('/', '.').replace('.py', '')
            
            self.graph.add_node(module_path)
            analysis['modules'][module_path] = str(path)
            
            try:
                # Check for duplicate methods
                methods = self.find_method_definitions(path)
                duplicates = {name: lines for name, lines in methods.items() if len(lines) > 1}
                if duplicates:
                    analysis['duplicate_methods'][module_path] = duplicates
                    print(f"\nWARNING: Duplicate methods found in {module_path}:")
                    for method, lines in duplicates.items():
                        print(f"  Method '{method}' defined at lines: {lines}")
                
                # Parse imports
                imports = self.parse_imports(path)
                deps = []
                for imp in imports:
                    if imp.startswith('src.') or imp.startswith('tests.'):
                        base_module = '.'.join(imp.split('.')[:3])
                        self.graph.add_edge(module_path, base_module)
                        deps.append(base_module)
                analysis['dependencies'][module_path] = deps
                
            except Exception as e:
                print(f"Error parsing {path}: {e}")
        
        # Save analysis to file
        output_file = os.path.join(self.project_root, 'tools', 'dependency_analysis.json')
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
    
    def visualize(self, output_file='dependencies.png'):
        """Create visualization of dependency graph"""
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_size=2000, 
                node_color='lightblue', font_size=8, arrows=True)
        plt.savefig(output_file)
        plt.close()
    
    def find_dependents(self, module):
        """Find all modules that depend on the given module"""
        if module not in self.graph:
            with open(os.path.join(self.project_root, 'tools', 'dependency_analysis.json'), 'r') as f:
                analysis = json.load(f)
            print(f"\nWarning: Module '{module}' not found in graph.")
            print("Available modules:", sorted(list(analysis['modules'].keys())))
            return []
        return list(nx.descendants(self.graph, module))
    
    def find_dependencies(self, module):
        """Find all modules that the given module depends on"""
        if module not in self.graph:
            with open(os.path.join(self.project_root, 'tools', 'dependency_analysis.json'), 'r') as f:
                analysis = json.load(f)
            print(f"\nWarning: Module '{module}' not found in graph.")
            print("Available modules:", sorted(list(analysis['modules'].keys())))
            return []
        return list(nx.ancestors(self.graph, module))

if __name__ == '__main__':
    # Get project root (parent of tools directory)
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    tracker = DependencyTracker(project_root)
    tracker.build_graph()
    tracker.visualize()
    
    # Example: Check dataset.py dependencies
    print("\nModules that depend on src.data.dataset:")
    print(tracker.find_dependents('src.data.dataset'))
    
    print("\nModules that src.data.dataset depends on:")
    print(tracker.find_dependencies('src.data.dataset')) 