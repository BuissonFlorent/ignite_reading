import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.visualization.protocol_effect import ProtocolEffectPlot
from src.examples.visualize_student import load_last_model

def plot_protocol_effect(model_dir='results', save_dir='results'):
    """Create and save protocol effect visualization."""
    # Load trained model
    model = load_last_model(model_dir)
    
    # Create and save plot
    plotter = ProtocolEffectPlot()
    fig = plotter.plot_protocol_effect(model)
    plotter.save_plot(fig, save_dir)
    
    return fig

if __name__ == '__main__':
    plot_protocol_effect() 