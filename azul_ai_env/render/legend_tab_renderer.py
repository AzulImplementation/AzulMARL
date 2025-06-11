"""
Legend tab renderer for displaying tile color legend.
"""

import matplotlib.pyplot as plt
from .base_renderer import BaseRenderer


class LegendTabRenderer(BaseRenderer):
    """Renderer for the legend tab showing tile colors and meanings"""
    
    def render(self, data, axes):
        """Render the legend tab content"""
        self.clear_axes(axes)
        
        # Draw legend
        ax = axes['legend']
        
        # Create legend elements
        legend_elements = []
        for i, (color, letter, name) in enumerate(zip(self.tile_colors, self.tile_letters, self.tile_names)):
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', 
                                               label=f'{letter} = {name}'))
        
        # Add first player marker to legend
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', 
                                           label='M = 1st Player Marker'))
        
        ax.axis('off')
        ax.legend(handles=legend_elements, loc='center', fontsize=14, 
                 title='Tiles', title_fontsize=16, 
                 frameon=True, fancybox=True, shadow=True) 