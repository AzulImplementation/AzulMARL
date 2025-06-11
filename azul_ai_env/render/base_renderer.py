"""
Base renderer class providing common functionality for all tab renderers.
"""

import matplotlib.pyplot as plt


class BaseRenderer:
    """Base class for all tab renderers"""
    
    def __init__(self):
        self.tile_colors = ['#4169E1', '#FFD700', '#DC143C', '#000000', '#F5F5F5']  # Blue, Yellow, Red, Black, White
        self.tile_names = ['Blue', 'Yellow', 'Red', 'Black', 'White']
        self.tile_letters = ['B', 'Y', 'R', 'K', 'W']
    
    def clear_axes(self, axes):
        """Clear all axes and set background color"""
        if isinstance(axes, dict):
            for ax in axes.values():
                ax.clear()
                ax.set_facecolor('#F0F8FF')
        else:
            axes.clear()
            axes.set_facecolor('#F0F8FF')
    
    def add_tile_count_labels(self, ax, bars, counts):
        """Add count labels on top of bars"""
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    def create_tile_bar_chart(self, ax, counts, title, ylabel='Count'):
        """Create a standard tile bar chart"""
        bars = ax.bar(range(5), counts, color=self.tile_colors, edgecolor='black', linewidth=1.5)
        bars[4].set_edgecolor('black')
        bars[4].set_linewidth(2)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(range(5))
        ax.set_xticklabels([f'{letter}' for letter in self.tile_letters], fontsize=10)
        
        self.add_tile_count_labels(ax, bars, counts)
        
        ax.set_ylim(0, max(counts) + 2 if max(counts) > 0 else 5)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        return bars
    
    def draw_tile_square(self, ax, x, y, tile_type, size=0.8, is_empty=False, alpha=1.0):
        """Draw a single tile square"""
        if is_empty:
            # Draw empty slot
            square = plt.Rectangle((x, y), size, size, facecolor='lightgray', 
                                 edgecolor='black', linewidth=1, alpha=0.3)
            ax.add_patch(square)
        elif tile_type == 5:  # First player marker
            # Draw first player marker
            square = plt.Rectangle((x, y), size, size, facecolor='white', 
                                 edgecolor='black', linewidth=2)
            ax.add_patch(square)
            ax.text(x + size/2, y + size/2, 'M',
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   color='black')
        else:
            # Draw regular tile
            square = plt.Rectangle((x, y), size, size, 
                                 facecolor=self.tile_colors[tile_type],
                                 edgecolor='black', linewidth=2, alpha=alpha)
            ax.add_patch(square)
            
            # Add tile letter
            text_color = 'white' if tile_type != 4 else 'black'
            ax.text(x + size/2, y + size/2, self.tile_letters[tile_type],
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   color=text_color)
    
    def render(self, data, axes):
        """Override this method in subclasses"""
        raise NotImplementedError("Subclasses must implement render method") 