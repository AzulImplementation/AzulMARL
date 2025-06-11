"""
Main tab renderer for displaying scores, center, and factories.
"""

import matplotlib.pyplot as plt
from .base_renderer import BaseRenderer


class MainTabRenderer(BaseRenderer):
    """Renderer for the main tab containing scores, center, and factories"""
    
    def render(self, data, axes):
        """Render the main tab content"""
        self.clear_axes(axes)
        
        self.draw_scores(data, axes['scores'])
        self.draw_center(data, axes['center'])
        self.draw_factories(data, axes['factories'])
    
    def draw_scores(self, data, ax):
        """Draw player scores table"""
        ax.set_title('Player Scores', fontsize=16, fontweight='bold', pad=25)
        player_scores = [player["score"] for player in data['state']["players"]]
        
        ax.axis('off')
        table_data = []
        for i, score in enumerate(player_scores):
            table_data.append([f'Player {i+1}', str(score)])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Player', 'Score'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.3, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4169E1')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows with alternating colors
        for i in range(1, len(table_data) + 1):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F0F8FF')
                else:
                    table[(i, j)].set_facecolor('#E6E6FA')
                table[(i, j)].set_text_props(weight='bold')
    
    def draw_center(self, data, ax):
        """Draw center statistics"""
        center_counts = data['center_counts']
        self.create_tile_bar_chart(ax, center_counts, 'Center Statistics')
    
    def draw_factories(self, data, ax):
        """Draw factories display"""
        ax.set_title('Factory Displays', fontsize=16, fontweight='bold', pad=15)
        
        factories = data['factories']
        num_factories = len(factories)
        cols = min(4, num_factories)  # Max 4 factories per row
        rows = (num_factories + cols - 1) // cols  # Ceiling division
        
        factory_width = 4  # Each factory takes 4 tile positions
        factory_height = 1
        margin_x = 1
        margin_y = 0.5
        
        # Draw each factory
        for factory_idx, factory_tiles in enumerate(factories):
            row = factory_idx // cols
            col = factory_idx % cols
            
            # Calculate factory position
            start_x = col * (factory_width + margin_x)
            start_y = (rows - 1 - row) * (factory_height + margin_y)
            
            # Draw factory label
            ax.text(start_x + factory_width/2, start_y + factory_height + 0.1,
                   f'Factory No. {factory_idx + 1}', ha='center', va='bottom',
                   fontsize=12, fontweight='bold')
            
            # Draw tiles in this factory
            tile_pos = 0
            # First, place actual tiles
            actual_tiles = []
            for tile_type, count in enumerate(factory_tiles):
                for _ in range(count):
                    if tile_pos < 4:  # Max 4 tiles per factory
                        actual_tiles.append(tile_type)
                        tile_pos += 1
            
            # Now draw all 4 positions (actual tiles + neutral for empty positions)
            for pos in range(4):
                x = start_x + pos
                y = start_y
                
                if pos < len(actual_tiles):
                    # Draw actual tile
                    tile_type = actual_tiles[pos]
                    self.draw_tile_square(ax, x, y, tile_type)
                else:
                    # Draw neutral/empty tile
                    self.draw_tile_square(ax, x, y, None, is_empty=True)
        
        # Set axis limits and remove ticks
        total_width = cols * (factory_width + margin_x) - margin_x
        total_height = rows * (factory_height + margin_y) + 0.5
        
        ax.set_xlim(-0.5, total_width + 0.5)
        ax.set_ylim(-0.5, total_height)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal') 