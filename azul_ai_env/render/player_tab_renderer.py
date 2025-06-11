"""
Player tab renderer for displaying individual player boards.
"""

import matplotlib.pyplot as plt
from .base_renderer import BaseRenderer


class PlayerTabRenderer(BaseRenderer):
    """Renderer for player tabs showing floor, pattern lines, and wall"""
    
    def render(self, data, axes, player_idx):
        """Render the player tab content"""
        if player_idx >= len(data['state']["players"]):
            return
        
        self.clear_axes(axes)
        player_data = data['state']["players"][player_idx]
        
        # Draw floor, pattern lines, and wall
        self.draw_floor(player_data, axes['floor'], player_idx)
        self.draw_pattern_lines(player_data, axes['pattern'], player_idx)
        self.draw_wall(player_data, axes['wall'], player_idx)
    
    def draw_floor(self, player_data, ax, player_idx):
        """Draw floor for a player"""
        ax.set_title('Floor', fontsize=14, fontweight='bold', pad=15)
        
        # Floor penalty values (standard Azul penalties)
        floor_penalties = [-1, -1, -2, -2, -2, -3, -3]
        floor_tiles = player_data["floor"]
        
        # Draw floor positions
        for pos in range(7):
            x = pos + 0.5
            y = 0.5
            
            # Draw penalty value below the slot
            ax.text(x + 0.4, y - 0.2, f'{floor_penalties[pos]}',
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   color='red')
            
            # Draw tile if present
            if pos < len(floor_tiles):
                tile_type = floor_tiles[pos]
                self.draw_tile_square(ax, x, y, tile_type)
            else:
                # Draw empty slot
                self.draw_tile_square(ax, x, y, None, is_empty=True)
        
        # Set axis limits and remove ticks
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    
    def draw_pattern_lines(self, player_data, ax, player_idx):
        """Draw pattern lines for a player"""
        ax.set_title('Pattern Lines', fontsize=14, fontweight='bold', pad=15)
        
        pattern_lines = player_data["pattern_lines"]
        
        # Draw each pattern line row in triangular formation (1 to 5 tiles)
        for row in range(5):
            row_length = row + 1  # Row 0 has 1 tile, row 1 has 2 tiles, etc.
            pattern_row = pattern_lines[row]
            
            # Right-align all rows to create proper Azul pattern lines structure
            right_edge = 4.5  # Right edge position
            start_x = right_edge - (row_length * 0.8)  # Start position for right alignment
            y = 4 - row + 0.5  # Row 0 at top, row 4 at bottom
            
            # Draw tiles for this row
            for pos in range(row_length):
                x = start_x + pos * 0.8
                
                # Check if there's a tile in this position
                if pos < len(pattern_row):
                    tile_type = pattern_row[pos]
                    
                    # Only draw if it's not empty (value 5 means empty)
                    if tile_type != 5:
                        self.draw_tile_square(ax, x, y, tile_type, size=0.7)
                    else:
                        self.draw_tile_square(ax, x, y, None, size=0.7, is_empty=True)
                else:
                    self.draw_tile_square(ax, x, y, None, size=0.7, is_empty=True)
        
        # Set axis limits and styling
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        
        # Add a subtle grid
        ax.grid(True, alpha=0.2, linestyle='--')
    
    def draw_wall(self, player_data, ax, player_idx):
        """Draw wall for a player"""
        ax.set_title('Wall', fontsize=14, fontweight='bold', pad=15)
        
        wall = player_data["wall"]
        
        # Standard Azul wall pattern (each row has different tile order)
        wall_pattern = [
            [0, 1, 2, 3, 4],  # Row 0
            [4, 0, 1, 2, 3],  # Row 1
            [3, 4, 0, 1, 2],  # Row 2
            [2, 3, 4, 0, 1],  # Row 3
            [1, 2, 3, 4, 0]   # Row 4
        ]
        
        # Draw wall for this player
        for row in range(5):
            for col in range(5):
                x = col + 0.5
                y = 4 - row + 0.5  # Row 0 at top, row 4 at bottom
                
                # Get expected tile type for this position
                expected_tile_type = wall_pattern[row][col]
                
                # Draw background - colored for non-white tiles, neutral for white
                if expected_tile_type == 4:  # White tile position
                    # Use neutral background for white positions to avoid confusion
                    background = plt.Rectangle((x, y), 0.8, 0.8, 
                                             facecolor='lightgray', 
                                             edgecolor='black', linewidth=1, alpha=0.2)
                else:
                    # Use colored background for other tiles with low alpha
                    background = plt.Rectangle((x, y), 0.8, 0.8, 
                                             facecolor=self.tile_colors[expected_tile_type], 
                                             edgecolor='black', linewidth=1, alpha=0.15)
                ax.add_patch(background)
                
                # Check if there's a tile placed in this position
                placed_tile_type = wall[row][col]
                
                if placed_tile_type != 5:  # 5 means empty
                    # Draw placed tile
                    self.draw_tile_square(ax, x, y, placed_tile_type)
                else:
                    # Show expected tile letter faintly for empty positions
                    ax.text(x + 0.4, y + 0.4, self.tile_letters[expected_tile_type],
                           ha='center', va='center', fontsize=10, fontweight='bold',
                           color='gray', alpha=0.3)
        
        # Set axis limits and styling
        ax.set_xlim(-0.5, 5.5)
        ax.set_ylim(-0.5, 5.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        
        # Add a subtle grid
        ax.grid(True, alpha=0.2, linestyle='--') 