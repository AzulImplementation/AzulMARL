"""
Bag and Lid tab renderer for displaying tile counts in bag and lid.
"""

from .base_renderer import BaseRenderer


class BagLidTabRenderer(BaseRenderer):
    """Renderer for the bag and lid tab showing tile counts"""
    
    def render(self, data, axes):
        """Render the bag and lid tab content"""
        self.clear_axes(axes)
        
        # Draw bag
        self.draw_bag(data, axes['bag'])
        
        # Draw lid
        self.draw_lid(data, axes['lid'])
    
    def draw_bag(self, data, ax):
        """Draw bag tile counts"""
        bag_counts = data['bag_counts']
        self.create_tile_bar_chart(ax, bag_counts, 'Bag Statistics')
    
    def draw_lid(self, data, ax):
        """Draw lid tile counts"""
        lid_counts = data['lid_counts']
        self.create_tile_bar_chart(ax, lid_counts, 'Lid Statistics') 