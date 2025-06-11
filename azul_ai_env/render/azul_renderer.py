"""
Main Azul renderer that coordinates all tab renderers and manages the tabbed interface.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from .main_tab_renderer import MainTabRenderer
from .legend_tab_renderer import LegendTabRenderer
from .bag_lid_tab_renderer import BagLidTabRenderer
from .player_tab_renderer import PlayerTabRenderer


class AzulRenderer:
    """Main renderer class that manages the tabbed interface and coordinates tab renderers"""
    
    def __init__(self):
        # Initialize tab renderers
        self.main_renderer = MainTabRenderer()
        self.legend_renderer = LegendTabRenderer()
        self.bag_lid_renderer = BagLidTabRenderer()
        self.player_renderer = PlayerTabRenderer()
        
        # Figure and UI components
        self.fig = None
        self.current_tab = 0
        self.tab_buttons = []
        self.tab_names = []
        self.all_axes = {}
        
        # Enable interactive mode
        plt.ion()
    
    def initialize_figure(self, num_players):
        """Initialize the matplotlib figure with tabbed interface"""
        if self.fig is None:
            self.create_tabbed_interface(num_players)
    
    def create_tabbed_interface(self, num_players):
        """Create the main figure with tabs"""
        self.fig = plt.figure(figsize=(18, 14))
        self.fig.patch.set_facecolor('#E6E6FA')  # Light lavender background
        
        # Create tab names
        self.tab_names = ['Main', 'Legend', 'Bag & Lid'] + [f'Player {i+1}' for i in range(num_players)]
        
        # Create tab buttons at the top
        self.create_tab_buttons()
        
        # Create axes for each tab
        self.create_all_tab_axes(num_players)
        
        # Initially show the main tab
        self.switch_tab(0)
    
    def create_tab_buttons(self):
        """Create tab buttons at the top of the interface"""
        button_width = 0.12
        button_height = 0.04
        button_y = 0.94
        
        for i, tab_name in enumerate(self.tab_names):
            button_x = 0.05 + i * (button_width + 0.01)
            ax_button = plt.axes([button_x, button_y, button_width, button_height])
            button = Button(ax_button, tab_name)
            button.on_clicked(lambda event, tab=i: self.switch_tab(tab))
            self.tab_buttons.append((ax_button, button))
    
    def create_all_tab_axes(self, num_players):
        """Create axes for all tabs"""
        # Main tab axes
        main_axes = {
            'scores': plt.subplot2grid((3, 1), (0, 0)),
            'center': plt.subplot2grid((3, 1), (1, 0)),
            'factories': plt.subplot2grid((3, 1), (2, 0))
        }
        self.all_axes['main'] = main_axes
        
        # Legend tab axes
        legend_axes = {
            'legend': plt.subplot2grid((1, 1), (0, 0))
        }
        self.all_axes['legend'] = legend_axes
        
        # Bag & Lid tab axes
        bag_lid_axes = {
            'bag': plt.subplot2grid((2, 1), (0, 0)),
            'lid': plt.subplot2grid((2, 1), (1, 0))
        }
        self.all_axes['bag_lid'] = bag_lid_axes
        
        # Player tab axes
        for i in range(num_players):
            player_axes = {
                'floor': plt.subplot2grid((3, 1), (0, 0)),
                'pattern': plt.subplot2grid((3, 1), (1, 0)),
                'wall': plt.subplot2grid((3, 1), (2, 0))
            }
            self.all_axes[f'player_{i}'] = player_axes
        
        # Set background colors for all axes and hide initially
        for tab_key, axes_dict in self.all_axes.items():
            for ax in axes_dict.values():
                ax.set_facecolor('#F0F8FF')
                ax.set_visible(False)
    
    def switch_tab(self, tab_index):
        """Switch to the specified tab"""
        self.current_tab = tab_index
        
        # Hide all axes
        for axes_dict in self.all_axes.values():
            for ax in axes_dict.values():
                ax.set_visible(False)
        
        # Show current tab axes
        if tab_index == 0:  # Main tab
            for ax in self.all_axes['main'].values():
                ax.set_visible(True)
        elif tab_index == 1:  # Legend tab
            for ax in self.all_axes['legend'].values():
                ax.set_visible(True)
        elif tab_index == 2:  # Bag & Lid tab
            for ax in self.all_axes['bag_lid'].values():
                ax.set_visible(True)
        else:  # Player tabs
            player_idx = tab_index - 3
            if f'player_{player_idx}' in self.all_axes:
                for ax in self.all_axes[f'player_{player_idx}'].values():
                    ax.set_visible(True)
        
        # Update button colors to show active tab
        for i, (ax_button, button) in enumerate(self.tab_buttons):
            if i == tab_index:
                ax_button.set_facecolor('#4169E1')  # Active tab color
            else:
                ax_button.set_facecolor('#E6E6FA')  # Inactive tab color
        
        # Redraw current tab content if data is available
        if hasattr(self, 'current_data'):
            self.update_current_tab_content()
    
    def render(self, state, bag_counts, lid_counts, center_counts, factories):
        """Main render method called by AzulEnv"""
        if state is None:
            return
        
        num_players = len(state["players"])
        
        # Initialize figure if needed
        self.initialize_figure(num_players)
        
        # Prepare data for renderers
        self.current_data = {
            'state': state,
            'bag_counts': bag_counts,
            'lid_counts': lid_counts,
            'center_counts': center_counts,
            'factories': factories
        }
        
        # Update current tab content
        self.update_current_tab_content()
        
        # Update display
        self.update_display()
    
    def update_current_tab_content(self):
        """Update the content of the currently visible tab"""
        if not hasattr(self, 'current_data'):
            return
        
        data = self.current_data
        
        if self.current_tab == 0:  # Main tab
            self.main_renderer.render(data, self.all_axes['main'])
        elif self.current_tab == 1:  # Legend tab
            self.legend_renderer.render(data, self.all_axes['legend'])
        elif self.current_tab == 2:  # Bag & Lid tab
            self.bag_lid_renderer.render(data, self.all_axes['bag_lid'])
        else:  # Player tabs
            player_idx = self.current_tab - 3
            if f'player_{player_idx}' in self.all_axes:
                self.player_renderer.render(data, self.all_axes[f'player_{player_idx}'], player_idx)
    
    def update_display(self):
        """Update the matplotlib display"""
        self.fig.tight_layout()
        self.fig.subplots_adjust(hspace=0.4)  # Add vertical spacing between subplots
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show(block=False)
        plt.pause(0.01)  # Brief pause to ensure the plot updates
    
    def close(self):
        """Close the renderer and clean up resources"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.current_tab = 0
            self.tab_buttons = []
            self.tab_names = []
            self.all_axes = {}
            if hasattr(self, 'current_data'):
                delattr(self, 'current_data') 