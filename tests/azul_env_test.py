import pytest
import numpy as np
import copy
from azul_marl_env.azul_env import AzulEnv
from gymnasium import spaces


class TestAzulEnv:
    
    @pytest.mark.parametrize("player_count", [2, 3, 4])
    def test_observation_space(self, player_count):
        env = AzulEnv(player_count=player_count)
        
        for agent in env.agents:
            obs_space = env.observation_space(agent)
            
            assert isinstance(obs_space, spaces.Dict), f"Observation space should be Dict for {agent}"
            
            required_keys = {"factories", "center", "players", "bag", "lid"}
            assert set(obs_space.spaces.keys()) == required_keys, f"Missing keys in observation space for {agent}"
            
            factories_space = obs_space.spaces["factories"]
            assert isinstance(factories_space, spaces.Box), "Factories should be Box space"
            expected_factories = 1 + 2 * player_count
            assert factories_space.shape == (expected_factories, 5), f"Factories shape should be ({expected_factories}, 5)"
            assert factories_space.low.min() == 0, "Factories low should be 0"
            assert factories_space.high.max() == 4, "Factories high should be 4"
            
            center_space = obs_space.spaces["center"]
            assert isinstance(center_space, spaces.Box), "Center should be Box space"
            assert center_space.shape == (5,), "Center shape should be (5,)"
            assert center_space.low.min() == 0, "Center low should be 0"
            expected_center_high = 3 * expected_factories
            assert center_space.high.max() == expected_center_high, f"Center high should be {expected_center_high}"
            
            players_space = obs_space.spaces["players"]
            assert isinstance(players_space, spaces.Tuple), "Players should be Tuple space"
            assert len(players_space.spaces) == player_count, f"Should have {player_count} player spaces"
            
            for i, player_space in enumerate(players_space.spaces):
                assert isinstance(player_space, spaces.Dict), f"Player {i} should be Dict space"
                player_keys = {"pattern_lines", "wall", "floor", "is_starting", "score"}
                assert set(player_space.spaces.keys()) == player_keys, f"Missing keys in player {i} space"
                
                pattern_lines_space = player_space.spaces["pattern_lines"]
                assert isinstance(pattern_lines_space, spaces.Box), "Pattern lines should be Box space"
                assert pattern_lines_space.shape == (5, 5), "Pattern lines shape should be (5, 5)"
                
                wall_space = player_space.spaces["wall"]
                assert isinstance(wall_space, spaces.Box), "Wall should be Box space"
                assert wall_space.shape == (5, 5), "Wall shape should be (5, 5)"
                
                floor_space = player_space.spaces["floor"]
                assert isinstance(floor_space, spaces.Box), "Floor should be Box space"
                assert floor_space.shape == (7,), "Floor shape should be (7,)"
                
                is_starting_space = player_space.spaces["is_starting"]
                assert isinstance(is_starting_space, spaces.Discrete), "is_starting should be Discrete space"
                assert is_starting_space.n == 2, "is_starting should have 2 values"
                
                score_space = player_space.spaces["score"]
                assert isinstance(score_space, spaces.Discrete), "Score should be Discrete space"
                assert score_space.n == 241, "Score should have 241 possible values"
            
            for key in ["bag", "lid"]:
                space = obs_space.spaces[key]
                assert isinstance(space, spaces.Box), f"{key} should be Box space"
                assert space.shape == (5,), f"{key} shape should be (5,)"
                assert space.low.min() == 0, f"{key} low should be 0"
                assert space.high.max() == 100, f"{key} high should be 100"
    
    @pytest.mark.parametrize("player_count", [2, 3, 4])
    def test_action_space(self, player_count):
        env = AzulEnv(player_count=player_count)
        
        for agent in env.agents:
            action_space = env.action_space(agent)
            
            assert isinstance(action_space, spaces.MultiDiscrete), f"Action space should be MultiDiscrete for {agent}"
            
            expected_factories = 1 + 2 * player_count
            expected_nvec = [expected_factories + 1, 5, 20, 5]
            assert list(action_space.nvec) == expected_nvec, f"Action space nvec should be {expected_nvec}"
            
            sample_action = action_space.sample()
            assert len(sample_action) == 4, "Action should have 4 components"
            assert 0 <= sample_action[0] <= expected_factories, f"Factory index should be 0-{expected_factories}"
            assert 0 <= sample_action[1] <= 4, "Tile color should be 0-4"
            assert 0 <= sample_action[2] <= 19, "Floor tiles should be 0-19"
            assert 0 <= sample_action[3] <= 4, "Pattern line should be 0-4"
    
    @pytest.mark.parametrize("player_count", [2, 3, 4])
    def test_step_taking_from_center(self, player_count):
        env = AzulEnv(player_count=player_count)
        state, info = env.reset()
        
        valid_moves = info["valid_moves"]
        center_moves = [move for move in valid_moves if move[0] == 0]
        
        if not center_moves:
            factory_moves = [move for move in valid_moves if move[0] != 0]
            if factory_moves:
                initial_move = factory_moves[0]
                initial_state = copy.deepcopy(env.state)
                env.step(initial_move)
                
                current_agent = env.agent_selection
                if "valid_moves" in env.infos[current_agent]:
                    valid_moves = env.infos[current_agent]["valid_moves"]
                    center_moves = [move for move in valid_moves if move[0] == 0]
        
        action = center_moves[0]
        factory_index, tile_color, floor_tiles, pattern_line_index = action
        
        initial_state = copy.deepcopy(env.state)
        initial_center = copy.deepcopy(env.state["center"])
        current_player_index = env.agents.index(env.agent_selection)
        initial_player = copy.deepcopy(env.state["players"][current_player_index])
        
        env.step(action)
        
        if initial_center[tile_color] > 0:
            assert env.state["center"][tile_color] < initial_center[tile_color], \
                f"Center should have fewer {tile_color} tiles after taking from center"
        
        current_player_state = env.state["players"][current_player_index]
        
        if floor_tiles == 0:
            pattern_line = current_player_state["pattern_lines"][pattern_line_index]
            initial_pattern_line = initial_player["pattern_lines"][pattern_line_index]
            
            assert not np.array_equal(pattern_line, initial_pattern_line), \
                "Pattern line should change when tiles are placed"
        else:
            assert len(current_player_state["floor"]) != len(initial_player["floor"]), \
                "Floor should change when tiles overflow"
    
    @pytest.mark.parametrize("player_count", [2, 3, 4])
    def test_step_taking_from_factory(self, player_count):
        env = AzulEnv(player_count=player_count)
        state, info = env.reset()
        
        # Get valid moves from reset
        valid_moves = info["valid_moves"]
        factory_moves = [move for move in valid_moves if move[0] == 1]
        
        action = factory_moves[0]
        factory_index, tile_color, floor_tiles, pattern_line_index = action
        assert factory_index == 1, "Factory index should be 1"
        
        initial_factory = copy.deepcopy(env.state["factories"][0])
        initial_center = copy.deepcopy(env.state["center"])
        current_player_index = env.agents.index(env.agent_selection)
        initial_player = copy.deepcopy(env.state["players"][current_player_index])
        
        # Make the move
        env.step(action)
        
        # Verify state changes
        # Factory should be empty or have fewer tiles
        current_factory = env.state["factories"][0]
        assert np.sum(current_factory) < np.sum(initial_factory), \
            "Factory should have fewer tiles after taking from it"
        
        # Factory should have no tiles of the selected color
        assert current_factory[tile_color] == 0, \
            f"Factory should have no {tile_color} tiles after taking all of that color"
        
        # Center should have received remaining tiles from factory (if any)
        remaining_tiles_moved_to_center = False
        for color in range(5):
            if color != tile_color and initial_factory[color] > 0:
                if env.state["center"][color] > initial_center[color]:
                    remaining_tiles_moved_to_center = True
                    break
        
        # If there were other colors in the factory, they should have moved to center
        other_colors_existed = any(initial_factory[color] > 0 for color in range(5) if color != tile_color)
        if other_colors_existed:
            assert remaining_tiles_moved_to_center, \
                "Remaining tiles from factory should move to center"
        
        # Player state should change
        current_player_state = env.state["players"][current_player_index]
        
        # Check that pattern line or floor changed
        if floor_tiles == 0:  # Tiles went to pattern line
            pattern_line = current_player_state["pattern_lines"][pattern_line_index]
            initial_pattern_line = initial_player["pattern_lines"][pattern_line_index]
            
            # Pattern line should have more tiles or be different
            assert not np.array_equal(pattern_line, initial_pattern_line), \
                "Pattern line should change when tiles are placed"
            
            # Check that the correct tile color was placed
            non_empty_tiles = pattern_line[pattern_line != 5]
            if len(non_empty_tiles) > 0:
                assert all(tile == tile_color for tile in non_empty_tiles), \
                    f"Pattern line should contain only color {tile_color}"
        else:  # Some tiles went to floor
            assert len(current_player_state["floor"]) != len(initial_player["floor"]), \
                "Floor should change when tiles overflow"
    
    @pytest.mark.parametrize("player_count", [2, 3, 4])
    def test_step_board_state_consistency(self, player_count):
        """Test that board state remains consistent after step operations."""
        env = AzulEnv(player_count=player_count)
        state, info = env.reset()
        
        # Take several moves and verify state consistency
        for move_count in range(3):  # Test first 3 moves
            # Get valid moves from current agent's info
            current_agent = env.agent_selection
            if "valid_moves" in env.infos[current_agent]:
                valid_moves = env.infos[current_agent]["valid_moves"]
            elif move_count == 0:  # First move, use reset info
                valid_moves = info["valid_moves"]
            else:
                valid_moves = []
            
            if not valid_moves:
                break
            
            action = valid_moves[0]
            
            # Make move
            env.step(action)
            
            # Verify state structure is maintained
            assert "factories" in env.state, "State should contain factories"
            assert "center" in env.state, "State should contain center"
            assert "players" in env.state, "State should contain players"
            assert "bag" in env.state, "State should contain bag"
            assert "lid" in env.state, "State should contain lid"
            
            # Verify all tile counts are non-negative
            assert np.all(env.state["center"] >= 0), "Center tile counts should be non-negative"
            for factory in env.state["factories"]:
                assert np.all(factory >= 0), "Factory tile counts should be non-negative"
            assert np.all(env.state["bag"] >= 0), "Bag tile counts should be non-negative"
            assert np.all(env.state["lid"] >= 0), "Lid tile counts should be non-negative"
            
            # Verify player states
            for i, player in enumerate(env.state["players"]):
                assert player["score"] >= 0, f"Player {i} score should be non-negative"
                assert len(player["floor"]) <= 7, f"Player {i} floor should not exceed 7 tiles"
                
                # Pattern lines should have valid structure
                for row_idx, row in enumerate(player["pattern_lines"]):
                    non_empty_count = np.sum(row != 5)
                    assert non_empty_count <= row_idx + 1, f"Pattern line {row_idx} should not exceed {row_idx + 1} tiles"
                    
                    # If not empty, all tiles should be the same color
                    if non_empty_count > 0:
                        non_empty_tiles = row[row != 5]
                        assert len(set(non_empty_tiles)) == 1, f"Pattern line {row_idx} should have only one color"
    
    @pytest.mark.parametrize("player_count", [2, 3, 4])
    def test_observe(self, player_count):
        """Test that observe method returns the correct state for each agent."""
        env = AzulEnv(player_count=player_count)
        state, _ = env.reset()

        # Test that observe returns the same state for all agents
        for agent in env.agents:
            observed_state = env.observe(agent)
            assert observed_state == state, f"Observed state for {agent} should match environment state"
            
            # Verify structure of observed state
            assert "factories" in observed_state, "Observed state should contain factories"
            assert "center" in observed_state, "Observed state should contain center"
            assert "players" in observed_state, "Observed state should contain players"
            assert "bag" in observed_state, "Observed state should contain bag"
            assert "lid" in observed_state, "Observed state should contain lid"
            
            # Verify shapes match environment configuration
            assert observed_state["factories"].shape == (env.factories, 5), "Factories shape mismatch"
            assert observed_state["center"].shape == (5,), "Center shape mismatch"
            assert len(observed_state["players"]) == player_count, "Number of players mismatch"
            assert observed_state["bag"].shape == (5,), "Bag shape mismatch"
            assert observed_state["lid"].shape == (5,), "Lid shape mismatch"