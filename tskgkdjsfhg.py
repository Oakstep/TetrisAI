def step(self, action):
        # Apply action
        rotation_index_map = {
            0: "spawn",
            1: "clockwise",
            2: "flip",
            3: "counterclockwise"
        }

        if action == (-1, -1) or action == [-1, -1]:
            if self.can_hold:
                self.can_hold = False
                if self.held_piece:
                    self.current_piece, self.held_piece = self.held_piece, self.current_piece
                    self.current_rotation_key = "spawn"
                else:
                    self.held_piece = self.current_piece
                    self.current_piece = self.get_next_piece()
                    self.current_rotation_key = "spawn"
                reward = self.reward()
            else:
                return self.get_state(), -1, False
        else:
            rotation_key = rotation_index_map[action[1]]
            column = action[0]
            valid = self.place_piece(rotation_key, column)
            if not valid:
                self.game_over = True
                return self.get_state(), -10, True
            reward = self.reward()
            self.current_piece = self.get_next_piece()
            self.current_rotation_key = "spawn"
            self.can_hold = True
        
        self.game_over = self.is_terminal_state()
        
        return self.get_state(), reward, self.game_over