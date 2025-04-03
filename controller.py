import win32gui
import win32api
import win32con
import mss
import mss.tools
import time
#from ai import AIPart

class SimActions:
    sct = mss.mss()
    monitor = None
    def __init__(self):
        windows_list = []
        toplist = []
        def enum_win(hwnd, result):
            win_text = win32gui.GetWindowText(hwnd)
            windows_list.append((hwnd, win_text))
        win32gui.EnumWindows(enum_win, toplist)
        self.game_hwnd = 0
        for (hwnd, win_text) in windows_list:
            if "Jstris" in win_text:
                self.game_hwnd = hwnd
                window_rect = win32gui.GetWindowRect(hwnd)
                l, t, r, b = window_rect
                width = r - l
                height = b - t
                self.monitor = {"top": t, "left": l, "width": width, "height": height}
    def clockwise(self):
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYDOWN, win32con.VK_UP)
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYUP, win32con.VK_UP)
    def c_clockwise(self):
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYDOWN, ord('z'), 0)
        win32api.SendMessage(self.game_hwnd, win32con.WM_CHAR, ord('z'), 0)
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYUP,  ord('z'), 0)
    def flip(self):
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYDOWN, ord('a'), 0)
        win32api.SendMessage(self.game_hwnd, win32con.WM_CHAR, ord('a'), 0)
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYUP,  ord('a'), 0)
    def down(self, t):
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYDOWN, win32con.VK_DOWN)
        time.sleep(t)
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYUP, win32con.VK_DOWN)
    def right(self):
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYDOWN, win32con.VK_RIGHT)
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYUP, win32con.VK_RIGHT)
    def left(self):
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYDOWN, win32con.VK_LEFT)
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYUP, win32con.VK_LEFT)
    def hard_drop(self):
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYDOWN, win32con.VK_SPACE)
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYUP, win32con.VK_SPACE)
    def hold(self):
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYDOWN, ord('c'), 0)
        win32api.SendMessage(self.game_hwnd, win32con.WM_CHAR, ord('c'), 0)
        win32api.SendMessage(self.game_hwnd, win32con.WM_KEYUP,  ord('c'), 0)
    
    def perform_action(self, action, grid, piece_shape):
        target_column = action["target"]["column"]
        final_rotation = action["final_rotation"]
        rotation_type = action["rotation_type"]

        # Get current position and orientation
        current_column = self.get_current_piece_column(grid, piece_shape)
        current_rotation = self.get_current_rotation(piece_shape)
        #Correct Rotation
        while current_rotation != final_rotation:
            if final_rotation == "clockwise":
                self.clockwise()
            elif final_rotation == "counterclockwise":
                self.c_clockwise()
            elif final_rotation == "flip":
                self.flip()
        #Align horizontally
        while current_column < target_column:
            self.right()
            current_column += 1
        while current_column > target_column:
            self.left()
            current_column -= 1
        
        self.hard_drop()