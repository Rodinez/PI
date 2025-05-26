import cv2
import numpy as np
import pygame
import mss
import win32gui
import win32con
import win32api

def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        return frame, monitor

def detect_objects(template_path, prev_positions):
    template = cv2.imread(template_path, 0)
    if template is None:
        return prev_positions

    screen_gray, monitor = capture_screen()
    result = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
    
    threshold = 0.9
    loc = np.where(result >= threshold)
    new_positions = [(pt[0], pt[1], template.shape[1], template.shape[0]) for pt in zip(*loc[::-1])]
    
    return new_positions if new_positions else prev_positions

def make_window_clickthrough(hwnd):
    ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
    win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT)
    win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(0, 0, 0), 0, win32con.LWA_COLORKEY)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

def draw_overlay(template_path):
    monitor = mss.mss().monitors[1]
    pygame.init()
    screen = pygame.display.set_mode((monitor["width"], monitor["height"]), pygame.NOFRAME | pygame.SRCALPHA)
    pygame.display.set_caption("Overlay")
    hwnd = pygame.display.get_wm_info()["window"]
    make_window_clickthrough(hwnd)
    clock = pygame.time.Clock()
    running = True
    prev_positions = []

    while running:
        screen.fill((0, 0, 0, 0))
        prev_positions = detect_objects(template_path, prev_positions)
        
        print(prev_positions)

        for (x, y, w, h) in prev_positions:
            pygame.draw.rect(screen, (0, 255, 0), (x, y, w, h), 2)
        
        pygame.display.update()
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
    
    pygame.quit()

draw_overlay("C:/xampp/htdocs/Projetos/PI/A_template.png")
