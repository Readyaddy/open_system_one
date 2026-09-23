"""Playable Snake with graphics, wired up to watch the exp7 ("Open-One") model
play via its text choice interface (see open_one_agent.py).

Usage:
    python game.py                          # heuristic agent (no checkpoint yet)
    python game.py --ckpt path/to/exp7.pt    # exp7 model plays
    python game.py --human                   # you play, arrow keys
    python game.py --ckpt path/to/exp7.pt --fps 4 --k 6
"""
import argparse
import sys

import os

import pygame

from snake_env import SnakeGame, Direction, DIRECTION_ORDER

DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "exp7_latest.pt")

CELL = 24
MARGIN = 1
HUD_HEIGHT = 110

BG = (18, 18, 22)
GRID = (32, 32, 38)
SNAKE_HEAD = (110, 220, 120)
SNAKE_BODY = (60, 160, 90)
FOOD = (230, 90, 90)
TEXT = (230, 230, 235)
DEAD = (200, 60, 60)
BAR_BG = (40, 40, 48)
BAR_FILL = (90, 150, 230)

KEY_TO_DIR = {
    pygame.K_UP: Direction.UP, pygame.K_w: Direction.UP,
    pygame.K_DOWN: Direction.DOWN, pygame.K_s: Direction.DOWN,
    pygame.K_LEFT: Direction.LEFT, pygame.K_a: Direction.LEFT,
    pygame.K_RIGHT: Direction.RIGHT, pygame.K_d: Direction.RIGHT,
}


def draw_hud(screen, font, small_font, game, probs, agent_label, width_px):
    y0 = game.height * CELL
    pygame.draw.rect(screen, BG, (0, y0, width_px, HUD_HEIGHT))

    status = f"{agent_label}   score {game.score}   length {len(game.snake)}   steps {game.steps}"
    if not game.alive:
        status += "   -- GAME OVER (press R to restart, Esc to quit)"
    screen.blit(font.render(status, True, TEXT if game.alive else DEAD), (10, y0 + 8))

    if probs:
        bar_w = 110
        gap = 14
        x = 10
        for d in DIRECTION_ORDER:
            p = probs.get(d, 0.0)
            label = small_font.render(f"{d.name:<5} {p:.2f}", True, TEXT)
            screen.blit(label, (x, y0 + 38))
            pygame.draw.rect(screen, BAR_BG, (x, y0 + 58, bar_w, 12))
            pygame.draw.rect(screen, BAR_FILL, (x, y0 + 58, int(bar_w * max(0.0, min(1.0, p))), 12))
            x += bar_w + gap


def draw_board(screen, game):
    for x in range(game.width):
        for y in range(game.height):
            rect = (x * CELL, y * CELL, CELL - MARGIN, CELL - MARGIN)
            pygame.draw.rect(screen, GRID, rect)

    for i, (x, y) in enumerate(game.snake):
        color = SNAKE_HEAD if i == 0 else SNAKE_BODY
        pygame.draw.rect(screen, color, (x * CELL, y * CELL, CELL - MARGIN, CELL - MARGIN))

    if game.food:
        fx, fy = game.food
        pygame.draw.rect(screen, FOOD, (fx * CELL, fy * CELL, CELL - MARGIN, CELL - MARGIN))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT,
                     help="Path to a trained exp7 checkpoint (.pt). Defaults to "
                          "checkpoints/exp7a_best_zeroshot_epoch6.pt; pass --ckpt '' to force the heuristic.")
    ap.add_argument("--width", type=int, default=20)
    ap.add_argument("--height", type=int, default=20)
    ap.add_argument("--fps", type=float, default=10.0, help="Game ticks per second in AI mode.")
    ap.add_argument("--k", type=int, default=None, help="Recurrent-depth passes to run (default: model's k_max).")
    ap.add_argument("--human", action="store_true", help="Play yourself with arrow keys / WASD instead of the model.")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    pygame.init()
    pygame.display.set_caption("Snake -- Open-One (exp7) plays")

    game = SnakeGame(width=args.width, height=args.height, seed=args.seed)
    width_px = game.width * CELL
    height_px = game.height * CELL + HUD_HEIGHT
    screen = pygame.display.set_mode((width_px, height_px))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)
    small_font = pygame.font.SysFont("consolas", 15)

    agent = None
    agent_label = "HUMAN"
    if not args.human:
        from open_one_agent import OpenOneAgent
        agent = OpenOneAgent(ckpt_path=args.ckpt, k=args.k)
        agent_label = agent.label if agent.available else "HEURISTIC (no checkpoint loaded)"

    pending_human_dir = None
    probs = None
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                    probs = None
                elif args.human and event.key in KEY_TO_DIR:
                    pending_human_dir = KEY_TO_DIR[event.key]

        if game.alive:
            if args.human:
                move = pending_human_dir or game.direction
            else:
                move, probs = agent.decide(game)
            game.step(move)

        screen.fill(BG)
        draw_board(screen, game)
        draw_hud(screen, font, small_font, game, probs, agent_label, width_px)
        pygame.display.flip()

        clock.tick(args.fps if not args.human else 12)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
