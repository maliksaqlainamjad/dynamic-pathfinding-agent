"""
=============================================================
  DYNAMIC PATHFINDING AGENT
  Algorithms : A* and Greedy Best-First Search (GBFS)
  Heuristics : Manhattan Distance and Euclidean Distance
  GUI        : Tkinter  (no extra libraries needed)
  Run        : python pathfinding_agent.py
=============================================================
"""

import tkinter as tk
from tkinter import scrolledtext
import heapq
import random
import time
import math
from collections import defaultdict

# ──────────────────────────────────────────────────────────
#  COLORS
# ──────────────────────────────────────────────────────────
C = {
    "bg"        : "#1b1b2f",
    "panel"     : "#162447",
    "dark"      : "#0f3460",
    "border"    : "#1a1a2e",
    "accent"    : "#e43f5a",
    "green"     : "#1db954",
    "red"       : "#e43f5a",
    "text"      : "#eaeaea",
    "sub"       : "#a8a8b3",
    "empty"     : "#1f2041",
    "wall"      : "#4a4a6a",
    "start"     : "#1db954",
    "goal"      : "#f5a623",
    "visited"   : "#4a90d9",
    "path"      : "#00e5ff",
    "agent"     : "#ff4dff",
    "grid_line" : "#2a2a4f",
    "metric_bg" : "#0f3460",
    "metric_val": "#00e5ff",
}

# ──────────────────────────────────────────────────────────
#  HEURISTICS
# ──────────────────────────────────────────────────────────
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ──────────────────────────────────────────────────────────
#  SEARCH  (A* and GBFS)
# ──────────────────────────────────────────────────────────
def run_search(grid, start, goal, algorithm, hfn):
    rows, cols = len(grid), len(grid[0])

    def neighbors(n):
        r, c = n
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                yield (nr, nc)

    tie = 0
    heap = []
    came_from = {}
    g = defaultdict(lambda: float('inf'))
    g[start] = 0
    h0 = hfn(start, goal)
    heapq.heappush(heap, (h0 if algorithm=="GBFS" else h0, tie, start))
    tie += 1
    visited = []
    closed = set()

    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur in closed: continue
        closed.add(cur)
        if cur not in (start, goal):
            visited.append(cur)
        if cur == goal:
            path, node = [], goal
            while node in came_from:
                path.append(node); node = came_from[node]
            path.append(start); path.reverse()
            return path, visited, len(path)-1
        for nb in neighbors(cur):
            if nb in closed: continue
            ng = g[cur] + 1
            if ng < g[nb]:
                came_from[nb] = cur
                g[nb] = ng
                h = hfn(nb, goal)
                f = h if algorithm=="GBFS" else ng+h
                heapq.heappush(heap, (f, tie, nb))
                tie += 1
    return None, visited, 0


# ──────────────────────────────────────────────────────────
#  APP
# ──────────────────────────────────────────────────────────
class App:
    CS = 26  # cell size px

    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Pathfinding Agent  |  A* & GBFS")
        self.root.configure(bg=C["bg"])
        self.root.geometry("1150x700")
        self.root.minsize(950, 600)

        # State
        self.ROWS = 20
        self.COLS = 25
        self.grid  = [[0]*self.COLS for _ in range(self.ROWS)]
        self.start = (0, 0)
        self.goal  = (self.ROWS-1, self.COLS-1)
        self.visited_set = set()
        self.path_set    = set()
        self.agent_pos   = None
        self.agent_step  = 0
        self.running     = False
        self.edit_mode   = "wall"

        # Control vars
        self.var_algo    = tk.StringVar(value="A*")
        self.var_heur    = tk.StringVar(value="Manhattan")
        self.var_rows    = tk.IntVar(value=20)
        self.var_cols    = tk.IntVar(value=25)
        self.var_density = tk.DoubleVar(value=0.28)
        self.var_speed   = tk.IntVar(value=60)
        self.var_dynamic = tk.BooleanVar(value=False)
        self.var_spawn   = tk.DoubleVar(value=0.04)
        self.var_status  = tk.StringVar(value="  Generate a map, then press START SEARCH.")

        # Metric vars
        self.var_nodes   = tk.StringVar(value="0")
        self.var_cost    = tk.StringVar(value="--")
        self.var_time    = tk.StringVar(value="--")
        self.var_replans = tk.StringVar(value="0")

        self._build_ui()
        self._draw_grid()

    # ─────────────────────────────────────────
    #  BUILD UI
    # ─────────────────────────────────────────
    def _build_ui(self):
        # ── STATUS BAR (top) ──────────────────
        tk.Frame(self.root, bg=C["dark"], pady=5).pack(fill=tk.X)
        status_bar = tk.Frame(self.root, bg=C["dark"])
        # (rebuilt below with label inside)
        status_bar2 = tk.Frame(self.root, bg=C["dark"], pady=4)
        status_bar2.pack(fill=tk.X, side=tk.TOP)
        tk.Label(status_bar2, textvariable=self.var_status,
                 bg=C["dark"], fg=C["text"],
                 font=("Consolas", 10)).pack(side=tk.LEFT, padx=10)

        # ── METRIC BAR ───────────────────────
        mbar = tk.Frame(self.root, bg=C["metric_bg"], pady=6)
        mbar.pack(fill=tk.X, side=tk.TOP)
        for lbl, var in [("Nodes Visited", self.var_nodes),
                          ("Path Cost",     self.var_cost),
                          ("Time (ms)",     self.var_time),
                          ("Re-plans",      self.var_replans)]:
            f = tk.Frame(mbar, bg=C["metric_bg"], padx=22)
            f.pack(side=tk.LEFT)
            tk.Label(f, text=lbl, bg=C["metric_bg"],
                     fg=C["sub"], font=("Consolas",9)).pack()
            tk.Label(f, textvariable=var, bg=C["metric_bg"],
                     fg=C["metric_val"],
                     font=("Consolas",16,"bold")).pack()

        # ── BODY ─────────────────────────────
        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill=tk.BOTH, expand=True)

        # Left panel (scrollable)
        panel_outer = tk.Frame(body, bg=C["panel"], width=240)
        panel_outer.pack(side=tk.LEFT, fill=tk.Y)
        panel_outer.pack_propagate(False)

        # Add a canvas+scrollbar so panel is scrollable
        panel_canvas = tk.Canvas(panel_outer, bg=C["panel"],
                                  highlightthickness=0, width=225)
        panel_scroll = tk.Scrollbar(panel_outer, orient=tk.VERTICAL,
                                     command=panel_canvas.yview)
        panel_canvas.configure(yscrollcommand=panel_scroll.set)
        panel_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        panel_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Inner frame inside the scrollable canvas
        p = tk.Frame(panel_canvas, bg=C["panel"], padx=12, pady=8)
        panel_win = panel_canvas.create_window((0,0), window=p, anchor="nw")

        def _on_frame_configure(e):
            panel_canvas.configure(scrollregion=panel_canvas.bbox("all"))
        def _on_canvas_configure(e):
            panel_canvas.itemconfig(panel_win, width=e.width)
        p.bind("<Configure>", _on_frame_configure)
        panel_canvas.bind("<Configure>", _on_canvas_configure)

        # Mousewheel on panel
        def _on_mousewheel(e):
            panel_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        panel_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._build_panel(p)

        # Right: grid canvas
        right = tk.Frame(body, bg=C["bg"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        cframe = tk.Frame(right, bg=C["bg"])
        cframe.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(cframe, bg=C["bg"],
                                highlightthickness=0, cursor="crosshair")
        vs = tk.Scrollbar(cframe, orient=tk.VERTICAL,   command=self.canvas.yview)
        hs = tk.Scrollbar(right,  orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        vs.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        hs.pack(fill=tk.X)

        self.canvas.bind("<Button-1>",  self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)

        # Legend
        legend = tk.Frame(right, bg=C["bg"], pady=3)
        legend.pack(fill=tk.X)
        for name, color in [("Start",C["start"]),("Goal",C["goal"]),
                             ("Wall",C["wall"]),("Visited",C["visited"]),
                             ("Path",C["path"]),("Agent",C["agent"])]:
            f = tk.Frame(legend, bg=C["bg"])
            f.pack(side=tk.LEFT, padx=6)
            tk.Canvas(f, width=13, height=13, bg=color,
                      highlightthickness=0).pack(side=tk.LEFT)
            tk.Label(f, text=name, bg=C["bg"], fg=C["sub"],
                     font=("Consolas",8)).pack(side=tk.LEFT, padx=2)

    def _build_panel(self, p):
        """Build all controls inside the scrollable left panel."""

        def section(title):
            tk.Frame(p, bg=C["border"], height=1).pack(fill=tk.X, pady=(12,3))
            tk.Label(p, text=title, bg=C["panel"], fg=C["accent"],
                     font=("Consolas",9,"bold")).pack(anchor="w")

        def btn(text, cmd, color=None, pady=6):
            b = tk.Button(p, text=text, command=cmd,
                          bg=color or C["dark"], fg="white",
                          relief=tk.FLAT, font=("Consolas",10,"bold"),
                          cursor="hand2", pady=pady,
                          activebackground=C["accent"],
                          activeforeground="white")
            b.pack(fill=tk.X, pady=3)
            return b

        def slider(label, var, lo, hi, res):
            tk.Label(p, text=label, bg=C["panel"], fg=C["text"],
                     font=("Consolas",9)).pack(anchor="w", pady=(5,0))
            tk.Scale(p, variable=var, from_=lo, to=hi, resolution=res,
                     orient=tk.HORIZONTAL, bg=C["panel"], fg=C["text"],
                     troughcolor=C["border"], highlightthickness=0,
                     activebackground=C["accent"], length=200).pack()

        def radios(var, options):
            for val in options:
                tk.Radiobutton(p, text=val, variable=var, value=val,
                               bg=C["panel"], fg=C["text"],
                               selectcolor=C["bg"],
                               activebackground=C["panel"],
                               activeforeground=C["accent"],
                               font=("Consolas",9)).pack(anchor="w")

        def spin_row(label, var, lo, hi):
            f = tk.Frame(p, bg=C["panel"])
            f.pack(fill=tk.X, pady=2)
            tk.Label(f, text=label, bg=C["panel"], fg=C["text"],
                     font=("Consolas",9), width=8, anchor="w").pack(side=tk.LEFT)
            tk.Spinbox(f, from_=lo, to=hi, textvariable=var, width=6,
                       bg=C["border"], fg=C["text"],
                       buttonbackground=C["dark"],
                       relief=tk.FLAT, font=("Consolas",9)).pack(side=tk.LEFT)

        # ══ START / STOP / RESET  ← AT THE TOP so always visible ══
        section("CONTROLS")
        btn("▶  START SEARCH", self._start_search, "#1db954", pady=8)
        btn("⏹  STOP",         self._stop,         "#e43f5a")
        btn("↺  RESET VIEW",   self._reset_view)

        slider("Speed  (ms/step)", self.var_speed, 10, 500, 10)

        # ── GRID SETTINGS ──
        section("GRID SETTINGS")
        spin_row("Rows :", self.var_rows, 5, 50)
        spin_row("Cols :", self.var_cols, 5, 60)
        slider("Obstacle Density", self.var_density, 0.05, 0.60, 0.05)
        btn("🎲  Generate Random Map", self._generate_map)
        btn("🗑   Clear All Walls",    self._clear_walls)

        # ── ALGORITHM ──
        section("ALGORITHM")
        radios(self.var_algo, ["A*", "GBFS"])

        # ── HEURISTIC ──
        section("HEURISTIC")
        radios(self.var_heur, ["Manhattan", "Euclidean"])

        # ── EDIT MODE ──
        section("EDIT MODE  (click on grid)")
        self.edit_btns = {}
        for mode, label in [("wall",  "Place / Remove Wall"),
                             ("start", "Move Start Node"),
                             ("goal",  "Move Goal Node")]:
            b = tk.Button(p, text=label,
                          command=lambda m=mode: self._set_edit_mode(m),
                          bg=C["dark"], fg=C["text"],
                          relief=tk.FLAT, font=("Consolas",8),
                          cursor="hand2", pady=4,
                          activebackground=C["accent"])
            b.pack(fill=tk.X, pady=1)
            self.edit_btns[mode] = b
        self._set_edit_mode("wall")

        # ── DYNAMIC MODE ──
        section("DYNAMIC OBSTACLES")
        tk.Checkbutton(p, text="Enable Dynamic Mode",
                       variable=self.var_dynamic,
                       bg=C["panel"], fg=C["text"],
                       selectcolor=C["bg"],
                       activebackground=C["panel"],
                       font=("Consolas",9)).pack(anchor="w")
        slider("Spawn Probability", self.var_spawn, 0.01, 0.15, 0.01)

        # Padding at bottom
        tk.Frame(p, bg=C["panel"], height=20).pack()

    # ─────────────────────────────────────────
    #  GRID DRAW
    # ─────────────────────────────────────────
    def _draw_grid(self):
        cs = self.CS
        self.canvas.delete("all")
        self.canvas.configure(
            scrollregion=(0, 0, self.COLS*cs, self.ROWS*cs))
        self.cell_rect = {}
        for r in range(self.ROWS):
            for c in range(self.COLS):
                x1, y1 = c*cs, r*cs
                rect = self.canvas.create_rectangle(
                    x1, y1, x1+cs-1, y1+cs-1,
                    fill=self._cell_color(r,c),
                    outline=C["grid_line"], width=1)
                self.cell_rect[(r,c)] = rect

    def _cell_color(self, r, c):
        if (r,c) == self.agent_pos:   return C["agent"]
        if (r,c) == self.start:       return C["start"]
        if (r,c) == self.goal:        return C["goal"]
        if self.grid[r][c] == 1:      return C["wall"]
        if (r,c) in self.path_set:    return C["path"]
        if (r,c) in self.visited_set: return C["visited"]
        return C["empty"]

    def _repaint(self, r, c):
        item = self.cell_rect.get((r,c))
        if item:
            self.canvas.itemconfig(item, fill=self._cell_color(r,c))

    def _repaint_all(self):
        for r in range(self.ROWS):
            for c in range(self.COLS):
                self._repaint(r,c)

    # ─────────────────────────────────────────
    #  MOUSE
    # ─────────────────────────────────────────
    def _px_to_cell(self, px, py):
        cs = self.CS
        c = int(self.canvas.canvasx(px) // cs)
        r = int(self.canvas.canvasy(py) // cs)
        if 0 <= r < self.ROWS and 0 <= c < self.COLS:
            return r, c
        return None

    def _on_click(self, event):
        if self.running: return
        cell = self._px_to_cell(event.x, event.y)
        if not cell: return
        r, c = cell
        if self.edit_mode == "start":
            old = self.start; self.start = (r,c); self.grid[r][c]=0
            self._repaint(*old); self._repaint(r,c)
        elif self.edit_mode == "goal":
            old = self.goal; self.goal = (r,c); self.grid[r][c]=0
            self._repaint(*old); self._repaint(r,c)
        else:
            if (r,c) not in (self.start, self.goal):
                self.grid[r][c] = 1 - self.grid[r][c]
                self._repaint(r,c)

    def _on_drag(self, event):
        if self.running or self.edit_mode != "wall": return
        cell = self._px_to_cell(event.x, event.y)
        if cell and cell not in (self.start, self.goal):
            r,c = cell
            if self.grid[r][c] == 0:
                self.grid[r][c] = 1
                self._repaint(r,c)

    def _set_edit_mode(self, mode):
        self.edit_mode = mode
        for m, b in self.edit_btns.items():
            b.configure(bg=C["accent"] if m==mode else C["dark"])

    # ─────────────────────────────────────────
    #  MAP GEN
    # ─────────────────────────────────────────
    def _generate_map(self):
        self._stop()
        self.ROWS  = self.var_rows.get()
        self.COLS  = self.var_cols.get()
        self.start = (0,0)
        self.goal  = (self.ROWS-1, self.COLS-1)
        d = self.var_density.get()
        self.grid  = [[0]*self.COLS for _ in range(self.ROWS)]
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if (r,c) not in (self.start, self.goal):
                    if random.random() < d:
                        self.grid[r][c] = 1
        self._reset_state()
        self._draw_grid()
        self.var_status.set("  Map generated!  Press  ▶ START SEARCH  to begin.")

    def _clear_walls(self):
        self._stop()
        self.grid = [[0]*self.COLS for _ in range(self.ROWS)]
        self._reset_state()
        self._draw_grid()
        self.var_status.set("  Walls cleared.")

    # ─────────────────────────────────────────
    #  SEARCH + ANIMATION
    # ─────────────────────────────────────────
    def _start_search(self):
        if self.running: return
        self._reset_state(); self._repaint_all()
        alg = self.var_algo.get()
        hfn = manhattan if self.var_heur.get()=="Manhattan" else euclidean
        self.var_status.set(f"  Running {alg} ({self.var_heur.get()})...")
        self.root.update_idletasks()

        t0 = time.perf_counter()
        path, visited, cost = run_search(self.grid, self.start, self.goal, alg, hfn)
        ms = (time.perf_counter()-t0)*1000

        self.var_time.set(f"{ms:.1f}")
        self.var_nodes.set(str(len(visited)))

        if path is None:
            self.var_cost.set("No path")
            self.var_status.set("  No path found! Remove some walls and try again.")
            self.visited_set = set(visited); self._repaint_all()
            return

        self.var_cost.set(str(cost))
        self.var_status.set(f"  Path found! Length={cost}. Animating...")
        self.running = True
        self._anim_visited(visited, path, 0)

    def _anim_visited(self, visited, path, idx):
        if not self.running: return
        if idx < len(visited):
            self.visited_set.add(visited[idx])
            self._repaint(*visited[idx])
            self.var_nodes.set(str(idx+1))
            self.root.after(max(1, self.var_speed.get()//5),
                            lambda: self._anim_visited(visited, path, idx+1))
        else:
            self.path_set = set(path); self._repaint_all()
            self.var_status.set("  Path shown. Agent moving...")
            self.agent_pos = self.start; self.agent_step = 0
            self.var_replans.set("0")
            self.root.after(300, lambda: self._anim_agent(path))

    def _anim_agent(self, path):
        if not self.running: return
        if self.agent_step >= len(path):
            self.var_status.set("  Goal reached!"); self.running=False; return

        if self.var_dynamic.get() and self.agent_step > 0:
            if self._spawn_obstacles():
                remaining = path[self.agent_step:]
                if any(self.grid[r][c]==1 for r,c in remaining):
                    self.var_status.set("  Path blocked! Re-planning...")
                    self.var_replans.set(str(int(self.var_replans.get())+1))
                    alg = self.var_algo.get()
                    hfn = manhattan if self.var_heur.get()=="Manhattan" else euclidean
                    t0  = time.perf_counter()
                    new_path, new_vis, new_cost = run_search(
                        self.grid, self.agent_pos, self.goal, alg, hfn)
                    ms = (time.perf_counter()-t0)*1000
                    self.var_time.set(f"{ms:.1f}")
                    if new_path is None:
                        self.var_status.set("  Agent trapped! No path exists.")
                        self.running=False; return
                    self.visited_set.update(new_vis)
                    self.path_set = set(new_path)
                    self.var_cost.set(str(new_cost))
                    self.var_nodes.set(str(int(self.var_nodes.get())+len(new_vis)))
                    self._repaint_all()
                    path = new_path; self.agent_step = 0
                    self.var_status.set("  New path found. Continuing...")

        old = self.agent_pos
        self.agent_pos  = path[self.agent_step]
        self.agent_step += 1
        if old: self._repaint(*old)
        self._repaint(*self.agent_pos)

        if self.agent_pos == self.goal:
            self.var_status.set("  Goal reached!"); self.running=False; return

        self.root.after(self.var_speed.get(), lambda: self._anim_agent(path))

    def _spawn_obstacles(self):
        spawned = False
        prob = self.var_spawn.get() * 0.06
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if (self.grid[r][c]==0 and
                        (r,c) not in (self.start, self.goal, self.agent_pos) and
                        random.random() < prob):
                    self.grid[r][c]=1; self._repaint(r,c); spawned=True
        return spawned

    # ─────────────────────────────────────────
    #  UTILITIES
    # ─────────────────────────────────────────
    def _stop(self):
        self.running=False
        self.var_status.set("  Stopped.")

    def _reset_state(self):
        self.running=False; self.visited_set=set(); self.path_set=set()
        self.agent_pos=None; self.agent_step=0
        self.var_nodes.set("0"); self.var_cost.set("--")
        self.var_time.set("--");  self.var_replans.set("0")

    def _reset_view(self):
        self._reset_state(); self._repaint_all()
        self.var_status.set("  View reset.")


# ──────────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
