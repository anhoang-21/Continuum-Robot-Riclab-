"""
==============================================================================
command_panel.py - Small command window for the language-conditioned demo (Tkinter)
==============================================================================
White window with the RICLAB logo (assets/riclab_logo.png), the pipes on the table, an
instruction box, quick commands and the live state (phase, VLM answer, chosen pipe,
result, history). It runs in the simulation's main thread: the simulation loop calls
poll() every step (Tk is updated, the next command is returned if the user sent one).
Used by play_language.py --ui.
==============================================================================
"""

from __future__ import annotations

import os
import queue
import tkinter as tk
from tkinter import ttk

from multi_pipe import PALETTE

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO = os.path.join(TASK_DIR, "assets", "riclab_logo.png")
WHITE, INK, GREY, LINE = "#ffffff", "#1d2430", "#6b7686", "#e3e7ee"
ACCENT, RED, GREEN = "#e3342f", "#e3342f", "#1f9d55"
FONT = "Segoe UI"


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(255 * c) for c in rgb)


def _dpi_aware():
    """Sharp text / logo on scaled Windows displays (Tk is not DPI aware by default)."""
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:                                      # noqa: BLE001  (not Windows / already set)
        pass


class CommandPanel:
    def __init__(self, model_name="Qwen3-VL-2B"):
        _dpi_aware()
        self.root = tk.Tk()
        scale = self.root.winfo_fpixels("1i") / 96.0           # 1.0 at 100 %, 1.5 at 150 % display scaling
        self.s = scale
        self.root.title("RICLAB · Continuum robot · language commands")
        self.root.configure(bg=WHITE)
        self.root.geometry(f"{int(560 * scale)}x{int(780 * scale)}")
        self.root.minsize(int(520 * scale), int(700 * scale))
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.alive = True
        self.commands: queue.Queue[str] = queue.Queue()
        self._logo = None

        pad = {"padx": 22}
        self._header(pad)
        tk.Label(self.root, text="Language-guided insertion", font=(FONT, 16, "bold"), fg=INK, bg=WHITE,
                 anchor="w").pack(fill="x", pady=(6, 0), **pad)
        tk.Label(self.root, text=f"Instruction → {model_name} (local) → image-based policy", font=(FONT, 10),
                 fg=GREY, bg=WHITE, anchor="w").pack(fill="x", **pad)
        self._rule()

        # pipes on the table
        tk.Label(self.root, text="PIPES ON THE TABLE  (left → right in the overview camera)", font=(FONT, 9, "bold"),
                 fg=GREY, bg=WHITE, anchor="w").pack(fill="x", **pad)
        self.chips = tk.Frame(self.root, bg=WHITE)
        self.chips.pack(fill="x", pady=(6, 0), **pad)
        self._rule()

        # instruction
        tk.Label(self.root, text="INSTRUCTION", font=(FONT, 9, "bold"), fg=GREY, bg=WHITE, anchor="w").pack(
            fill="x", **pad)
        row = tk.Frame(self.root, bg=WHITE)
        row.pack(fill="x", pady=(6, 0), **pad)
        self.entry = tk.Entry(row, font=(FONT, 12), relief="solid", bd=1, highlightthickness=1,
                              highlightcolor=ACCENT, highlightbackground=LINE)
        self.entry.pack(side="left", fill="x", expand=True, ipady=6)
        self.entry.bind("<Return>", lambda e: self._send(self.entry.get()))
        self.run_btn = tk.Button(row, text="Run", font=(FONT, 11, "bold"), fg=WHITE, bg=ACCENT, activebackground="#c42a26",
                                 activeforeground=WHITE, relief="flat", padx=18, command=lambda: self._send(self.entry.get()))
        self.run_btn.pack(side="left", padx=(8, 0), ipady=3)
        self.quick = tk.Frame(self.root, bg=WHITE)
        self.quick.pack(fill="x", pady=(8, 0), **pad)
        fixed = tk.Frame(self.root, bg=WHITE)
        fixed.pack(fill="x", pady=(6, 0), **pad)
        for text, cmd in (("Leftmost", "Go through the leftmost pipe."), ("Rightmost", "Go through the rightmost pipe."),
                          ("New scene", "__new__")):
            self._button(fixed, text, cmd, fg=INK, bg="#f1f3f7").pack(side="left", padx=(0, 6))
        self._rule()

        # state
        tk.Label(self.root, text="STATE", font=(FONT, 9, "bold"), fg=GREY, bg=WHITE, anchor="w").pack(fill="x", **pad)
        grid = tk.Frame(self.root, bg=WHITE)
        grid.pack(fill="x", pady=(4, 0), **pad)
        self.vars = {}
        for r, key in enumerate(("status", "phase", "VLM box (0-1000)", "chosen pipe", "result")):
            tk.Label(grid, text=key, font=(FONT, 10), fg=GREY, bg=WHITE, anchor="w", width=16).grid(row=r, column=0,
                                                                                                    sticky="w")
            var = tk.StringVar(value="-")
            lab = tk.Label(grid, textvariable=var, font=(FONT, 10, "bold"), fg=INK, bg=WHITE, anchor="w")
            lab.grid(row=r, column=1, sticky="w")
            self.vars[key] = (var, lab)
        self._rule()

        tk.Label(self.root, text="HISTORY", font=(FONT, 9, "bold"), fg=GREY, bg=WHITE, anchor="w").pack(fill="x", **pad)
        self.history = tk.Listbox(self.root, font=(FONT, 10), relief="flat", bg="#fafbfc", fg=INK, height=7,
                                  highlightthickness=1, highlightbackground=LINE, activestyle="none")
        self.history.pack(fill="both", expand=True, pady=(4, 16), **pad)
        self.set_state(status="starting the simulation…")
        self.entry.focus_set()
        self.poll()

    # ------------------------------------------------------------------
    def _header(self, pad):
        frame = tk.Frame(self.root, bg=WHITE)
        frame.pack(fill="x", pady=(16, 0), **pad)
        if os.path.isfile(LOGO):
            try:
                from PIL import Image, ImageTk

                img = Image.open(LOGO).convert("RGBA")
                w = int(380 * self.s)
                img = img.resize((w, int(img.height * w / img.width)), Image.LANCZOS)
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                self._logo = ImageTk.PhotoImage(Image.alpha_composite(bg, img).convert("RGB"))
                tk.Label(frame, image=self._logo, bg=WHITE).pack(anchor="w")
                return
            except Exception as exc:                        # noqa: BLE001  (fall back to the text header)
                print(f"logo not shown: {exc}")
        tk.Label(frame, text="RIC", font=(FONT, 30, "bold"), fg=WHITE, bg=RED, padx=6).pack(side="left")
        tk.Label(frame, text="LAB", font=(FONT, 30, "bold"), fg="#111111", bg=WHITE, padx=4).pack(side="left")
        tk.Label(frame, text="Robotics and Intelligent\nControl Laboratory", font=(FONT, 9), fg=GREY, bg=WHITE,
                 justify="left").pack(side="left", padx=10)

    def _rule(self):
        tk.Frame(self.root, bg=LINE, height=1).pack(fill="x", padx=22, pady=12)

    def _button(self, parent, text, cmd, fg=WHITE, bg=INK):
        return tk.Button(parent, text=text, font=(FONT, 10), fg=fg, bg=bg, relief="flat", padx=10, pady=3,
                         activebackground=LINE, command=lambda: self._send(cmd))

    def _send(self, text):
        text = (text or "").strip()
        if text:
            self.commands.put(text)
            if text != "__new__":
                self.entry.delete(0, "end")

    def _close(self):
        self.alive = False
        self.root.destroy()

    # ------------------------------------------------------------------
    def poll(self):
        """Process window events; return the next command (None if there is none)."""
        if not self.alive:
            return None
        try:
            self.root.update()
        except tk.TclError:
            self.alive = False
            return None
        try:
            return self.commands.get_nowait()
        except queue.Empty:
            return None

    def set_pipes(self, colors, example=""):
        """colors: tube colours from left to right; one chip + one quick command per pipe."""
        for w in list(self.chips.children.values()) + list(self.quick.children.values()):
            w.destroy()
        for c in colors:
            chip = tk.Frame(self.chips, bg=WHITE)
            chip.pack(side="left", padx=(0, 14))
            d = int(26 * self.s)
            sw = tk.Canvas(chip, width=d, height=d, bg=WHITE, highlightthickness=0)
            sw.create_oval(3, 3, d - 3, d - 3, fill=_hex(PALETTE[c]), outline="#f2c12e", width=max(3, int(3 * self.s)))
            sw.pack(side="left")
            tk.Label(chip, text=c, font=(FONT, 11), fg=INK, bg=WHITE).pack(side="left", padx=(4, 0))
        for c in colors:
            self._button(self.quick, f"{c} pipe", f"Go through the {c} pipe.", fg=WHITE if c not in ("white",) else INK,
                         bg=_hex(PALETTE[c])).pack(side="left", padx=(0, 6))
        self.entry.delete(0, "end")
        if example:
            self.entry.insert(0, example)
            self.entry.select_range(0, "end")

    def set_state(self, **kw):
        for key, value in kw.items():
            key = {"box": "VLM box (0-1000)", "chosen": "chosen pipe"}.get(key, key)
            var, lab = self.vars[key]
            var.set(value)
            if key == "result":
                lab.configure(fg=GREEN if value.startswith("THROUGH") else (RED if value not in ("-", "") else INK))
        self.poll()

    def set_busy(self, busy):
        state = "disabled" if busy else "normal"
        self.run_btn.configure(state=state)
        self.entry.configure(state=state)
        for w in self.quick.children.values():
            w.configure(state=state)

    def add_history(self, line):
        self.history.insert(0, line)


if __name__ == "__main__":                               # look at the window without the simulation
    p = CommandPanel()
    p.set_pipes(["blue", "red", "white"], "Go through the red pipe.")
    p.set_state(status="ready", phase="-", result="THROUGH THE PIPE!")
    p.add_history("Go through the red pipe.  →  red  →  SUCCESS (3.1 s)")
    while p.alive:
        cmd = p.poll()
        if cmd:
            p.add_history(cmd)
