import tkinter as tk
from tkinter import ttk
from gui import FireMonitoringApp


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    FireMonitoringApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
