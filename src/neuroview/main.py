import customtkinter as ctk
from neuroview.gui.main_window import MainWindow

def main() -> None:
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    app = MainWindow()
    app.mainloop()

if __name__ == "__main__":
    main()
