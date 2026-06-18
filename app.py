"""Streamlit entrypoint for the thesis debate evaluation app.

The original Streamlit implementation lives in ``src.streamlit_app`` so this
root file can stay small while preserving existing commands and API imports.
"""

from src.streamlit_app import *  # noqa: F401,F403
from src.streamlit_app import main


if __name__ == "__main__":
    main()
