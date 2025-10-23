"""
Hugging Face Spaces entry point for the Book Recommender app.
This file is the main entry point that Hugging Face Spaces will use to launch the app.
"""

from gradio_dashboard import dashboard

if __name__ == "__main__":
    # Launch the dashboard with settings optimized for Hugging Face Spaces
    dashboard.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
