from pathlib import Path

from dotenv import load_dotenv


def load_environment_variables(file_path=".env"):
    """
    Load environment variables from a .env file.

    Args:
        file_path (str): The path to the .env file.
    """
    dotenv_path = Path(file_path)
    load_dotenv(dotenv_path=dotenv_path)
