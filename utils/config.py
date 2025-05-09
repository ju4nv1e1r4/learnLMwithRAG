import os

import dotenv

dotenv.load_dotenv()


class LoadEnvVars:
    def __init__(self, key: str):
        self.key = key

    def get_key(self) -> str:
        key = os.getenv(self.key)
        if not key:
            raise ValueError(
                """
                Your environment variable was not found.
                Please double check this step and try again.
                """
            )

        return key
