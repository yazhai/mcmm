class BaselineRunner:
    def __init__(self) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def process_result(self) -> None:
        raise NotImplementedError

