class EraserState:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EraserState, cls).__new__(cls)
            cls._instance.history = {}
        return cls._instance