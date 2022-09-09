class PlaceholderNotAssignedError(Exception):
    def __repr__(self):
        return "Placeholder Error"


class NoPathFoundError(Exception):
    def __repr__(self) -> str:
        return "No Path Found Error"