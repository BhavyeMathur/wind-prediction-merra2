from .abstract import AtmosphericVariable4D


class Temperature(AtmosphericVariable4D, name="temperature", unit="K",
                  cmap=["#d7e4fc", "#5fb1d4", "#4e9bc8", "#466ae1", "#6b1966",
                        "#952c5e", "#d12d3e", "#fa7532", "#f5d25f"]):
    def __init__(self, celsius: bool = False):
        super().__init__()
        if celsius:
            self._unit = "°C"
        else:
            self._unit = "K"
        self._celsius = celsius

    def _getitem_post(self, ds):
        if self._celsius:
            return ds - 273.15  # Kelvin to Celsius
        return ds

    @staticmethod
    def get_vlims(level: int):
        if level == 1000:
            return -40, 40
        elif level == 150:
            return -70, -40
        raise ValueError("Unknown level for value limits")


__all__ = ["Temperature"]
