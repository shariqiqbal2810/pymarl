REGISTRY = {}

from .basic_controller import BasicMAC
REGISTRY["basic_mac"] = BasicMAC

from .icql_controller import ICQLMAC
REGISTRY["icql_mac"] = ICQLMAC