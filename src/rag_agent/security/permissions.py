from typing import Dict, List
from . import config  # будет позже

from ..tools.base import RiskLevel


class PermissionManager:
    """Центральный менеджер разрешений инструментов"""
    
    def __init__(self):
        self.enabled_high_risk = False  # главный рубильник
        self.require_confirmation = {"medium", "high"}
    
    def can_use_tool(self, risk_level: RiskLevel, user_confirmed: bool = False) -> bool:
        if risk_level == "safe":
            return True
        if risk_level == "medium":
            return True  # пока разрешаем, позже можно ужесточить
        if risk_level == "high":
            return self.enabled_high_risk and user_confirmed
        return False

    def get_required_confirmation(self, risk_level: RiskLevel) -> bool:
        return risk_level in self.require_confirmation


permission_manager = PermissionManager()