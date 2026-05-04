"""Shared helpers for analysis routers."""

from __future__ import annotations


def _categorize_themes(themes: list[str]) -> str:
    """Map GDELT themes to a simple category for color coding."""
    theme_str = " ".join(themes).upper()
    if any(
        k in theme_str
        for k in [
            "MILITARY",
            "WAR",
            "ARMED",
            "TERROR",
            "CONFLICT",
            "KILL",
            "WOUND",
            "ARREST",
            "CRIME",
            "ATTACK",
            "REBELLION",
            "INSURGENT",
            "DRONE",
            "WEAPON",
            "BOMB",
            "SHOOT",
            "HOSTAGE",
            "SIEGE",
        ]
    ):
        return "conflict"
    if any(
        k in theme_str
        for k in [
            "ECON",
            "MARKET",
            "TRADE",
            "FINANCE",
            "BUSINESS",
            "TAX_FNCACT",
            "STOCK",
            "INVEST",
            "BANKRUPT",
            "INFLATION",
            "GDP",
            "UNEMPLOY",
            "CRYPTO",
            "TARIFF",
            "DEBT",
            "REVENUE",
        ]
    ):
        return "economy"
    if any(
        k in theme_str
        for k in [
            "ELECT",
            "POLITIC",
            "GOVERN",
            "DIPLOMAT",
            "LEGISLAT",
            "VOTE",
            "PARLIAMENT",
            "CONGRESS",
            "PRESIDENT",
            "MINISTER",
            "SANCTION",
            "TREATY",
            "SUMMIT",
            "CAUCUS",
            "CAMPAIGN",
            "PARTY",
        ]
    ):
        return "politics"
    if any(
        k in theme_str
        for k in [
            "HEALTH",
            "DISEASE",
            "MEDICAL",
            "PANDEMIC",
            "HOSPITAL",
            "VACCINE",
            "DRUG",
            "VIRUS",
            "OUTBREAK",
            "WHO_",
            "SURGEON",
            "CANCER",
        ]
    ):
        return "health"
    if any(
        k in theme_str
        for k in [
            "ENV_",
            "ENVIRON",
            "CLIMATE",
            "DISASTER",
            "QUAKE",
            "FLOOD",
            "HURRICANE",
            "WILDFIRE",
            "DROUGHT",
            "EMISSION",
            "CARBON",
            "STORM",
            "TSUNAMI",
            "TORNADO",
            "VOLCANO",
        ]
    ):
        return "environment"
    if any(
        k in theme_str
        for k in [
            "TECH",
            "CYBER",
            "AI_",
            "DIGITAL",
            "SCIENCE",
            "ROBOT",
            "SPACE",
            "INTERNET",
            "SOFTWARE",
            "HACK",
            "DATA_",
            "COMPUTING",
        ]
    ):
        return "technology"
    if any(
        k in theme_str
        for k in [
            "HUMAN_RIGHTS",
            "PROTEST",
            "REFUGEE",
            "MIGRATION",
            "EDUCATION",
            "WOMEN",
            "CHILD",
            "POVERTY",
            "DISCRIMINATION",
            "RIGHTS",
            "RELIGION",
            "CULTURE",
            "SPORT",
            "ENTERTAINMENT",
        ]
    ):
        return "society"
    return "general"
