"""
Representation of how much agent values resources as a unit of `MONEY_TOKEN`
"""

from dataclasses import dataclass
from negotiationarena.constants import *
from negotiationarena.game_objects.resource import Resources


@dataclass
class Valuation:
    valuation_dict: dict = None
    # e.g. {X:2, Y:4,} where 2 => 2M, 4 => 4M

    def value(self, resources: Resources):
        val_of_resources = 0
        for k, v in resources.resource_dict.items():
            if k != MONEY_TOKEN:
                val_of_resources += self.valuation_dict[k] * v
            else:
                val_of_resources += v
        return val_of_resources

    def to_prompt(self):
        res = [f"{k}: {v}" for k, v in self.valuation_dict.items()]
        return ", ".join(res)

    def __str__(self):
        res = [
            f"{k}: {v} {MONEY_TOKEN}" for k, v in self.valuation_dict.items()
        ]
        return ", ".join(res)

    def __repr__(self):
        res = [f"'{k}': {v}" for k, v in self.valuation_dict.items()]
        return "{" + ", ".join(res) + "}"

    def json(self):
        return {"_type": "valuation", "_value": self.valuation_dict}
