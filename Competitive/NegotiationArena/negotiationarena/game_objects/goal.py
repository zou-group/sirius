import json
import logging
from dataclasses import dataclass
from collections import defaultdict
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.valuation import Valuation
from abc import abstractmethod
from negotiationarena.constants import *


class Goal:
    def __init__(self):
        pass

    @abstractmethod
    def goal_reached(self):
        pass

    @abstractmethod
    def json(self):
        pass


@dataclass
class ResourceGoal(Goal, Resources):
    def goal_reached(self, resources: Resources):
        return all(
            resources.resource_dict.get(k, 0) >= v
            for k, v in self.resource_dict.items()
        )

    def json(self):
        return {"_type": "resource_goal", "_value": self.resource_dict}


class MaximisationGoal(Goal):
    goal = "Acquire as many resources as possible"

    def __init__(self, inital_resources: Resources):
        self.inital_resources = inital_resources

    def __str__(self):
        return self.goal

    def to_prompt(self):
        return self.goal

    def goal_reached(self, final_resources: Resources):
        return final_resources - self.inital_resources

    def json(self):
        return {"_type": "maximisation_goal", "_value": self.inital_resources}


class UltimatumGoal(Goal):
    goal = ""

    def __str__(self):
        return self.goal

    def __repr__(self):
        return self.goal

    def to_prompt(self):
        return self.goal

    def goal_reached(
        self, inital_resource: Resources, final_resources: Resources
    ):
        return final_resources - inital_resource

    def json(self):
        return {"_type": "ultimatum_goal", "_value": self.goal}


class BuyerGoal(Goal):
    def __init__(self, willingness_to_pay: Valuation):
        super().__init__()
        self.willingness_to_pay = willingness_to_pay
        self.goal = f"Buy resources with <{MONEY_TOKEN}>. You are willing to pay at most {willingness_to_pay} for the resources."

    def __repr__(self):
        return self.goal

    def get_valuation(self):
        return self.willingness_to_pay

    def __str__(self):
        return self.goal

    def to_prompt(self):
        return self.goal

    def json(self):
        return {
            "_type": "buyer_goal",
            "_value": self.willingness_to_pay.json(),
        }


class SellerGoal(Goal):
    def __init__(self, cost_of_production: Valuation):
        super().__init__()
        self.cost_of_production = cost_of_production
        self.goal = f"Sell resources for <{MONEY_TOKEN}>. It costed {self.cost_of_production} to produce the resources"

    def __repr__(self):
        return self.goal

    def get_valuation(self):
        return self.cost_of_production

    def __str__(self):
        return self.goal

    def to_prompt(self):
        return self.goal

    def json(self):
        return {
            "_type": "seller_goal",
            "_value": self.cost_of_production.json(),
        }
