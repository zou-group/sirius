import os
import time
import json
import copy
import inspect
from pathlib import Path
from typing import List
from abc import ABC, abstractmethod, abstractproperty
from negotiationarena.constants import MESSAGE_TAG
from negotiationarena.logging import GameEncoder
from negotiationarena.parser import GameParser
from negotiationarena.agents.agents import Agent
from negotiationarena.utils import get_next_filename


class Game(ABC):
    """
    Base class for games.

    A ratbench should take in :
    (1) players: players of the ratbench as a list of agents
    (2) game_interface: interface specifiying ratbench rules (as prompt) and communication interface (as a parser)


    """

    def __init__(self, players: List[List], log_dir=".logs", log_path=None):
        self.run_epoch_time_ms = str(round(time.time() * 1000))

        self.players = players
        self.game_state = None

        # logging
        self.log_dir = os.path.abspath(log_dir)
        self.log_path = (
            os.path.join(self.log_dir, self.run_epoch_time_ms)
            if log_path is None
            else log_path
        )

    @abstractmethod
    def set_game_state(self, game_state_dict):
        pass

    def to_dict(self):
        return {
            "class": self.__class__.__name__,
            **copy.deepcopy(self.__dict__),
        }

    def log_state(self):
        """
        logging full ratbench state
        """
        Path(self.log_path).mkdir(parents=True, exist_ok=True)
        # log full state
        with open(os.path.join(self.log_path, "game_state.json"), "w") as f:
            json.dump(self.to_dict(), f, cls=GameEncoder, indent=2)

        self.log_human_readable_state()

    @abstractmethod
    def log_human_readable_state(self):
        pass

    @classmethod
    def from_dict(cls, game_state_dict):
        game_state_dict = copy.deepcopy(game_state_dict)
        class_name = game_state_dict.pop("class")
        subclasses = cls.get_all_subclasses()
        constructor = (
            cls
            if class_name == cls.__name__
            else next(
                (sub for sub in subclasses if sub.__name__ == class_name), None
            )
        )
        if constructor:
            # intialize game interface object
            game_state_dict["game_interface"] = GameParser.from_dict(
                game_state_dict["game_interface"]
            )
            # initialize players
            game_state_dict["players"] = [
                Agent.from_dict(player)
                for player in game_state_dict["players"]
            ]

            # the constructor actually corrupts the player conversations because of "init_player", so we deep copy a clean version first
            _game_state_dict = copy.deepcopy(game_state_dict)
            obj = constructor(**game_state_dict)

            obj.set_game_state(_game_state_dict)
            return obj
        else:
            raise ValueError(f"Unknown subclass: {class_name}")

    @classmethod
    def get_all_subclasses(cls):
        subclasses_set = set()
        # Recursively get subclasses of subclasses
        for subclass in cls.__subclasses__():
            subclasses_set.add(subclass)
            subclasses_set.update(subclass.get_all_subclasses())

        return list(subclasses_set)
