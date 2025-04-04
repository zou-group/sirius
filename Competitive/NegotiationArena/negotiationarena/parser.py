from abc import ABC, abstractmethod
from negotiationarena.game_objects.trade import Trade
from negotiationarena.utils import *
from negotiationarena.constants import *


class GameParser(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def instantiate_prompt(self, **kwargs):
        """
        Returns the inital ratbench prompt
        """
        pass

    @abstractmethod
    def parse(self, response):
        """
        Parses the ratbench response
        """
        pass

    @classmethod
    def from_dict(cls, state):
        state = copy.deepcopy(state)
        class_name = state.pop("class")
        subclasses = cls.get_all_subclasses()
        constructor = (
            cls
            if class_name == cls.__name__
            else next(
                (sub for sub in subclasses if sub.__name__ == class_name), None
            )
        )
        if constructor:
            obj = constructor(**state)
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


class ExchangeGameDefaultParser(GameParser, ABC):
    """
    This class provides a high level abstractions for all the games that are based on exchanges.
    """

    def __init__(self):
        super().__init__()

    def parse_proposed_trade(self, s):
        """
        :param s:
        :return:
        """
        trade = {}

        c = s.strip().replace("\n", " ")
        for player in c.split("|"):
            player_name = player.split("Player")[1].split("Gives")[0].strip()
            resources = player.split("Gives")[1].strip()
            # NOTE: We are casting the resources to int.

            print("resources", resources)

            parse_resources = {
                i.split(":")[0].strip(): int(i.split(":")[1].strip().split(' ')[0])
                for i in resources.split(",")
            }


            trade[player_name] = parse_resources

        return trade

    def parse_trade(self, response, interest_tag):
        contents = get_tag_contents(response, interest_tag).lstrip().rstrip()
        if contents == REFUSING_OR_WAIT_TAG:
            return contents
        return Trade(self.parse_proposed_trade(contents))
