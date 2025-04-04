from negotiationarena.alternating_game import AlternatingGameEndsOnTag
from negotiationarena.parser import ExchangeGameDefaultParser
from negotiationarena.constants import *
from negotiationarena.utils import *
from negotiationarena.agent_message import AgentMessage
from games.simple_game.prompt import simple_game_prompt
from typing import List


class SimpleGameDefaultParser(ExchangeGameDefaultParser):
    def instantiate_prompt(self, initial_resources, social_behavior):
        return simple_game_prompt(initial_resources, social_behavior)

    def parse(self, response):
        ms = AgentMessage()

        answer = get_tag_contents(response, PLAYER_ANSWER_TAG)
        message = get_tag_contents(response, MESSAGE_TAG)
        trade = self.parse_trade(response, PROPOSED_TRADE_TAG)

        ms.add_public(MESSAGE_TAG, message)
        ms.add_public(PLAYER_ANSWER_TAG, answer)
        ms.add_public(PROPOSED_TRADE_TAG, trade)

        return ms


class SimpleGame(AlternatingGameEndsOnTag):
    def __init__(
        self,
        resources_support_set,
        player_initial_resources,
        player_roles,
        player_social_behaviour,
        **kwargs,
    ):
        self.game_interface = SimpleGameDefaultParser()

        super().__init__(end_tag=ACCEPTING_TAG, **kwargs)

        #################
        # Game State    #
        #################

        self.game_state: List[dict] = [
            {
                "current_iteration": "START",
                "turn": "None",
                "settings": dict(
                    resources_support_set=resources_support_set,
                    player_initial_resources=player_initial_resources,
                    player_roles=player_roles,
                    player_social_behaviour=player_social_behaviour,
                ),
            }
        ]

        self.resources_support_set = resources_support_set
        self.player_initial_resources = player_initial_resources
        self.player_roles = player_roles
        self.player_social_behaviour = player_social_behaviour

        # init players
        self.init_players()

    def init_players(self):
        settings = self.game_state[0]["settings"]

        #################
        # Agent Setup   #
        #################

        for idx, player in enumerate(self.players):
            game_prompt = self.game_interface.instantiate_prompt(
                initial_resources=settings["player_initial_resources"][idx],
                social_behavior=settings["player_social_behaviour"][idx],
            )

            player.init_agent(game_prompt, settings["player_roles"][idx])

    def game_over(self):
        """
        check if the game is over
        """

        state = self.game_state[-1]
        if state:
            response = state["player_public_info_dict"].get(
                PLAYER_ANSWER_TAG, REFUSING_OR_WAIT_TAG
            )

            # Game is over if one of the players accepts
            if response == ACCEPTING_TAG:
                return True

        return False

    def after_game_ends(self):
        datum = dict(current_iteration="END", turn="None", summary=dict())

        self.game_state.append(datum)
