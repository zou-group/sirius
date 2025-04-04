from negotiationarena.alternating_game import AlternatingGameEndsOnTag
from negotiationarena.game_objects.resource import Resources
from negotiationarena.constants import (
    REASONING_TAG,
    PLAYER_ANSWER_TAG,
    MESSAGE_TAG,
    PROPOSAL_COUNT_TAG,
    PROPOSED_TRADE_TAG,
    RESOURCES_TAG,
    GOALS_TAG,
    ACCEPTING_TAG,
)

from negotiationarena.utils import extract_multiple_tags
from games.buy_sell_game.prompt import buy_sell_prompt
from negotiationarena.parser import ExchangeGameDefaultParser
from negotiationarena.agent_message import AgentMessage


class BuySellGameDefaultParser(ExchangeGameDefaultParser):
    def __init__(self):
        super().__init__()

    def instantiate_prompt(
        self,
        resources_available_in_game,
        starting_initial_resources,
        player_goal,
        maximum_number_of_proposals,
        player_social_behaviour,
    ):
        return buy_sell_prompt(
            resources_available_in_game,
            starting_initial_resources,
            player_goal,
            maximum_number_of_proposals,
            player_social_behaviour,
        )

    def parse(self, response):
        """
        Parse the response from the player. We are going to extract multiple lines of text from
        different tags.

        For example, we extrac <REASONING_TAG> reasoning from the model. </REASONING_TAG>

        :param response:
        :return:
        """
        (
            resources,
            goal,
            reasoning,
            answer,
            message,
            proposal_count,
            trade,
        ) = extract_multiple_tags(
            response,
            [
                RESOURCES_TAG,
                GOALS_TAG,
                REASONING_TAG,
                PLAYER_ANSWER_TAG,
                MESSAGE_TAG,
                PROPOSAL_COUNT_TAG,
                PROPOSED_TRADE_TAG,
            ],
        )
        resources = Resources.from_string(resources)
        trade = self.parse_trade(response, PROPOSED_TRADE_TAG)

        # create the message, we are going to split between public messages and secret messages.

        ms = AgentMessage()

        for tag, content in zip(
            [MESSAGE_TAG, PLAYER_ANSWER_TAG, PROPOSED_TRADE_TAG],
            [message, answer, trade],
        ):
            ms.add_public(tag, content)

        for tag, content in zip(
            [RESOURCES_TAG, GOALS_TAG, REASONING_TAG, PROPOSAL_COUNT_TAG],
            [resources, goal, reasoning, proposal_count],
        ):
            ms.add_secret(tag, content)

        return ms


class BuySellGame(AlternatingGameEndsOnTag):
    def __init__(
        self,
        player_goals,
        player_starting_resources,
        player_social_behaviour,
        player_conversation_roles,
        game_interface=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        # we compute the set of resources available in game.
        # this is done just to "inform" the agents of the resources available in the game.
        resources_support_set = {}

        if len(player_starting_resources[0].resource_dict) > 1:
            raise ValueError(
                "Only one resource is supported due to rendering in the prompt. Update the prompt to support more resources"
            )

        for resource in player_starting_resources[0].resource_dict:
            resources_support_set[resource] = 0

        resources_support_set = Resources(resources_support_set)

        self.game_state = [
            {
                "current_iteration": "START",
                "turn": "None",
                "settings": dict(
                    resources_support_set=resources_support_set,
                    player_goals=player_goals,
                    player_initial_resources=player_starting_resources,
                    player_social_behaviour=player_social_behaviour,
                    player_roles=player_conversation_roles,
                    player_valuation=[g.get_valuation() for g in player_goals],
                ),
            }
        ]

        # we are going to set all the parameter we might need later
        self.resources_support_set = resources_support_set
        self.player_goals = player_goals
        self.player_starting_resources = player_starting_resources
        self.player_social_behaviour = player_social_behaviour
        self.player_conversation_roles = player_conversation_roles

        self.game_interface = (
            BuySellGameDefaultParser()
            if game_interface is None
            else game_interface
        )

        # init players
        self.init_players()

    def init_players(self):
        settings = self.game_state[0]["settings"]
        for idx, player in enumerate(self.players):
            game_prompt = self.game_interface.instantiate_prompt(
                resources_available_in_game=settings[
                    "resources_support_set"
                ].only_keys(),
                starting_initial_resources=settings[
                    "player_initial_resources"
                ][idx],
                player_goal=settings["player_goals"][idx],
                maximum_number_of_proposals=self.iterations // 2 - 1,
                player_social_behaviour=settings["player_social_behaviour"][
                    idx
                ],
            )

            player.init_agent(game_prompt, settings["player_roles"][idx])

    def after_game_ends(self):
        """
        This method is called after the game ends. For example
        the agent has accepted.

        This method can be much simpler if you don't want to compute the outcome of the game.

        :return:
        """
        end_state = self.game_state[-1]

        # if there is only one iteration, we are going to set the game state to END
        if int(end_state["current_iteration"]) <= 1:
            datum = dict(
                current_iteration="END",
                turn="None",
            )
            self.game_state.append(datum)
        else:
            # we compute the outcome of the game

            player_response = end_state["player_public_info_dict"][
                PLAYER_ANSWER_TAG
            ]
            initial_resources = self.game_state[0]["settings"][
                "player_initial_resources"
            ]
            player_valuation = self.game_state[0]["settings"][
                "player_valuation"
            ]
            player_goals = self.game_state[0]["settings"]["player_goals"]
            proposed_trade = self.game_state[-2]["player_public_info_dict"][
                PROPOSED_TRADE_TAG
            ]

            if player_response == ACCEPTING_TAG:
                # get proposed trade
                final_resources = [
                    proposed_trade.execute_trade(res, idx)
                    for idx, res in enumerate(initial_resources)
                ]
            else:
                final_resources = initial_resources

            outcome = [
                v.value(final - initial)
                for v, initial, final in zip(
                    player_valuation, initial_resources, final_resources
                )
            ]

            datum = dict(
                current_iteration="END",
                turn="None",
                summary=dict(
                    player_goals=player_goals,
                    player_initial_resources=initial_resources,
                    proposed_trade=proposed_trade,
                    player_valuation=player_valuation,
                    final_response=player_response,  # ACCEPT / REJECT / NONE
                    final_resources=final_resources,
                    player_outcome=outcome,
                ),
            )

            self.game_state.append(datum)
