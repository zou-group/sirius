import sys

sys.path.append(".")
# from dotenv import load_dotenv
import inspect
from negotiationarena.agents import *
from negotiationarena.agents.agent_behaviours import (
    SelfCheckingAgent,
    ReasoningAgent,
)
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.goal import ResourceGoal
from games.trading_game.game import TradingGame
from games.trading_game.interface import TradingGameDefaultParser
from negotiationarena.constants import *

# load_dotenv(".env")


if __name__ == "__main__":
   





   
    
    
    
   
    model=["gpt-3.5-turbo","gpt-3.5-turbo"]

    for i in range(60):
        print(i)
        a1 = ChatGPTAgent(
            model= model[0],
            # model="ft:gpt-3.5-turbo-0125:merty::ASJCtzbh",
            # model= "ft:gpt-4o-mini-2024-07-18:merty::ASJB4mWP",
            agent_name=AGENT_ONE,
        )
        a2 = ChatGPTAgent(
            # model= model,
            model=model[1],
            # model= "ft:gpt-4o-mini-2024-07-18:merty::ASJB4mWP",
            agent_name=AGENT_TWO,
        )

        c = TradingGame(
            players=[a1, a2],
            iterations=8,
            resources_support_set=Resources({"X": 0, "Y": 0}),
            player_goals=[
                ResourceGoal({"X": 25, "Y": 25}),
                ResourceGoal({"X": 25, "Y": 25}),
            ],
            player_initial_resources=[
               
                Resources({"X": 35, "Y": 15}),
                Resources({"X": 15, "Y": 35}),
            ],
            player_social_behaviour=["", ""],
            player_roles=[
                f"You are {AGENT_ONE}, start by making a proposal.",
                f"You are {AGENT_TWO}, start by responding to a trade.",
            ],
            log_dir=f"./logs/trading/35_15/{f'{model[0]}_{model[1]}'}/",
        )

        try:
            c.run()
        except Exception as e:
            print(e)
            continue
