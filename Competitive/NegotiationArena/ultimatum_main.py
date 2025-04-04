# from dotenv import load_dotenv
from negotiationarena.agents.chatgpt import ChatGPTAgent
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.goal import UltimatumGoal
from games.ultimatum.game import MultiTurnUltimatumGame
from negotiationarena.constants import *

# load_dotenv(".env")

if __name__ == "__main__":

    model=["ft:gpt-3.5-turbo-0125:merty::AXe1QEYz",'gpt-4o-mini-2024-07-18']

    print(model)

    for i in range(60):
    
        a1 = ChatGPTAgent(
            agent_name=AGENT_ONE,
            model= model[0],
        )
        a2 = ChatGPTAgent(
            agent_name=AGENT_TWO,
            model= model[1],
        )

        c = MultiTurnUltimatumGame(
            players=[a1, a2],
            iterations=8,
            resources_support_set=Resources({"Dollars": 0}),
            player_goals=[
                UltimatumGoal(),
                UltimatumGoal(),
            ],
            player_initial_resources=[
                Resources({"Dollars": 100}),
                Resources({"Dollars": 0}),
            ],
            player_social_behaviour=["", ""],
            player_roles=[
                f"You are {AGENT_ONE}.",
                f"You are {AGENT_TWO}.",
            ],
            log_dir=f"./logs/ultimatum_multi_period_100/{f'{model[0]}_{model[1]}'}/",
        )
        try:
            c.run()
        except Exception as e:
            print(e)
            continue

