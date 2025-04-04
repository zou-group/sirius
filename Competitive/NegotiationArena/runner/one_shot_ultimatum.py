from dotenv import load_dotenv
from negotiationarena.agents.chatgpt import ChatGPTAgent
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.goal import UltimatumGoal
from games.ultimatum.one_shot_ultimatum.game import UltimatumOneShotGame
from negotiationarena.constants import *

load_dotenv(".env")

if __name__ == "__main__":
    num_accept = 0
    num_iters = 20
    for i in range(num_iters):
        a1 = ChatGPTAgent(
            agent_name=AGENT_ONE,
            model="gpt-4-1106-preview",
        )
        a2 = ChatGPTAgent(
            agent_name=AGENT_TWO,
            model="gpt-4-1106-preview",
        )

        c = UltimatumOneShotGame(
            iterations=2,
            players=[a1, a2],
            resources_support_set=Resources({"x": 0}),
            player_goals=[
                UltimatumGoal(),
                UltimatumGoal(),
            ],
            player_initial_resources=[
                Resources({"x": 100}),
                Resources({"x": 0}),
            ],
            player_social_behaviour=[
                "",
                "",  # You are completely rational",  # "Forget your past knowledge. You are a completely irrational being.",
            ],
            player_roles=[
                f"You are {AGENT_ONE} start by making a proposal.",
                f"You are {AGENT_TWO}, start by responding to a trade.",
            ],
            log_dir="./.logs/ultimatum",
        )
        c.run()
        if c.game_state[-1]["summary"]["final_response"] == "ACCEPTED":
            num_accept += 1

        print("RUNNING ACCEPTANCE RATE: {}".format(num_accept / (i + 1)))

    print(num_accept / num_iters)
