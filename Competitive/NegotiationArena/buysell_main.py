import sys
# from dotenv import load_dotenv

from negotiationarena.agents.chatgpt import ChatGPTAgent
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.goal import BuyerGoal, SellerGoal
from negotiationarena.game_objects.valuation import Valuation
from negotiationarena.constants import *
import traceback
from games.buy_sell_game.game import BuySellGame


# load_dotenv(".env")


if __name__ == "__main__":

    
    model=["gpt-3.5-turbo","gpt-3.5-turbo"]

   
    for i in range(60):
        try:
            a1 = ChatGPTAgent(agent_name=AGENT_ONE, model=model[0])
            a2 = ChatGPTAgent(agent_name=AGENT_TWO, model=model[1])

            c = BuySellGame(
                players=[a1, a2],
                iterations=8,
                player_goals=[
                    SellerGoal(cost_of_production=Valuation({"X": 30})),
                    BuyerGoal(willingness_to_pay=Valuation({"X": 70})),
                ],
                player_starting_resources=[
                    Resources({"X": 1}),
                    Resources({MONEY_TOKEN: 100}),
                ],
                player_conversation_roles=[
                    f"You are {AGENT_ONE}.",
                    f"You are {AGENT_TWO}.",
                ],
                player_social_behaviour=[
                    "",
                    ""
                    # "You are very kind and generous. Be friendly and helpful with the other player, they are your dearest friend.",
                ],
                log_dir=f"./logs/buysell/3070/{model[0]}_{model[1]}/",
            )

            c.run()
        except Exception as e:
            exception_type = type(e).__name__
            exception_message = str(e)
            stack_trace = traceback.format_exc()

            # Print or use the information as needed
            print(f"Exception Type: {exception_type}")
            print(f"Exception Message: {exception_message}")
            print(f"Stack Trace:\n{stack_trace}")
