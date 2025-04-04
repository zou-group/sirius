from abc import ABC
from negotiationarena.agents.agents import Agent
from negotiationarena.constants import REASONING_TAG


class SelfCheckingAgent(Agent, ABC):
    def think(self):
        # do one step of thinking
        super().think()
        # print("reflecting!")
        # prompt agent to check proposal
        self.update_conversation_tracking(
            "user", "Double check your proposal."
        )
        # think again
        return super().think()


class ReasoningAgent(Agent, ABC):
    def init_agent(self, system_prompt, role):
        system_prompt = (
            system_prompt
            + f"\nReason succinctly step by step about your response within <{REASONING_TAG}> [add reasoning] </{REASONING_TAG}> tags."
        )
        super().init_agent(system_prompt, role)
