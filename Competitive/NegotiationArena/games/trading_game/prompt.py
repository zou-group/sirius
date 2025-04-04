from negotiationarena.constants import *


# Player 1 will suggest an initial trade:

# <{PLAYER_ANSWER_TAG}> NONE </{PLAYER_ANSWER_TAG}>
# <{PROPOSED_TRADE_TAG}> Player 1 Gives item1: amount, item2: amount, Player 2 Gives item1: amount, item2: amount, ... </{PROPOSED_TRADE_TAG}>


def trading_prompt(
    agent_name,
    resources_in_game,
    initial_resources,
    goal,
    number_of_proposals,
    social_behaviour,
):
    prompt = f"""You are playing a strategic game of trading resources with another player whose resources you have no knowledge about. 

RULES:
```

1. You can either:

A) Accept the trade by saying:
<{PLAYER_ANSWER_TAG}> {ACCEPTING_TAG} </{PLAYER_ANSWER_TAG}>
<{PROPOSED_TRADE_TAG}> {REFUSING_OR_WAIT_TAG} </{PROPOSED_TRADE_TAG}>

B) Reject and propose a new trade (you can only trade integer amounts, not decimals):
<{PLAYER_ANSWER_TAG}> {REFUSING_OR_WAIT_TAG} </{PLAYER_ANSWER_TAG}>
<{PROPOSED_TRADE_TAG}> {AGENT_ONE} Gives item1: amount, item2: amount, ... | {AGENT_TWO} Gives item1: amount, item2: amount, ... </{PROPOSED_TRADE_TAG}>

C) Don't accept or propose anything and wait for a new offer:\n
<{PLAYER_ANSWER_TAG}> {REFUSING_OR_WAIT_TAG} </{PLAYER_ANSWER_TAG}>
<{PROPOSED_TRADE_TAG}> {REFUSING_OR_WAIT_TAG} </{PROPOSED_TRADE_TAG}>\n

Note: the game will end if one of the players accepts. This means that you have to be careful about both accepting and proposing a trade.

2. You are allowed at most {number_of_proposals} proposals of your own to complete the game, after which you can only {ACCEPTING_TAG} or {REFUSING_OR_WAIT_TAG}.
DO NOT propose a new trade after {number_of_proposals} proposals. Your limit for proposals is {number_of_proposals}.

3. You can reason step by step by using the following format:
<{REASONING_TAG}> [add reasoning] </{REASONING_TAG}>
Add as much text as you want. This information will not be sent to the other player. 
It is just for you to keep track of your reasoning.

4. At each turn send messages to each other by using the following format:
<{MESSAGE_TAG}> [add message] </{MESSAGE_TAG}>
You can decide if you want disclose your resources and goals in the message.
```

Here is what you have access to:
```
Resources available in the game: {resources_in_game}
<{RESOURCES_TAG}> {initial_resources} </{RESOURCES_TAG}>
<{GOALS_TAG}> {goal} </{GOALS_TAG}>
<{MY_NAME_TAG}> {agent_name} </{MY_NAME_TAG}>
```

All the responses you send should contain the following and in this order:

```
<{MY_NAME_TAG}> [add here] </{MY_NAME_TAG}>
<{RESOURCES_TAG}> [add here] </{RESOURCES_TAG}>
<{GOALS_TAG}> [add here] </{GOALS_TAG}>
<{REASONING_TAG}> [add here] </{REASONING_TAG}>
<{PLAYER_ANSWER_TAG}> [add here] </{PLAYER_ANSWER_TAG}>
<{MESSAGE_TAG}> [add here] </{MESSAGE_TAG}
<{PROPOSED_TRADE_TAG}> [add here] </{PROPOSED_TRADE_TAG}>
```

Please be sure to include all.

More resources in general are always better.
This is the last round of trading. There are no future rounds after this one.

{social_behaviour}
"""

    return prompt
