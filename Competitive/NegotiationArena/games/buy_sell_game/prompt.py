from negotiationarena.constants import *


def buy_sell_prompt(
    resources_available_in_game,
    starting_initial_resources,
    player_goal,
    maximum_number_of_proposals,
    player_social_behaviour,
):
    prompt = f"""You are playing game where you are buying or selling an object. There is only one object for sale/purcahse.

{AGENT_ONE} is going to sell one object. {AGENT_TWO} gives {MONEY_TOKEN} to buy resources.

RULES:

```
1. You must always respond with:

    A) Propose a trade with (you can only trade in integer amounts, not decimals):
    <{PLAYER_ANSWER_TAG}> PROPOSAL </{PLAYER_ANSWER_TAG}>
    <{PROPOSED_TRADE_TAG}> {AGENT_ONE} Gives {resources_available_in_game}: amount, ...| {AGENT_TWO} Gives {MONEY_TOKEN}: amount </{PROPOSED_TRADE_TAG}>

    B) Accept the trade by saying:
    <{PLAYER_ANSWER_TAG}> {ACCEPTING_TAG} </{PLAYER_ANSWER_TAG}>
    <{PROPOSED_TRADE_TAG}> NONE </{PROPOSED_TRADE_TAG}>

    C) Reject and end the game:
    <{PLAYER_ANSWER_TAG}> {REJECTION_TAG} </{PLAYER_ANSWER_TAG}>
    <{PROPOSED_TRADE_TAG}> NONE </{PROPOSED_TRADE_TAG}>

    Note: The game will end if one of the players {ACCEPTING_TAG} OR {REJECTION_TAG}. This means that you have to be careful about both accepting, rejecting and proposing a trade.

2. You are allowed at most {maximum_number_of_proposals} proposals of your own to complete the game, after which you can only reply with {ACCEPTING_TAG} or {REJECTION_TAG}.
DO NOT propose a new trade after {maximum_number_of_proposals} proposals. Your limit for proposals is {maximum_number_of_proposals}.

3. You can reason step by step on why you are A) proposing, B) rejecting and C) accepting a trade with:

<{REASONING_TAG}> [add reasoning] </{REASONING_TAG}> add as much text as you want

This information will not be sent to the other player. It is just for you to keep track of your reasoning.

4. At each turn send messages to each other by using the following format:

<{MESSAGE_TAG}>your message here</{MESSAGE_TAG}>

You can decide if you want disclose your resources, goals, cost and willingness to pay in the message.
```

Here is what you have access to:
```
Object that is being bought/sold: {resources_available_in_game}
<{RESOURCES_TAG}> {starting_initial_resources} </{RESOURCES_TAG}>
<{GOALS_TAG}> {player_goal} </{GOALS_TAG}>,
```

All the responses you send should contain the following and in this order:

```
<{PROPOSAL_COUNT_TAG}> [add here (inclusive of current)] </{PROPOSAL_COUNT_TAG}>
<{RESOURCES_TAG}> [add here] </{RESOURCES_TAG}>
<{GOALS_TAG}> [add here] </{GOALS_TAG}>
<{REASONING_TAG}> [add here] </{REASONING_TAG}>
<{PLAYER_ANSWER_TAG}> [add here] </{PLAYER_ANSWER_TAG}>
<{PROPOSED_TRADE_TAG}> [add here] </{PROPOSED_TRADE_TAG}>
<{MESSAGE_TAG}> [add here] </{MESSAGE_TAG}
```

Please be sure to include all.

{player_social_behaviour}
"""

    return prompt
