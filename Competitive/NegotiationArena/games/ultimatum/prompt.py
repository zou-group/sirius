from negotiationarena.constants import *


def ultimatum_prompt(
    player_1_initial_resources,
    resources_in_game,
    initial_resources,
    iterations,
    number_of_proposals,
    social_behaviour,
):
    agent_one_proposals = (
        number_of_proposals + 1 if iterations % 2 else number_of_proposals
    )
    agent_two_proposals = number_of_proposals
    proposal_limit = (
        number_of_proposals + 1 if iterations % 2 else number_of_proposals
    )

    prompt = f""""You are playing a game involving a split of resources. 
{AGENT_ONE} starts with {player_1_initial_resources}, {AGENT_TWO} has none to trade.

RULES:
```

1. {AGENT_ONE} has {agent_one_proposals} moves while {AGENT_TWO} has {agent_two_proposals} moves.

2. You must either:

    A) Accept the current proposal by saying:
        <{PLAYER_ANSWER_TAG}> {ACCEPTING_TAG} </{PLAYER_ANSWER_TAG}>
        <{PROPOSED_TRADE_TAG}> NONE </{PROPOSED_TRADE_TAG}>

    B) Reject the current proposal, whereby both players lose all their resources and the game ends immediately:
        <{PLAYER_ANSWER_TAG}> {REJECTION_TAG} </{PLAYER_ANSWER_TAG}>
        <{PROPOSED_TRADE_TAG}> NONE </{PROPOSED_TRADE_TAG}>

    C) Propose a new trade:
        <{PLAYER_ANSWER_TAG}> NONE </{PLAYER_ANSWER_TAG}>
        <{PROPOSED_TRADE_TAG}> {AGENT_ONE} Gives item1: amount | {AGENT_TWO} Gives item1: 0 </{PROPOSED_TRADE_TAG}>

    {AGENT_ONE if iterations % 2 else AGENT_TWO} cannot do (C) on {TURN_OR_MOVE_TAG} {proposal_limit}/{proposal_limit} and MUST ONLY answer with {ACCEPTING_TAG} (A) or {REJECTION_TAG} (B) but {AGENT_TWO if iterations % 2 else AGENT_ONE} is not affected by this condition.        
    
    Note: the game ends immediately if you {ACCEPTING_TAG} or {REJECTION_TAG}.
    
3. If either player REJECTS, both players lose all their resources and the ends immediately.

4. There are no future games.

5. You can reason step by step on why you are A) proposing, B) rejecting and C) accepting a trade with:
<{REASONING_TAG}> [add reasoning] </{REASONING_TAG}> add as much text as you want
This information will not be sent to the other player. It is just for you to keep track of your reasoning.

6. At each turn send messages to each other by using the following format:
<{MESSAGE_TAG}>your message here</{MESSAGE_TAG}>
You can disclose anything in the message.
```

Here is what you have access to:
```
Resources available in the game: {resources_in_game}
<{RESOURCES_TAG}> {initial_resources} </{RESOURCES_TAG}>
```

All the responses you send should contain the following and in this order:

```
<{MY_NAME_TAG}> [add here] </{MY_NAME_TAG}>
<{TURN_OR_MOVE_TAG}> [add here] / [add here]  </{TURN_OR_MOVE_TAG}> 
<{RESOURCES_TAG}> [add here] </{RESOURCES_TAG}>
<{REASONING_TAG}> [add here] </{REASONING_TAG}>
<{PLAYER_ANSWER_TAG}> [add here] </{PLAYER_ANSWER_TAG}>
<{MESSAGE_TAG}> [add here] </{MESSAGE_TAG}
<{PROPOSED_TRADE_TAG}> [add here] </{PROPOSED_TRADE_TAG}>
```

Please be sure to include all.

{social_behaviour}
"""

    return prompt
