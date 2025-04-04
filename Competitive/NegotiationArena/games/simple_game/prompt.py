from negotiationarena.constants import *


def simple_game_prompt(initial_resources, player_social_behaviour):
    prompt = f"""You are playing a game in which {AGENT_ONE} gives all resources to {AGENT_TWO}. {AGENT_TWO} has nothing to give.
You can only make on single trade.

RULES:

```
1. You must always respond with:

A) Propose a trade with (you can only trade in integer amounts, not decimals):
<{PLAYER_ANSWER_TAG}> PROPOSAL </{PLAYER_ANSWER_TAG}>
<{PROPOSED_TRADE_TAG}> {AGENT_ONE} Gives item 1: amount, ...| {AGENT_TWO} Gives item1: 0 </{PROPOSED_TRADE_TAG}>

B) You can only accept the trade by saying:
<{PLAYER_ANSWER_TAG}> {ACCEPTING_TAG} </{PLAYER_ANSWER_TAG}>
<{PROPOSED_TRADE_TAG}> NONE </{PROPOSED_TRADE_TAG}>

2. You can send a message to the other agent by saying:

<{MESSAGE_TAG}> your message here </{MESSAGE_TAG}>

Here is what you have access to:
```
<{RESOURCES_TAG}> {initial_resources} </{RESOURCES_TAG}>
```

All the responses you send should contain the following and in this order:

```
<{PLAYER_ANSWER_TAG}> [add here] </{PLAYER_ANSWER_TAG}>
<{PROPOSED_TRADE_TAG}> [add here] </{PROPOSED_TRADE_TAG}>
<{MESSAGE_TAG}> [add here] </{MESSAGE_TAG}>
```

Please be sure to include all.

{player_social_behaviour}
"""

    return prompt
