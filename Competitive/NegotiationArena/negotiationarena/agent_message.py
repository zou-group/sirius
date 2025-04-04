from abc import ABC, abstractmethod
from negotiationarena.utils import from_name_and_tag_to_message


class AgentMessage:
    """
    Structured format for agent messages.
    Should define what agents can see of each other messages.

    Note that for public messages, order in the dict is important.
    """

    def __init__(self):
        self.public = {}
        self.secret = {}

    def add_public(self, key, message):
        """

        :param key:
        :param message:
        :return:
        """

        self.public[key] = message

    def add_secret(self, key, message):
        self.secret[key] = message

    def message_to_other_player(self):
        response = []
        for key, value in self.public.items():
            response.append(from_name_and_tag_to_message(key, value))

        r = "\n".join(response)

        return r
