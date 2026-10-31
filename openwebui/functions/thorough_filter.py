"""
title: Thorough check
author: rangan
version: 1.0.0
description: Takes time to think the question through, then checks each statement against the sources. Answers take longer.
"""
# A toggle button in the chat input bar. When switched on, Open WebUI lists this filter's id in the request metadata
# ("filter_ids"); the query pipe reads it and runs the RQ3 verification loop (split → check → revise, second revision
# round with thinking). Off by default: the model's first answer, streamed as it is written. The filter itself changes
# nothing in the request.
from pydantic import BaseModel

# magnifier with a check mark (currentColor, so it follows the light/dark theme)
ICON = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' "
        "stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
        "<circle cx='10.5' cy='10.5' r='6.5'/><path d='M15.5 15.5 21 21'/><path d='m7.8 10.6 1.9 1.9 3.6-3.8'/></svg>")


class Filter:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True          # shown as a switchable button in the chat input bar
        self.icon = ICON

    async def inlet(self, body: dict, __user__: dict | None = None) -> dict:
        return body
