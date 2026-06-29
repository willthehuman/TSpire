"""TSpire — play Slay the Spire from a remote terminal, without a mod.

The host reads the game screen (template matching + OCR) to reconstruct game state and
acts as a virtual Xbox360 controller to send input. The client renders the state in a
terminal and relays the human's commands. See the approved plan for the full design.
"""

__version__ = "0.1.0"
