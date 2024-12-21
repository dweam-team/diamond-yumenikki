"""
Credits: some parts are taken and modified from the file `config.py` from https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning/
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pygame
import torch

from .keymap import CSGO_KEYMAP


@dataclass
class CSGOAction:
	keys: List[int]

	@property
	def key_names(self) -> List[str]:
		return [pygame.key.name(key) for key in self.keys]




def print_csgo_action(action: CSGOAction) -> Tuple[str]:
	action_names = [CSGO_KEYMAP[k] for k in action.keys] if len(action.keys) > 0 else []
	keys = " + ".join(action_names)
	return f"{keys}"


def encode_csgo_action(csgo_action: CSGOAction, device: torch.device) -> torch.Tensor:

	action_vector = np.zeros((5,))
	action_index = None
	for key in csgo_action.key_names:
		if key == 'left':
			action_index = 0
		elif key == 'up':
			action_index = 1
		elif key == 'down':
			action_index = 2
		elif key == 'right':
			action_index = 3
		elif key == "z":
			action_index = 4

	if action_index is not None:
		action_vector[action_index] = 1.

	return torch.tensor(
		action_vector,
		device=device,
		dtype=torch.float32,
	)


def decode_csgo_action(y_preds: torch.Tensor) -> CSGOAction:
	action_vector = y_preds.squeeze()
	action_index = torch.argmax(action_vector).item()
	keys_pressed = []
	if action_vector[action_index] > 0.5:
		if action_index == 0:
			keys_pressed.append('left')
		elif action_index == 1:
			keys_pressed.append('up')
		elif action_index == 2:
			keys_pressed.append('down')
		elif action_index == 3:
			keys_pressed.append('right')
		elif action_index == 4:
			keys_pressed.append('z')

	keys_pressed = [pygame.key.key_code(x) for x in keys_pressed]
	return CSGOAction(keys_pressed)

