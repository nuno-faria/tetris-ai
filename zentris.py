#!/usr/bin/python3

import random
import numpy as np
import itertools
from datetime import datetime
import re

# Ajouter metrique 'nb voisins bombe' ou somme des voisins


# Zentris game class
class Zentris:
	'''Zentris game class'''

	# BOARD
	BOARD_WIDTH = 6
	BOARD_HEIGHT = 6
	BLOCK_TRUEMAX_HEIGHT = 7
	
	CONF = 'cbBOA2D'
	BLOCK_MAX_HEIGHT = 4
	BOARD_INIT_CELLS = 5
	BOARD_BOMB_PROBABILITY = 40
	NB_EXPLORED_MOVES = 3

	# piece      0 1 2 3 4 5 6 7
	# weights = [1,3,2,1,2,2,1,1]
	# CUM_WEIGHTS_HEIGHT = [
	# # height 1   2   3   4   5   6   7
	#         [0,  0,  0, 25, 50, 13, 12], # piece 0
	#         [0,  0, 25, 50, 25,  0,  0], # piece 1
	#         [0, 25, 50, 25,  0,  0,  0], # piece 2
	#         [0, 25, 50, 25,  0,  0,  0], # piece 3
	#         [0, 25, 50, 25,  0,  0,  0], # piece 4
	#         [0, 25, 50, 25,  0,  0,  0], # piece 5
	#         [0, 33, 33, 33,  0,  0,  0], # piece 6
	#         [0, 33, 33, 33,  0,  0,  0], # piece 7
	# ]
	#
	# mais random.choices prefere cum_weights
	CUM_WEIGHTS_ID = [1, 4, 6, 7, 9, 11, 12, 13]
	CUM_WEIGHTS_HEIGHT = [
	# height 1   2   3   4    5    6    7
	        [0,  0,  0,  25,  75,  88, 100], # piece 0
	        [0,  0, 25,  75, 100, 100, 100], # piece 1
	        [0, 25, 75, 100, 100, 100, 100], # piece 2
	        [0, 25, 75, 100, 100, 100, 100], # piece 3
	        [0, 25, 75, 100, 100, 100, 100], # piece 4
	        [0, 25, 75, 100, 100, 100, 100], # piece 5
	        [0, 33, 67, 100, 100, 100, 100], # piece 6
	        [0, 33, 67, 100, 100, 100, 100], # piece 7
	]

	TETROMINOS = [
		################ ----- #############
		########## 1 to 4 long #############
		{ # I-1
			0: [(0,0)],
			90: [(0,0)],
			180: [(0,0)],
			270: [(0,0)],
		},
		{ # I-2
			0: [(0,0), (1,0)],
			90: [(0,0), (0,1)],
			180: [(1,0), (0,0)],
			270: [(0,1), (0,0)],
		},
		{ # I-3
			0: [(0,0), (1,0), (2,0)],
			90: [(0,0), (0,1), (0,2)],
			180: [(2,0), (1,0), (0,0)],
			270: [(0,2), (0,1), (0,0)],
		},
		{ # I-4
			0: [(0,0), (1,0), (2,0), (3,0)],
			90: [(0,0), (0,1), (0,2), (0,3)],
			180: [(3,0), (2,0), (1,0), (0,0)],
			270: [(0,3), (0,2), (0,1), (0,0)],
		},

		############# with angle ##########
		########## 1 to 2 long ############

		{ # short L
			0:   [(0,1), (0,2), (1,2)],
			90:  [(0,1), (1,1), (1,0)],
			180: [(1,1), (1,0), (0,0)],
			270: [(1,0), (0,0), (0,1)],
		},
		{ # short T
			0: [(1,0), (0,1), (1,1), (2,1)],
			90: [(0,1), (1,2), (1,1), (1,0)],
			180: [(1,1), (2,0), (1,0), (0,0)],
			270: [(1,1), (0,0), (0,1), (0,2)],
		},
		{ # L
			0: [(1,0), (1,1), (1,2), (2,2)],
			90: [(0,1), (1,1), (2,1), (2,0)],
			180: [(1,2), (1,1), (1,0), (0,0)],
			270: [(2,1), (1,1), (0,1), (0,2)],
		},
		{ # J
			0: [(1,0), (1,1), (1,2), (0,2)],
			90: [(0,1), (1,1), (2,1), (2,2)],
			180: [(1,2), (1,1), (1,0), (2,0)],
			270: [(2,1), (1,1), (0,1), (0,0)],
		},
	]

	def __init__(self):
		self.pd = PreComputedData()
		self.myzentris = MyZentris()
		self.myzentris.randomFill(nbNonZero=Zentris.BOARD_INIT_CELLS, maxHeight=2)
		self.pieces = []
		self.nextStates = []
		self.next_props = {}

	def reset(self):
		self.myzentris = MyZentris()
		self.myzentris.randomFill(nbNonZero=Zentris.BOARD_INIT_CELLS, maxHeight=2)
		self.pieces = [ MyPiece(self.pd) for _ in range(3) ]
		self.nextStates = []
		self.next_props = {}
		return self.myzentris._computeProps(Zentris.CONF, self.pd) 

	def get_next_states(self):
		self.nextStates = exploreBFS(self.myzentris, self.pieces)
		self.next_props = { nextState[0][-1]: nextState[0][-1]._computeProps(Zentris.CONF, self.pd) for nextState in self.nextStates }
		return self.next_props

	def play(self, action, render):
		if render:
			# recherche des actions
			result = [ x for x in self.nextStates if np.array_equal(x[0][-1], action) ]
			result = result[0]
			# affichage
			for p in result[2]:
				p.print()
				print()
			_printDescriptions(result[1])
			print()
			# affichage du board
			# action.print(referenceBoard=self.myzentris)
			for i in range(1, 4):
				result[0][i].print(referenceBoard=result[0][i-1])
				print()
			print('scoreDelta =', action.scoreDelta, 'cleared lines / bombed =', action.cleared, action.bombed)
			# print('props = ', action._computeProps(Zentris.CONF, self.pd))
			print('-'*60)

		self.myzentris = action
		scoreDelta = self.myzentris.scoreDelta
		self.myzentris.play()
		# Reset stuff
		self.nextStates = []
		self.next_props = {}
		self.pieces = [ MyPiece(self.pd) for _ in range(3) ]
		return scoreDelta

	def get_state_info(self):
		return self.myzentris._propsInfo(Zentris.CONF)

	def get_game_score(self):
		return self.myzentris.score

# Copie sur gym openAI
class ZentrisEnv():
	# env.action_space.n      					--> 1
	# env.observation_space.shape 				--> nb_props
	def __init__(self):
		pass

	def step(self, action):
		"""

		Parameters
		----------
		action :

		Returns
		-------
		ob, reward, episode_over, info : tuple
			ob (object) :
				an environment-specific object representing your observation of
				the environment.
			reward (float) :
				amount of reward achieved by the previous action. The scale
				varies between environments, but the goal is always to increase
				your total reward.
			episode_over (bool) :
				whether it's time to reset the environment again. Most (but not
				all) tasks are divided up into well-defined episodes, and done
				being True indicates the episode has terminated. (For example,
				perhaps the pole tipped too far, or you lost your last life.)
			info (dict) :
				 diagnostic information useful for debugging. It can sometimes
				 be useful for learning (for example, it might contain the raw
				 probabilities behind the environment's last state change).
				 However, official evaluations of your agent are not allowed to
				 use this for learning.
		"""
		self._take_action(action)
		self.status = self.env.step()
		reward = self._get_reward()
		ob = self.env.getState()
		episode_over = self.status != hfo_py.IN_GAME
		return ob, reward, episode_over, {}					# observation, reward, episode_over, info 

	def reset(self):
		pass

#########################
import numpy as np
import copy
import colorama

def _autoCrop(array):
	array = array[:, np.any(array != 0, axis=0)]
	array = array[np.any(array != 0, axis=1), :]
	return array

def _printValue(value, isBomb=False, isNew=None, forceStyle=None):
	char = str(value) if value else '.'
	fore_attr  = colorama.Fore.YELLOW if isBomb else colorama.Fore.RESET
	back_attr  = colorama.Back.RESET
	style_attr = colorama.Style.NORMAL if isNew is None else (colorama.Style.BRIGHT if isNew else colorama.Style.DIM)
	if forceStyle:
		print(forceStyle                         + char, end='')
	else:
		print(fore_attr + back_attr + style_attr + char, end='')
	print(colorama.Style.RESET_ALL + ' ', end='')

def _printDescriptions(descriptions):
	colors = [colorama.Back.RED, colorama.Back.GREEN, colorama.Back.BLUE]
	def _lookFor(coords, listOfList):
		result = [ (i, x[2]) for i in range(len(listOfList)) for x in listOfList[i] if x[:2]==coords ]
		return result
	for y in range(6):
		for x in range(6):
			res = _lookFor((y,x), descriptions)
			if len(res) == 0:
				value = '.'
				style = colorama.Back.RESET + colorama.Style.DIM
			elif len(res) == 1:
				value = str(res[0][1])
				style = colors[ res[0][0] ]
			else:
				value = str(res[-1][1])
				style = colorama.Back.MAGENTA
			print(style + value, end='')
			print(colorama.Style.RESET_ALL + ' ', end='')
		print()

def _print_results(results):
	colors = [colorama.Back.RED, colorama.Back.GREEN, colorama.Back.BLUE]
	for i in range(0, 3):
		results[0][i].print(referenceBoard=results[0][i-1] if i>0 else None, addPiece=results[1][i], stylePiece=colors[i])
	results[0][3].print()

class SlowPiece:
	def __init__(self, shape=None):
		if shape is not None:
			self.unrotated = shape
			self.rotated   = self.unrotated.copy()
			self.rotation  = 0
		else:
			self.unrotated = np.zeros([1, 1], dtype=np.int8)
			self.rotated   = self.unrotated.copy()
			self.rotation  = 0
			self.random()
		
	def random(self):
		print('FUNCTION NOT MAINTAINED - SlowPiece.random()')
		pass
		# piece_id = random.randrange(len(Zentris.TETROMINOS))
		# description = Zentris.TETROMINOS[piece_id][0]
		# print('dbg=', piece_id, description)
		# # Convert to numpy
		# sizeX = max([x[0] for x in description])+1
		# sizeY = max([x[1] for x in description])+1
		# self.unrotated = np.zeros([sizeY, sizeX], dtype=np.int8)
		# # All pieces have either max_height or min_height as size
		# max_height = random.randint(1, Zentris.BLOCK_MAX_HEIGHT)
		# min_height = max(1, max_height-1)
		# for xy in description:
		# 	self.unrotated[xy[1], xy[0]] = random.randint(min_height, max_height)
		# self.unrotated = _autoCrop(self.unrotated)
		
		# self.rotated = self.unrotated.copy()

	def print(self, unrotated=False):
		array = self.unrotated if unrotated else self.rotated
		for y in range(array.shape[0]):
			for x in range(array.shape[1]):
				_printValue(array[y][x])
			print()

	def rotate(self, angle, fromUnrotated=False):
		# 0 = 0e, 1 = 90e, 2 = 180e, 3 = 270e
		self.rotated = np.rot90(self.unrotated if fromUnrotated else self.rotated, k=angle)
		if fromUnrotated:
			self.rotation = angle
		else:
			self.rotation += angle
			self.rotation %= 4

	def insidePositionsYX(self, boardSize):
		pieceSizeX = self.rotated.shape[1]
		boardSizeX = boardSize[1]
		maxPositionX = boardSizeX - pieceSizeX

		pieceSizeY = self.rotated.shape[0]
		boardSizeY = boardSize[0]
		maxPositionY = boardSizeY - pieceSizeY

		return [range(maxPositionY+1), range(maxPositionX+1)]

	def expandToBoard(self, posYX, boardSize):
		beforeYX = posYX
		afterYX  = [ boardSize[i] - self.rotated.shape[i] - posYX[i] for i in [0,1] ]
		if afterYX[0] < 0 or afterYX[1] < 0:
			raise Error("posYX incompatible with sizes " + str(self.rotated.shape) + ' ' + str(boardSize))
		return np.pad(self.rotated, ((beforeYX[0], afterYX[0]), (beforeYX[1], afterYX[1])),
					  'constant', constant_values=0)

	def convert(self, posYX):
		# rotation 0 --> start at (0,0)
		# rotation 1 --> start at (n,0)
		# rotation 2 --> start at (n,n)
		# rotation 3 --> start at (0,n)
		(sizeY, sizeX) = self.rotated.shape
		rangeY = range(sizeY) if self.rotation in [0,3] else range(sizeY-1, -1, -1)
		rangeX = range(sizeX) if self.rotation in [0,1] else range(sizeX-1, -1, -1)
		if self.rotation % 2 == 0:
			listCoords = [ (y+posYX[0], x+posYX[1], self.rotated[y,x]) for y in rangeY for x in rangeX if self.rotated[y,x] ]
		else:
			listCoords = [ (y+posYX[0], x+posYX[1], self.rotated[y,x]) for x in rangeX for y in rangeY if self.rotated[y,x] ]
		return listCoords

class PreComputedData:
	def __init__(self, boardSize = [6,6]):
		self.boardSize = boardSize
		self.descriptions = {}
		self.generate()
		
	def _generateSingle(self, piece_id, p: SlowPiece, pieceIndex=-1):
		result = []
		for rot in range(4):
			p.rotate(rot, fromUnrotated=True)
			listPosYX = p.insidePositionsYX(self.boardSize)
			result += [ p.convert([y,x]) for (y,x) in itertools.product(listPosYX[0], listPosYX[1]) ]
		if piece_id not in self.descriptions.keys():
			self.descriptions[piece_id] = {}
		self.descriptions[piece_id][p] = result

	def _generateAllPatterns(self):
		result = []
		for piece_id in range(len(Zentris.TETROMINOS)):
			description = Zentris.TETROMINOS[piece_id][0]
			# Convert to numpy
			sizeX = max([x[0] for x in description])+1
			sizeY = max([x[1] for x in description])+1
			n = len(description)

			# Autorise les pieces d'1 item a aller jusque 6
			# et les pieces de 2 items a aller jusque 5
			max_height = Zentris.BLOCK_MAX_HEIGHT
			if piece_id == 0: 
				max_height += 3
			elif piece_id == 1:
				max_height += 1

			for max_height in range(1, max_height+1):
				min_height = max(1, max_height-1)
				possible_heights = [min_height, max_height] if max_height != min_height else [max_height]
				for heights in itertools.product(possible_heights, repeat=n):
					if all([h!=max_height for h in heights]):
						continue
					unrotated = np.zeros([sizeY, sizeX], dtype=np.int8)
					# All pieces have either max_height or min_height as size
					for (xy,h) in zip(description, heights):
						unrotated[xy[1], xy[0]] = h
					# unrotated = _autoCrop(unrotated)
					result.append((piece_id, _autoCrop(unrotated)))
		return result

	def generate(self):
		patterns = self._generateAllPatterns()
		for (piece_id, pattern) in patterns:
			p = SlowPiece(pattern)
			self._generateSingle(piece_id, p)

	def print(self):
		for piece_id, patterns in self.descriptions.items():
			for pattern, description in patterns.items():
				# for p in pattern:
				# 	print(p)
				print('Pattern', piece_id, ' ', len(description), 'possibilities')
				pattern.print()
				print(description[10])

class MyPiece:
	def __init__(self, pl: PreComputedData):
		self.piece_id     = None
		self.pattern      = None
		self.descriptions = None
		self.bombIndex    = None
		if pl is not None:
			self.pl = pl
			self._randomPl()

	def _set(self, piece_id, pattern):
		self.piece_id     = piece_id
		self.pattern      = pattern
		self.descriptions = self.pl.descriptions[piece_id][pattern] if self.pl else None
		self.bombIndex    = None

	def _randomPl(self):
		descr = self.pl.descriptions
		# piece_id = random.choice(list(self.pl.descriptions.keys()))
		piece_id   = random.choices(list(descr.keys()), cum_weights=Zentris.CUM_WEIGHTS_ID, k=1)[0]
		max_height = random.choices(list(range(1,Zentris.BLOCK_TRUEMAX_HEIGHT+1)), cum_weights=Zentris.CUM_WEIGHTS_HEIGHT[piece_id], k=1)[0]
		pattern    = random.choice([p for p in descr[piece_id].keys() if p.unrotated.max() == max_height ])
		self.piece_id     = piece_id
		self.pattern      = pattern
		self.descriptions = descr[piece_id][pattern]
		draw = [ random.randrange(Zentris.BOARD_BOMB_PROBABILITY) == 0 for _ in range(len(self.descriptions[0])) ]
		self.bombIndex = next(i for i,v in enumerate(draw) if v) if any(draw) else None

	def getPossibilities(self):
		return self.descriptions

	def print(self):
		array = self.pattern.unrotated
		bombCoords = None if self.bombIndex is None else self.descriptions[0][self.bombIndex][:2]
		for y in range(array.shape[0]):
			for x in range(array.shape[1]):
				_printValue(array[y][x], isBomb=((y,x)==bombCoords))
			print()

class MyZentris:
	def __init__(self, board=None):
		if board is None:
			self.board      = np.zeros([1, 1], dtype=np.int8)
			self.boardSize  = self.board.shape
			self.reset()
		else:
			self.board      = board
			self.boardSize  = self.board.shape
		self.score      = 0
		self.scoreDelta = 0
		self.cleared    = []
		self.bombed     = 0
		self.bombCoords = []

	def reset(self):
		self.board = np.zeros([Zentris.BOARD_HEIGHT, Zentris.BOARD_WIDTH], dtype=np.int8)
		self.boardSize = self.board.shape
		self.score      = 0
		self.scoreDelta = 0
		self.cleared    = []
		self.bombed     = 0
		self.bombCoords = []

	def copy(self):
		result = MyZentris(board=self.board.copy())
		result.score      = self.score
		result.scoreDelta = self.scoreDelta
		result.cleared    = self.cleared.copy()
		result.bombed     = self.bombed
		result.bombCoords = self.bombCoords.copy()
		return result

	def play(self):
		self.scoreDelta = 0
		self.cleared    = []
		self.bombed     = 0

	def randomFill(self, nbNonZero, maxHeight=7):
		positions = [ p for p in itertools.product(range(self.boardSize[0]), range(self.boardSize[1])) ]
		chosenPositions = random.sample(positions, nbNonZero)
		for (y,x) in chosenPositions:
			height = random.randint(1,maxHeight)
			self.board[y, x] = height
			if random.randrange(100) == 0:
				self.bombCoords.append((y,x))

	def print(self, referenceBoard=None, addPiece=None, stylePiece=''):
		def _lookFor(coords, piece):
			if piece is None:
				return None
			result = [ x[2] for x in piece if x[:2]==coords ]
			return result[0] if len(result)>0 else None

		array = self.board
		shape = self.boardSize
		for y in range(array.shape[0]):
			for x in range(array.shape[1]):
				isBomb = (y,x) in self.bombCoords
				isNew = None if referenceBoard is None else (array[y,x] != referenceBoard.board[y,x])
				isPiece = _lookFor((y,x), addPiece)
				value = array[y][x] if isPiece is None else isPiece
				_printValue(value, isBomb, isNew, None if isPiece is None else stylePiece)
			print()
		if self.score != 0:
			print('Score=', self.score, self.scoreDelta, end='  ')
		if len(self.bombCoords) != 0:
			print('Bombe(s)=', self.bombCoords, end= '  ')
		print()

	def check(self, pieceDescription):	
		incompatible = any([self.board[y,x] for (y,x,_) in pieceDescription])
		return not(incompatible)

	def _addPiece(self, pieceDescription, bombIndex):
		for (y,x,h) in pieceDescription:
			self.board[y,x] = h
		if bombIndex is not None:
			bombCoord = pieceDescription[bombIndex]
			self.bombCoords.append(bombCoord[:2])

	def checkAndAddCopy(self, pieceDescription, bombIndex):
		# if any([self.board[y,x] for (y,x,_) in pieceDescription]):
		if not self.check(pieceDescription):
			return None
		result = self.copy()
		result._addPiece(pieceDescription, bombIndex)
		cleared = result._clearLines()
		bombed  = result._clearBombs()
		score   = result._updateScore(cleared, bombed)
		result.cleared.append(cleared)
		result.bombed      += bombed
		result.scoreDelta  += score
		result.score       += score
		return result

	def _clearLines(self):
		scoreCols  = np.amin(self.board,axis=0)
		scoreLines = np.amin(self.board,axis=1)
		# self.board=np.minimum(
		# 	self.board - scoreCols [np.newaxis, :],
		# 	self.board - scoreLines[:, np.newaxis]
		# )
		self.board = np.maximum(self.board - scoreCols [np.newaxis, :] - scoreLines[:, np.newaxis], 0)
		return sum(scoreCols)+sum(scoreLines)

	def _clearBombs(self):
		toRemove = None
		bombedCells = 0
		while toRemove != []:
			toRemove = []
			for (y,x) in self.bombCoords:
				if self.board[y,x] == 0:
					# Bomb explodes
					bombedCells += self.board[max(0, y-1):y+2, max(0, x-1):x+2].sum()
					self.board[max(0, y-1):y+2, max(0, x-1):x+2] = 0
					toRemove.append((y,x))
			for z in toRemove:
				self.bombCoords.remove(z)
		return bombedCells

	def _updateScore(self, cleared=0, bombed=0):
		# Cleared lines
		if cleared < 5:
			scoreTable = [1, 40, 120, 250, 420]
			score = scoreTable[cleared]
		else:
			score = int(22.5 * (cleared ** 2) + 14.5 * cleared + 2.5)
		# Bombed
		# score += 5 * min(bombed, 10)**2
		score += 10 * bombed

		return score

	def _computeProps(self, conf, pl: PreComputedData):
		result = []

		if 'BOA2D' in conf:
			conf = conf.replace('BOA2D', '')
			result = result + [np.reshape(self.board, [6,6,1])]

		if 'BOARD' in conf:
			conf = conf.replace('BOARD', '')
			result = result + self.board.flatten().tolist()

		if 'BHV' in conf:
			conf = conf.replace('BHV', '')
			# bumpiness  horizontal,   ertical
			bumpinessV_array = np.sum(np.abs(np.diff(self.board, axis=0)), axis=0)
			bumpinessV = [np.amin(bumpinessV_array), np.amax(bumpinessV_array)]
			bumpinessH_array = np.sum(np.abs(np.diff(self.board, axis=1)), axis=1)
			bumpinessH = [np.amin(bumpinessH_array), np.amax(bumpinessH_array)]

			result = bumpinessH + bumpinessV + result

		if 'A' in conf:
			pieces_str = re.findall('A[0-9]*', conf)[0]
			conf = conf.replace(pieces_str, '')
			pieces = [ int(c) for c in pieces_str[1:] ]
			# nb se positions possibles pour une nouvelle piece en L, idem´avec T
			availPositions = [0]*len(pieces)
			for piece_id in pieces:
				pattern = list(pl.descriptions[piece_id].keys())[0]
				descriptions = pl.descriptions[piece_id][pattern]
				availPositions[pieces.index(piece_id)] = [self.check(d) for d in descriptions].count(True)
			result = availPositions + result

		if 'w' in conf:
			conf = conf.replace('w', '')
			weights = np.array([
				[1, 1, 1, 1, 1, 1],
				[1, 2, 2, 2, 2, 1],
				[1, 2, 4, 4, 2, 1],
				[1, 2, 4, 4, 2, 1],
				[1, 2, 2, 2, 2, 1],
				[1, 1, 1, 1, 1, 1] ])
			weighted_average = np.average(self.board, weights=weights)
			result = weighted_average + result

		if 's' in conf:
			conf = conf.replace('s', '')
			# Bomb stats
			bombedHeights = 0
			bombHeights = 0
			for (y,x) in self.bombCoords:
				bombedHeights += self.board[max(0, y-1):y+2, max(0, x-1):x+2].sum()
				bombHeights   += self.board[y,x]
			bomb_stats = [len(self.bombCoords), bombedHeights, bombHeights]
			result = result + bomb_stats

		if 'b' in conf:
			conf = conf.replace('b', '')
			result = [np.array([self.bombed])] + result
		if 'c' in conf:
			conf = conf.replace('c', '')
			result = [np.array([sum(self.cleared)])] + result
		if 'C' in conf:
			conf = conf.replace('C', '')
			result = self.cleared + result
		if 'm' in conf:
			conf = conf.replace('m', '')
			result = [self.board.max()] + result
		if 'n' in conf:
			conf = conf.replace('n', '')
			result = [self.board.sum()] + result
		if 'o' in conf:
			conf = conf.replace('o', '')
			result = [np.count_nonzero(self.board)] + result

		return result

	def _propsInfo(self, conf):
		hyper_params =  'conf3_' + str(Zentris.BLOCK_MAX_HEIGHT)
		hyper_params += '_'      + str(Zentris.BOARD_INIT_CELLS)
		hyper_params += '_'      + str(Zentris.BOARD_BOMB_PROBABILITY)
		hyper_params += '_'      + str(Zentris.NB_EXPLORED_MOVES)

		conf_svg = conf
		size = 0
		if 'BOA2D' in conf:
			conf = conf.replace('BOA2D', '')
			size += 1

		if 'BOARD' in conf:
			conf = conf.replace('BOARD', '')
			size += 36

		if 'BHV' in conf:
			conf = conf.replace('BHV', '')
			size += 4

		if 'A' in conf:
			pieces_str = re.findall('A[0-9]*', conf)[0]
			conf = conf.replace(pieces_str, '')
			size += len(pieces_str[1:])
		if 'w' in conf:
			conf = conf.replace('w', '')
			size += 1
		if 's' in conf:
			conf = conf.replace('s', '')
			size += 3
		if 'b' in conf:
			conf = conf.replace('b', '')
			size += 1
		if 'c' in conf:
			conf = conf.replace('c', '')
			size += 1
		if 'C' in conf:
			conf = conf.replace('C', '')
			size += 3
		if 'm' in conf:
			conf = conf.replace('m', '')
			size += 1
		if 'n' in conf:
			conf = conf.replace('n', '')
			size += 1
		if 'o' in conf:
			conf = conf.replace('o', '')
			size += 1

		return (size, conf_svg + '_' + hyper_params)

def listNextStates_Single(z: MyZentris, p: MyPiece):
	liste = []

	# newBoards = z.checkAndAddCopyVector(p.getPossibilities(), p.bombIndex)
	# for newBoard in [ x for x in newBoards if x is not None ]:
	# 	liste.append([newBoard, []])

	for pieceDescription in p.getPossibilities():
		newBoard = z.checkAndAddCopy(pieceDescription, p.bombIndex)
		if newBoard is not None:
			####
			### Add position info to Piece class ###
			####
			liste.append([newBoard, pieceDescription])
	return liste

def listNextStates_Multi(z: MyZentris, piecesList):
	nb = len(piecesList)
	if nb == 0:
		return [ [z, []] ]
	result = []
	for i in range(nb):
		nextStates = listNextStates_Single(z, piecesList[i])
		nextPieces = [ p for p in piecesList ]
		del nextPieces[i]			
		# Recursivity
		for (nextState, expandedPiece) in nextStates:
			l = listNextStates_Multi(nextState, nextPieces)
			l = [ [x[0], [expandedPiece]+x[1]] for x in l ]
			result += l
		# 	del nextState
		# for p in nextPieces:
		# 	del p
	return result

def exploreBFS(z: MyZentris, piecesList):
	result = [ ([z], [], []) ]
	for level in range(len(piecesList)):
		newResult = []
		for boards, descriptions, patterns in result:
			board = boards[-1]
			for piece in [ p for p in piecesList if p not in patterns ]:
				for pieceDescription in piece.getPossibilities():
					newBoard = board.checkAndAddCopy(pieceDescription, piece.bombIndex)
					if newBoard is not None:
						newResult.append((boards      +[newBoard],
										  descriptions+[pieceDescription],
										  patterns    +[piece] ))
		# print('Level', level, ': ', len(result), '->', len(newResult), 'states', end='')
		result = filterNextStates(newResult, maxNb=Zentris.NB_EXPLORED_MOVES*10**(level+2))
		# print(' filtered =', len(result))
	return result

def filterNextStates(result, maxNb=1000000, dbg=False):
	# print('Filtering results: initial length', len(result), end='')
	# Remove duplicates
	filteredResults = []
	setVisited = set()
	for i in range(len(result)):
		x = result[i]
		# comparison = [ np.array_equal(x[0].board, y[0].board) for y in filteredResults ]
		# if not any(comparison):
		# 	filteredResults.append(x)
		s = x[0][-1].board.tostring()
		if s not in setVisited:
			filteredResults.append(x)
			setVisited.add(s)
			if len(filteredResults) >= maxNb:
				break
	# print('-> Final length', len(filteredResults))
	return filteredResults

def string2piece(s, pl):
	table_description = [ ('I', 1), ('I', 2), ('I', 3), ('I', 4), ('L', 3), ('T', 4), ('L', 4), ('J', 4) ]
	s_list = s.upper().split('B')
	s, b = s_list[0], s_list[1] if len(s_list) >= 2 else []

	shape, length = s[0].upper(), len(s)-1
	if (shape, length) not in table_description:
		print('Forme inconnue:', shape, length, s)
		return None
	piece_id = table_description.index((shape, length))
	patterns = pl.descriptions[piece_id]

	heights = [int(c) for c in s[1:]]
	for pattern in patterns.keys():
		heights_ = [x[2] for x in pattern.convert((0,0))]
		if heights_ == heights:
			p = MyPiece(pl)
			p._set(piece_id, pattern)
			if b:
				p.bombIndex = int(b)
			return p

	print('Hauteurs non trouvees', s, heights)
	for k,v in patterns.items():
		print([x[2] for x in k.convert((0,0))])
	return None

def string2boardline(s):
	heights = [int(c) for c in s]
	return np.array(heights)

def askBoard():
	board = MyZentris()
	board.reset()
	print('                       ......')
	for y in range(Zentris.BOARD_HEIGHT):
		while True:
			s = input('Saisir rangee numero ' + str(y) + ':')
			line = string2boardline(s)
			if line.size == 0:
				# Full board = 0
				return board
			if line.size == Zentris.BOARD_WIDTH:
				board.board[y, :] = line
				break

	# saisir: (score=0, bombes en 2,3 et 5,5)
	# 0 B2355 
	s = input('Saisir score courant et bombesYX:')
	s_list = s.upper().split('B')
	s, b = s_list[0], s_list[1] if len(s_list) >= 2 else []
	board.score = int(s)
	for i in range(0, len(b), 2):
		board.bombCoords.append( (int(b[i]), int(b[i+1])) )
	return board

def askPieces(pl):
	pieces = []
	help_ = '''\
T:    A    (4)      I:  ABCD  (1-4)
     BCD 
L:     D  (3-4)     J:   ABD   (4)
     ABC                   C'''
	print(help_)
	while True:
		for i in range(3):
			piece_string = input('Saisir piece numero ' +  str(i) + ':')
			pieces.append(string2piece(piece_string, pl))
		for p in pieces:
			if p:
				p.print()
				print()
		if None not in pieces:
			# if len(input('Confirmer pieces (rien=ok, sinon=recommencer)')) == 0:
			break
		pieces = []
		print(' -- RESAISIE -- ')
	return pieces

#####################################################################
def testBomb():
	pl = PreComputedData()
	board = MyZentris()
	board.randomFill(15)
	board.print()
	print()
	
	p = MyPiece(pl)
	while p.bombIndex is None:
		p = MyPiece(pl)
	p.print()
	print()

	for description in p.getPossibilities():
		if board.check(description):
			break
	print(description)
	print()

	newBoard = board.checkAndAddCopy(description, p.bombIndex)
	newBoard.print()
	breakpoint()

def test():
	import gc
	random.seed(19841114)

	pl = PreComputedData()
	for _ in range(1):
		pieces = [ MyPiece(pl) for _ in range(3) ] 
		for p in pieces:
			p.print()
			print()

		board = MyZentris()
		board.randomFill(10, maxHeight=2)
		board.print()

		# tmp = pieces[0].getPossibilities()[0]
		# pr = cProfile.Profile()
		# pr.enable()
		# for _ in range(100000):
		# 	board.checkAndAddCopy(tmp)
		# pr.disable()
		# pstats.Stats(pr).sort_stats(SortKey.CUMULATIVE).print_stats(10)
		# exit()

		print()

		# result = listNextStates_Multi(board, pieces)
		result = exploreBFS(board, pieces)
		# print(result)
		print(len(result), 'states')
		# result = filterNextStates(result)
		
		if len(result) > 0:
			for _ in range(1):
				i = random.randrange(len(result))
				print('State i=', i)
				# board.print()
				# print()
				_printDescriptions(result[i][1])
				print()
				result[i][0][-1].print(referenceBoard=board)
				# print(result[i][1])
				print('occupiedCells, nbItems, score, maxHeight, availPositions', result[i][0][-1]._computeProps(Zentris.CONF, pl))
				print()
				print()
			bestState = max(result, key=lambda x: x[0][-1].score)
			goodScore = int(bestState[0][-1].score*0.9)
			bests = [ x for x in result if x[0][-1].score >= goodScore ]
			for best in bests[:]:
				# board.print()
				# print()
				_printDescriptions(best[1])
				print()
				best[0][-1].print()
				print('occupiedCells, nbItems, score, maxHeight, availPositions', bestState[0][-1]._computeProps(Zentris.CONF, pl))
				print()
				print('-'*20)
			print(len(bests), 'solutions with score > 90% of max score = ', goodScore)
			# breakpoint()

		del result
		print()
		gc.collect()

def testGroup():
	random.seed(19841114)
	pl = PreComputedData()
	piece = MyPiece(pl)
	smartGroup(piece)

def play():
	pl = PreComputedData()
	board = MyZentris()
	board.randomFill(10, maxHeight=2)
	board.print()

	pieces = [ MyPiece(pl) for _ in range(3) ]
	nextStates = exploreBFS(board, pieces)
	
	#for _ in range(1):
	while (len(nextStates) > 0):
		nextState = max(nextStates, key=lambda x: x[0][-1].score)
		board = nextState[0][-1]
		_print_results(nextState)
		print(len(nextStates), 'coups possibles')
		# _printDescriptions(nextState[1])
		board.print()
		print('occupiedCells, nbItems, score, maxHeight, availPositions', board._computeProps(Zentris.CONF, pl))
		print()
		pieces = [ MyPiece(pl) for _ in range(3) ]
		nextStates = exploreBFS(board, pieces)

	print('the end, score=', board.score)
	for p in pieces: p.print() ; print()

def play2():
	random.seed(19841114)
	env = Zentris()
	env.reset()
	done = False
	env.myzentris.print()
	print()

	while not done:
		next_states = env.get_next_states()
		if len(next_states) == 0:
			done = True
		else:
			best_state = random.choice(list(next_states.values()))
			best_action = None
			for action, state in next_states.items():
				if state == best_state:
					best_action = action
					break
			reward = env.play(best_action, render=True)
			current_state = next_states[best_action]

	for p in env.pieces:
		p.print()
		print()
	print(env.get_game_score())	
	print(env.myzentris._propsInfo(Zentris.CONF))

def play3():
	env = Zentris()
	print(env.myzentris._propsInfo(Zentris.CONF))
	scores = []
	for _ in range(10):
		env = Zentris()
		env.reset()
		done = False

		while not done:
			next_states = env.get_next_states()
			if len(next_states) == 0:
				done = True
			else:
				best_state = max(next_states.values(), key=lambda x: (x[2], x[0]))
				best_action = None
				for action, state in next_states.items():
					if state == best_state:
						best_action = action
						break
				reward = env.play(best_action, render=False)
				current_state = next_states[best_action]

		finalScore = env.get_game_score()
		print('final score=', finalScore)
		scores.append(finalScore)
	print('AVERAGE=', sum(scores)/len(scores))

def realplay(filename=None):
	import pickle
	import os
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"] = ""
	from keras.models import Sequential, load_model
	from keras.layers import Dense


	if filename:
		agent = pickle.load(open(filename, 'rb'))
	pl = PreComputedData()
	memory = {'board': None, 'pieces': []}
	memory_name = f'game_{datetime.now().strftime("%m%d-%H%M")}.pickle'

	if True:
		board = askBoard()
	else: 
		board = MyZentris()
		board.randomFill(10, maxHeight=2)
	memory['board'] = board.copy()
	board.print()

	while True:
		pieces = askPieces(pl) if True else [ MyPiece(pl) for _ in range(3) ]
		memory['pieces'].append(pieces)
		with open(memory_name, 'wb') as f:
			pickle.dump(memory, f, pickle.HIGHEST_PROTOCOL)

		print('Calcul...')
		nextStates = exploreBFS(board, pieces)
		next_props = { nextState[0][-1]: nextState[0][-1]._computeProps(Zentris.CONF, pl) for nextState in nextStates }
		best_action, best_state = agent.find_best_state(next_props)

		if best_action is None:
			print('*** FIN ***')
			board.print()
			return board.score
		board = best_action
		# affichage des actions
		actions = [ x for x in nextStates if np.array_equal(x[0][-1], best_action) ][0]
		_print_results(actions)
		# Reset stuff
		board.play()


def models_battle(filenames, confs, piecesList=None):
	import pickle
	import os
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"] = ""
	from keras.models import Sequential, load_model
	from keras.layers import Dense
	import copy
	import random

	agents = [ pickle.load(open(filename, 'rb')) for filename in filenames ]

	while True:

		pl = PreComputedData()
		piecesList_copy = copy.deepcopy(piecesList)
		random.shuffle(piecesList_copy)
		board = MyZentris()
		board.randomFill(10, maxHeight=2)
		boards = [ board.copy() for _ in confs ]
		board.print()

		while len([ 1 for board in boards if board is not None ]) > 0:
			pieces = piecesList_copy.pop(0) if piecesList_copy else [ MyPiece(pl) for _ in range(3) ]
			for agent, conf, board, i in zip(agents, confs, boards, range(len(confs))):
				if board is None:
					continue
				print('[',i,']', end=' ')
				# print('Exploration des etats...', end='')
				nextStates = exploreBFS(board, pieces)
				next_props = { nextState[0][-1]: nextState[0][-1]._computeProps(conf, pl) for nextState in nextStates }
				best_action, best_state = agent.find_best_state(next_props, allow_random=False)

				if best_action is None:
					print('*'*20 + ' FIN pour ', conf, i, board.score, '*'*20)
					boards[i] = None
					continue
				boards[i] = best_action
				# actions = [ x for x in nextStates if np.array_equal(x[0][-1], best_action) ][0]
				# _print_results(actions)

				# Reset stuff
				print('score =', boards[i].score, end='   ')
				boards[i].play()
			print('\r', end='')
		print()
	
	print()
	print()


if __name__ == '__main__':
	profile = False
	colorama.init()
	root            = '/home/best/dev/tetris-ai/logs/'
	medi_model_name = root + '0510-0839-BOA2D_conf3_4_5_40_3-dsc0.9-eps0.1-0.01-bsz1024-ep5-nn10-10-[10, 10, 6, 6, 6]-ff10_-10000/model_0511-0821.pickle'
	good_model_name = root + '0514-2204-BOA2D_conf3_4_5_40_3-dsc0.9-eps0.1-0.01-bsz1024-ep5-nn10-10-[10, 10, 32, 32, 16, 16, 16]-ff10_-10000/model_0516-0950.pickle'
	best_model_name = root + '0512-1202-BOA2D_conf3_4_5_40_3-dsc0.99-eps0.1-0.01-bsz1024-ep5-nn10-10-[10, 10, 32, 32, 16, 16, 16]-ff10_-10000/model_0514-0728.pickle'

	# models_battle([medi_model_name, good_model_name, best_model_name], ['CbA256', 'CbA256BHV', 'CbA256BHVs'])
	realplay(good_model_name)