# coding=utf-8
import random, re, datetime
import math
import copy


class Agent(object):
	def __init__(self, game):
		self.game = game
		self.dic = {}
		self.list=[[],[(1, 1), (3, 1), (3, 3), (4, 1), (4, 2), (4, 3), (4, 4)],[(2, 1), (2, 2), (3, 2)],[(19, 1), (17, 1), (17, 3), (16, 1), (16, 2), (16, 3), (16, 4)],[(18, 1), (18, 2), (17, 2)]]
		self.last_action = None
		self.depth = 3
		self.action_list = []
		self.count=0

	def getAction(self, state):
		raise Exception("Not implemented yet")


class RandomAgent(Agent):
	def getAction(self, state):
		legal_actions = self.game.actions(state)
		self.action = random.choice(legal_actions)


class SimpleGreedyAgent(Agent):
	# a one-step-lookahead greedy agent that returns action with max vertical advance
	def getAction(self, state):
		legal_actions = self.game.actions(state)
		player = self.game.player(state)
		# self.action = random.choice(legal_actions)
		for action in legal_actions:
			if action[1][0] == 2 or (action[1][0] == 3 and action[1][1] == 2):
				if self.game.succ(state, action)[1].board_status[(action[1][0], action[1][1])] == player + 2:
					self.action = action
		if player == 1:
			max_vertical_advance_one_step = max([action[0][0] - action[1][0] for action in legal_actions])

			max_actions = [action for action in legal_actions if
			               action[0][0] - action[1][0] == max_vertical_advance_one_step]
		else:
			max_vertical_advance_one_step = max([action[1][0] - action[0][0] for action in legal_actions])
			max_actions = [action for action in legal_actions if
			               action[1][0] - action[0][0] == max_vertical_advance_one_step]
		self.action = random.choice(max_actions)


class TeamNameMinimaxAgent(Agent):
	def getAction(self, state):
		player = state[0]
		board = state[1]
		self.start = datetime.datetime.now()
		global count,delta
		legal_actions = self.game.actions(state)
		player_status = board.getPlayerPiecePositions(player)
		self.count += 1
		if self.count >= 100:
			self.count = 0
		if self.game.isEnd(state,100):
			self.count = 0

		#先用贪婪加快速度，打乱对手的节奏
		if self.count <= 10:
			legal_actions = self.game.actions(state)
			player = self.game.player(state)
			board = state[1]
			flag = 0

			if player == 1:
				max_vertical_advance_one_step = -100
				max_actions = {}
				for action in legal_actions:
					if board.board_status[action[0]] == 3:  # special pegs
						if action[0] == (2, 1) or action[0] == (2, 2) or action[0] == (3, 2) or action[1] == (1, 1):
							continue
						else:
							if board.board_status[action[0]] == 3 and (
									action[1] == (2, 1) or action[1] == (2, 2) or action[1] == (3, 2)):
								flag = 1
								self.action = action
								break
							else:
								if (action[0][0] < 13):
									priority = 1
								else:
									priority = 0
								v = action[0][0] - action[1][0] + priority
								if v >= max_vertical_advance_one_step:
									max_vertical_advance_one_step = v
									max_actions.setdefault(v, []).append(action)
					else:
						if action[1] == (2, 1) or action[1] == (2, 2) or action[1] == (3, 2):
							continue
						v = action[0][0] - action[1][0]
						if v >= max_vertical_advance_one_step:
							max_vertical_advance_one_step = v
							max_actions.setdefault(v, []).append(action)
			else:
				max_vertical_advance_one_step = -100
				max_actions = {}
				for action in legal_actions:
					if board.board_status[action[0]] == 4:  # special pegs
						if action[0] == (18, 1) or action[0] == (18, 2) or action[0] == (17, 2) or action[1] == (19, 1):
							continue
						else:
							if board.board_status[action[0]] == 4 and (
									action[1] == (18, 1) or action[1] == (18, 2) or action[1] == (17, 2)):
								flag = 1
								self.action = action
								break
							else:
								if (action[0][0] > 7):
									priority = 1
								else:
									priority = 0
								v = action[1][0] - action[0][0] + priority
								if v >= max_vertical_advance_one_step:
									max_vertical_advance_one_step = v
									max_actions.setdefault(v, []).append(action)
					else:
						if action[1] == (18, 1) or action[1] == (18, 2) or action[1] == (17, 2):
							continue
						v = action[1][0] - action[0][0]
						if v >= max_vertical_advance_one_step:
							max_vertical_advance_one_step = v
							max_actions.setdefault(v, []).append(action)
			if flag == 0:
				self.action = random.choice(max_actions[max_vertical_advance_one_step])

		else:
			action = self.ALPHA_BETA_SEARCH(state)
			now = datetime.datetime.now()
			time = str(now - self.start)
			delta = float(time.split(':')[-1])  # 时间差
			print(delta)
			if self.last_action is not None and self.last_action[0] == action[1]:
				cnt = 1
				sorted_d = sorted(self.dic.keys(), reverse=True)
				while (self.last_action[0] == action[1]):
					action = self.dic[sorted_d[cnt]]
					cnt += 1
			self.last_action = action
			self.action = action


	def ALPHA_BETA_SEARCH(self, state):
		v = self.MAX_VALUE(state, -9999, 9999, self.depth)
		#if delta > 0.97:  # 时间到时返回局部最优解
			#return self.dic[v]
		return self.dic[v]


	def MAX_VALUE(self, state, alpha, beta, depth):
		if depth == 1:
			return self.Estimate_Func(state)
		v = -99999
		depth -= 1
		legal_actions = self.game.actions(state)
		random.shuffle(legal_actions)
		for action in legal_actions:
			self.action_list.append(action)
			v = max(v, self.MIN_VALUE(self.game.succ(state, action), alpha, beta, depth))  # ,action_list))
			if depth == 2:
				if v in self.dic.keys():
					v+=self.action_list[0][0][0]-self.action_list[0][1][0]
				self.dic[v] = action
			self.action_list.pop()
			if v >= beta:
				return v
			alpha = max(alpha, v)
		return v

	def MIN_VALUE(self, state, alpha, beta, depth):
		if depth == 1:
			return self.Estimate_Func(state)
		v = 99999
		depth -= 1
		legal_actions = self.game.actions(state)
		random.shuffle(legal_actions)
		for action in legal_actions:
			self.action_list.append(action)
			v = min(v, self.MAX_VALUE(self.game.succ(state, action), alpha, beta, depth))  # ,action_list))
			self.action_list.pop()
			if depth == 2:
				self.dic[v] = action
			if v <= alpha:
				return v
			beta = min(beta, v)
		return v

	def Estimate_Func(self, state):  # ,action_list):

		value = float(0.0)
		weight2 = 1
		player = state[0]
		board = state[1]
		player2=player
		if player==2:
			player2+=1
		state[1].board_status[self.action_list[1][0]] = state[1].board_status[self.action_list[1][1]]
		state[1].board_status[self.action_list[1][1]] = 0
		state[1].board_status[self.action_list[0][0]] = state[1].board_status[self.action_list[0][1]]
		state[1].board_status[self.action_list[0][1]] = 0

		pos = state[1].getPlayerPiecePositions(player)
		pos1 = set((row, col) for (row, col) in pos if state[1].board_status[(row, col)] == player)
		pos2= set((row, col) for (row, col) in pos if state[1].board_status[(row, col)] == player+2)#special pos
		pos1=list(pos1)
		pos2=list(pos2)
		unoccupied_common_des = set(self.list[player2]).difference(pos1)
		unoccupied_special_des = set(self.list[player2+1]).difference(pos2)

		#对特殊棋子进行处理
		if state[1].board_status[self.action_list[0][0]] == player + 2:
			weight=10
			if len(unoccupied_special_des) == 1:
				weight=11
			if len(unoccupied_special_des) == 0:
				weight=0
			if self.action_list[0][1] in self.list[player2+1]:#要跳进来了
				if self.action_list[0][0] in self.list[player2+1]:
					value -= 1000000000
					weight=0
				else:
					value+=1000000000
			if self.action_list[0][0] in self.list[player2+1]:
				value-=1000000000
				weight=0
		else:#蓝色的 跳出去
			weight=1
			if self.action_list[0][0] in self.list[player2+1]:
				if self.action_list[0][1] in self.list[player2+1]:#没必要在里面跳来跳去
					value -= 1000000
				else:
					value+=1000000
			if len(unoccupied_special_des)==0:#黄色的走完了等蓝色的
				self.count=0



		if player==1:
			if (self.action_list[0][1][0] - 2) > self.action_list[0][0][0]:
				value -= 5000 * weight
			if self.action_list[0][0][0] < self.action_list[0][1][0]:
				weight *= 5
			if self.action_list[0][1] in unoccupied_common_des:
				weight *= 5

			value += 1000 * (self.action_list[0][0][0] - self.action_list[0][1][0]) * weight + 200 * (
					self.action_list[1][0][0] - self.action_list[1][1][0])
			value += weight2 * 150 * (self.action_list[0][0][0] - 4)
			if self.action_list[0][0] in self.list[player2] and self.action_list[0][1][0] > 4:
				value -= 2000
			# 不让它们进入原来的位置
			if self.action_list[0][0][0] < 16 and self.action_list[0][1][0] > 15:
				return float('-inf')
			# 禁止往回跳
			if self.action_list[0][0][0] <= self.action_list[0][1][0] and self.action_list[0][0][0] > 8:
				return float('-inf')

		else:
			if (self.action_list[0][0][0] - 2) > self.action_list[0][1][0]:
				value -= 5000 * weight
			if self.action_list[0][1][0] < self.action_list[0][0][0]:
				weight *= 5
			if self.action_list[0][1] in unoccupied_common_des:
				weight *= 5
			value += 1000 * (self.action_list[0][1][0] - self.action_list[0][0][0]) * weight + 200 * (
					self.action_list[1][1][0] - self.action_list[1][0][0])
			value += weight2 * 150 * (16-self.action_list[0][0][0] )
			if self.action_list[0][0] in self.list[player2] and self.action_list[0][1][0] <16:
				value -= 2000
			if self.action_list[0][0][0] >4  and self.action_list[0][1][0]<6 :#不是很清楚
				return float('-inf')
			# 禁止往回跳
			if self.action_list[0][0][0] >= self.action_list[0][1][0] and self.action_list[0][0][0] <10:
				return float('-inf')




		return value




### END CODE HERE ###
