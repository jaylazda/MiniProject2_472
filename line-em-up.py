import time
import numpy as np

class Game:
	MINIMAX = False
	ALPHABETA = True
	HUMAN = 'H'
	AI = 'AI'
	LETTERS = 'ABCDEFGHIJ'
	def __init__(self, n = 5, b = 3, s = 3, blocs = [(1,2),(0,4),(0,0)], d1 = 3, d2 = 3, t = 0.3, recommend = True):
		self.recommend = recommend
		self.n = n
		self.b = b
		self.s = s
		self.blocs = blocs
		self.d1 = d1
		self.d2 = d2
		self.t = t
		self.logger = open(f'gameTrace-{n}{b}{s}{t}.txt', 'a')
		self.logger.write(f'n={n} b={b} s={s} t={t}\n')
		self.logger.write(f'blocs={blocs}\n\n')
		self.max_depth = d1
		self.initialize_game()
		# Variables used for statistics
		self.total_moves = 0
		self.e1_heuristic_eval_times = []
		self.e2_heuristic_eval_times = []
		self.state_count = 0
		self.e1_game_state_count = 0
		self.e2_game_state_count = 0
		self.depth_state_count = {}
		self.e1_game_depth_state_count = {}
		self.e2_game_depth_state_count = {}
		# Initialize the state count dictionaries
		for i in range(max(d1, d2)):
			self.depth_state_count[i+1] = 0
			self.e1_game_depth_state_count[i+1] = 0
			self.e2_game_depth_state_count[i+1] = 0
		self.start_time = 0


		
	def initialize_game(self):
		self.current_state = [ ['.'] * self.n for _ in range(self.n)]
		self.current_state = np.array(self.current_state)
		print(self.current_state)
		# Add blocks
		for bloc in self.blocs:
			print(bloc[0])
			print(bloc[1])
			self.current_state[bloc[0]][bloc[1]] = 'x'
		# Player with black piece always plays first
		print(self.current_state)
		self.player_turn = 'b'

	def draw_board(self):
		print()
		print(f'  {self.LETTERS[:self.n]}')
		self.logger.write(f'  {self.LETTERS[:self.n]}\n')
		dash = '-' * self.n
		print(f' +{dash}')
		self.logger.write(f' +{dash}\n')
		for x in range(self.n):
			print(x,'|', sep='', end='')
			self.logger.write(f'{x}|')
			for y in range(self.n):
				print(f'{self.current_state[x][y]}', end='')
				self.logger.write(f'{self.current_state[x][y]}')
			print()
			self.logger.write('\n')
		print()
		self.logger.write('\n')
		
	def is_valid(self, px, py):
		if px < 0 or px >= self.n or py < 0 or py >= self.n:
			return False
		elif self.current_state[px][py] != '.':
			return False
		else:
			return True
	
	def is_off_board(self, x, y):
		if x < 0 or x >= self.n or y < 0 or y >= self.n:
			return True
		else:
			return False

	def is_end(self):
		winner = self.row_col_diag_check(self.current_state.tolist())
		if (winner):
			return winner
		cols, diags = self.cols_and_diags()
		winner = self.row_col_diag_check(cols)
		if (winner):
			return winner
		winner = self.row_col_diag_check(diags)
		if (winner):
			return winner
		for i in range(self.n):
			for j in range(self.n):
				# There's an empty field, we continue the game
				if (self.current_state[i][j] == '.'):
					return None
		# It's a tie!
		return '.'
	
	def row_col_diag_check(self, list2D):
		for i in range(len(list2D)):
			for j in range(len(list2D[i]) + 1 - self.s):
				mask = list2D[i][j:j+self.s]
				if mask == ['b'] * self.s:
					return 'b'
				elif mask == ['w'] * self.s:
					return 'w'
		return None

	def check_end(self):
		self.result = self.is_end()
		# Printing the appropriate message if the game has ended
		if self.result != None:
			if self.result == 'b':
				print('The winner is black!')
				self.logger.write('The winner is black!\n')
			elif self.result == 'w':
				print('The winner is white!')
				self.logger.write('The winner is white!\n')
			elif self.result == '.':
				print("It's a tie!")
				self.logger.write("It's a tie!\n")
			self.initialize_game()
		return self.result

	def input_move(self):
		while True:
			print(F'Player {self.player_turn}, enter your move:')
			px = int(input('enter the x coordinate: '))
			py = int(input('enter the y coordinate: '))
			if self.is_valid(px, py):
				return (px,py)
			else:
				print('The move is not valid! Try again.')

	def switch_player(self):
		if self.player_turn == 'b':
			self.player_turn = 'w'
		elif self.player_turn == 'w':
			self.player_turn = 'b'
		return self.player_turn

	def minimax(self, max=False, level=1):
		# Minimizing for 'b' and maximizing for 'w'
		# Possible values are:
		# -10000 - win for 'b'
		# 0  - a tie
		# 10000  - loss for 'b'
		# We're initially setting it to 2 or -2 as worse than the worst case:
		value = 20000
		if max:
			value = -20000
		x = None
		y = None
		result = self.is_end()
		if result == 'b':
			return (-10000, x, y, level)
		elif result == 'w':
			return (10000, x, y, level)
		elif result == '.':
			return (0, x, y, level)
		for i in range(self.n):
			for j in range(self.n):
				if self.current_state[i][j] == '.':
					time_elapsed = time.time() - self.start_time
					if max:
						self.current_state[i][j] = 'w'
						if level == self.max_depth or time_elapsed >= self.t: # We are at the max depth or time is up, so evaluate the heuristic
							if self.heuristic == "e1":
								v = self.e1('w', level)
							else:
	 							v = self.e2(level)
							self.depth_state_count[level] += 1
							self.e1_game_depth_state_count[level] +=1
						else: # Traverse to the next level
							(v, _, _, _) = self.minimax(max=False, level=level+1)
						if v > value:
							value = v
							x = i
							y = j
					else:
						self.current_state[i][j] = 'b'
						if level == self.max_depth or time_elapsed >= self.t: # We are at the max depth, so evaluate the heuristic
							if self.heuristic == "e1":
								v = self.e1('b', level)
							else:
	 							v = self.e2(level)
						else: # Traverse to the next level
							(v, _, _, _) = self.minimax(max=True, level=level+1)
						if v < value:
							value = v
							x = i
							y = j
					self.current_state[i][j] = '.'
		return (value, x, y, level)

	def alphabeta(self, alpha=-2, beta=2, max=False, level=1):
		# Minimizing for 'b' and maximizing for 'w'
		# Possible values are:
		# -10000 - win for 'b'
		# 0  - a tie
		# 10000  - loss for 'b'
		# We're initially setting it to 20000 or -20000 as worse than the worst case:
		value = 20000
		if max:
			value = -20000
		x = None
		y = None
		result = self.is_end()
		if result == 'b':
			return (-10000, x, y, level)
		elif result == 'w':
			return (10000, x, y, level)
		elif result == '.':
			return (0, x, y, level)
		for i in range(self.n):
			for j in range(self.n):
				if self.current_state[i][j] == '.':
					time_elapsed = time.time() - self.start_time
					if max:
						self.current_state[i][j] = 'w'
						if level == self.max_depth or time_elapsed >= self.t: # We are at the max depth or time is up, so evaluate the heuristic
							if self.heuristic == "e1":
								v = self.e1('w', level)
							else:
	 							v = self.e2(level)
						else: # Traverse to the next level
							(v, _, _, depth) = self.alphabeta(alpha, beta, max=False, level=level+1)
						if v > value:
							value = v
							x = i
							y = j
					else:
						self.current_state[i][j] = 'b'
						if level == self.d1 or time_elapsed >= self.t: # We are at the max depth or time is up, so evaluate the heuristic
							if self.heuristic == "e1":
								v = self.e1('b',level)
							else:
	 							v = self.e2(level)
						else: # Traverse to the next level
							(v, _, _, depth) = self.alphabeta(alpha, beta, max=True, level=level+1)
						if v < value:
							value = v
							x = i
							y = j
					self.current_state[i][j] = '.'
					if max: 
						if value >= beta:
							return (value, x, y, level)
						if value > alpha:
							alpha = value
					else:
						if value <= alpha:
							return (value, x, y, level)
						if value < beta:
							beta = value
		return (value, x, y, level)

	# Basic heuristic
	# Counts the number of b's, w's and x's in each row
	# H is determined by subtracting the b's, adding the w's and subtracting or adding the x's depending
	# on if the algorithm is Max or Min
	def e1(self, player='b', level=1):
		self.state_count += 1
		self.e1_game_state_count += 1
		self.depth_state_count[level] += 1
		self.e1_game_depth_state_count[level] +=1
		h = 0
		for row in self.current_state.tolist():
			black_tiles = row.count('b')
			white_tiles = row.count('w')
			blocs = row.count('x')
			if player == 'b':
				h += white_tiles - black_tiles + blocs
			else:
				h += white_tiles - black_tiles - blocs
		return h
	
	# Applies a mask of size s and goes through all rows, columns and diagonals and evaluates the heuristic
	# Will sum up every permutation of the mask across the board and that will result in the final h()
	# If only b and . are present in the mask ie ['b','b','.','b'] then the # of b's is added to h() (ex: 3)
	# If only w and . are present in the mask ie ['w','.','w','.'] then the # of w's is subtracted from h() (ex: -2)
	# If a bloc x is detected in the mask, or there is a mix of w and b in the mask, then nothing is added to h()
	# If a win state is detected (mask with all b's), h() is set to 10000 and the function returns 
	# If a lose state is detected (mask with all w's), h() is set to -10000 and the function returns
	def e2(self, level=1):
		self.state_count += 1
		self.e2_game_state_count += 1
		self.depth_state_count[level] += 1
		self.e2_game_depth_state_count[level] +=1
		h = 0
		rowH = self.calculate_e2_h(self.current_state.tolist())
		if rowH == 10000 or rowH == -10000:
			return h
		cols, diags = self.cols_and_diags()
		colH = self.calculate_e2_h(cols)
		if colH == 10000 or colH == -10000:
			return h
		diagH = self.calculate_e2_h(diags)
		if diagH == 10000 or diagH == -10000:
			return h
		h = rowH + colH + diagH
		return h

	def calculate_e2_h(self, list2D):
		h = 0
		for i in range(len(list2D)):
			for j in range(len(list2D[i]) + 1 - self.s):
				mask = list2D[i][j:j+self.s]
				if 'x' not in mask:
					if not ('b' in mask and 'w' in mask):
						black_tiles = mask.count('b')
						white_tiles = mask.count('w')
						if black_tiles == 4:
							h = -10000
							return h
						elif white_tiles == 4:
							h = 10000
							return h
						h -= black_tiles
						h += white_tiles
		return h
		
	# returns 2 lists, the first being a list of all the columns in the current_state
	# and the second being a list of all diagonals of length greater than or equal to s in the current_state
	def cols_and_diags(self):
		cols = [i[0] for i in self.current_state]
		diags = [self.current_state[::-1,:].diagonal(i) for i in range(-self.current_state.shape[0]+1,self.current_state.shape[1])]
		diags.extend(self.current_state.diagonal(i) for i in range(self.current_state.shape[1]-1,-self.current_state.shape[0],-1))
		diags = [n.tolist() for n in diags if len(n) >= self.s]
		return cols, diags


	def play(self,algo=None,player_b=None,player_w=None,p1_heuristic="e1",p2_heuristic="e1"):
		if algo == None:
			algo = self.ALPHABETA
		if player_b == None:
			player_b = self.HUMAN
		if player_w == None:
			player_w = self.HUMAN
		self.logger.write(f'Player 1: {player_b} d={self.d1} a={algo} e1\n')
		self.logger.write(f'Player 2: {player_w} d={self.d2} a={algo} e1\n\n')
		while True:
			self.draw_board()
			self.total_moves += 1
			self.state_count = 0
			self.depth_state_count = dict.fromkeys(self.depth_state_count, 0)
			if self.check_end():
				break
			self.start_time = time.time()
			if algo == self.MINIMAX:
				if self.player_turn == 'b':
					self.max_depth = self.d1
					self.heuristic = p1_heuristic
					(h, x, y, _) = self.minimax(max=False)
				else:
					self.max_depth = self.d2
					self.heuristic = p2_heuristic
					(h, x, y, _) = self.minimax(max=True)

			else: # algo == self.ALPHABETA
				if self.player_turn == 'b':
					self.max_depth = self.d1
					self.heuristic = p1_heuristic
					(m, x, y, _) = self.alphabeta(max=False)
				else:
					self.max_depth = self.d2
					self.heuristic = p2_heuristic
					(m, x, y, _) = self.alphabeta(max=True)
			end = time.time()
			if (self.player_turn == 'b' and player_b == self.HUMAN) or (self.player_turn == 'w' and player_w == self.HUMAN):
				if self.recommend:
					print(F'Evaluation time: {round(end - self.start_time, 7)}s')
					print(F'Recommended move: x = {x}, y = {y}')
				(x,y) = self.input_move()
			if (self.player_turn == 'b' and player_b == self.AI) or (self.player_turn == 'w' and player_w == self.AI):
				self.logger.write(f'Player {self.player_turn} under AI control plays: x = {x}, y = {y}\n\n')
				print(F'Evaluation time: {round(end - self.start_time, 7)}s')
				print(F'Player {self.player_turn} under AI control plays: x = {x}, y = {y}')
			# Log statistics
			self.logger.write(f'i   Evaluation time: {round(end - self.start_time, 7)}s\n')
			self.logger.write(f'ii  Heuristic evaluations: {self.state_count}\n')
			self.logger.write(f'iii Evaluations by depth: {self.depth_state_count}\n')
			total = 0
			for level in self.depth_state_count:
				total += level*self.depth_state_count[level]
			avg_depth = round(total/self.state_count, 4)
			self.logger.write(f'iv  Average evaluation depth: {avg_depth}\n')
			self.logger.write(f'v   Average recursion depth: \n')
			if self.heuristic == "e1":
				self.e1_heuristic_eval_times.append(round(end - self.start_time, 7))
			else:
				self.e2_heuristic_eval_times.append(round(end - self.start_time, 7))
			self.current_state[x][y] = self.player_turn
			self.switch_player()
		# End of game logging
		e1_total = 0
		if self.e1_game_state_count > 0:
			for level in self.e1_game_depth_state_count:
				e1_total += level*self.e1_game_depth_state_count[level]
			e1_avg_depth = round(e1_total/self.e1_game_state_count, 4)
		e2_total = 0
		if self.e2_game_state_count > 0:
			for level in self.e2_game_depth_state_count:
				e2_total += level*self.e2_game_depth_state_count[level]
			e2_avg_depth = round(e2_total/self.e2_game_state_count, 4)
		if p1_heuristic != p2_heuristic:
			self.logger.write(f'Heuristic e1:\n')
			self.logger.write(f'6(b)i   Average evaluation time: {round(sum(self.e1_heuristic_eval_times)/len(self.e1_heuristic_eval_times), 7)}s\n')
			self.logger.write(f'6(b)ii  Total heuristic evaluations: {self.e1_game_state_count}\n')
			self.logger.write(f'6(b)iii Evaluations by depth: {self.e1_game_depth_state_count}\n')
			self.logger.write(f'6(b)iv  Average evaluation depth: {e1_avg_depth}\n')
			self.logger.write(f'6(b)v   Average recursion depth: \n')
			self.logger.write(f'Heuristic e2:\n')
			self.logger.write(f'6(b)i   Average evaluation time: {round(sum(self.e2_heuristic_eval_times)/len(self.e2_heuristic_eval_times), 7)}s\n')
			self.logger.write(f'6(b)ii  Total heuristic evaluations: {self.e2_game_state_count}\n')
			self.logger.write(f'6(b)iii Evaluations by depth: {self.e2_game_depth_state_count}\n')
			self.logger.write(f'6(b)iv  Average evaluation depth: {e2_avg_depth}\n')
			self.logger.write(f'6(b)v   Average recursion depth: \n')
			self.logger.write(f'6(b)vi  Total moves: {self.total_moves}\n')
		else:
			if p1_heuristic == "e1":
				self.logger.write(f'Heuristic e1:\n')
				self.logger.write(f'6(b)i   Average evaluation time: {round(sum(self.e1_heuristic_eval_times)/len(self.e1_heuristic_eval_times), 7)}s\n')
				self.logger.write(f'6(b)ii  Total heuristic evaluations: {self.e1_game_state_count}\n')
				self.logger.write(f'6(b)iii Evaluations by depth: {self.e1_game_depth_state_count}\n')
				self.logger.write(f'6(b)iv  Average evaluation depth: {e1_avg_depth}\n')
				self.logger.write(f'6(b)v   Average recursion depth: \n')
				self.logger.write(f'6(b)vi  Total moves: {self.total_moves}\n')
			else:
				self.logger.write(f'Heuristic e2:\n')
				self.logger.write(f'6(b)i   Average evaluation time: {round(sum(self.e2_heuristic_eval_times)/len(self.e2_heuristic_eval_times), 7)}s\n')
				self.logger.write(f'6(b)ii  Total heuristic evaluations: {self.e2_game_state_count}\n')
				self.logger.write(f'6(b)iii Evaluations by depth: {self.e2_game_depth_state_count}\n')
				self.logger.write(f'6(b)iv  Average evaluation depth: {e2_avg_depth}\n')
				self.logger.write(f'6(b)v   Average recursion depth: \n')
				self.logger.write(f'6(b)vi  Total moves: {self.total_moves}\n')
		self.logger.close()
		return

def userInput():
    while True:
        n = int(input("Size of the board [3...10]: "))
        if n < 3 or n > 10:
            print("Input is outside valid range")
        else:
            break
    while True:
        b = int(input("Number of blocs [0...2^n]: "))
        if b < 0 or b > 2**n:
            print("Input is outside valid range")
        else:
            break
    while True:
        s = int(input("Winning line up size [3...n]: "))
        if s < 3 or s > n:
            print("Input is outside valid range")
        else:
            break
    blocs = []
    for i in range(b):
        while True:
            x = int(input(f"Enter bloc {i+1}'s x coordinate: "))
            y = int(input(f"Enter bloc {i+1}'s y coordinate: "))
            if x < 0 or y < 0 or x > n-1 or y > n-1:
                print("Input is outside valid range")
            elif (x,y) in blocs:
                print("There is already a bloc at this coordinate")
            else:
                break
        blocs.append((x,y))
    d1 = int(input("Max depth for player 1: "))
    d2 = int(input("Max depth for player 2: "))
    t = int(input("Max time to return a move (seconds): "))
    while True:
        a = input("Use minimax (false) or alphabeta (true): ").lower()
        if a != "false" and a != "true":
            print("Please type true/false")
        else:
            a = True if a == "true" else False
            break
    while True:
        m = input("Play mode (H-H/H-AI/AI-H/AI-AI): ").upper()
        if m != "H-H" and m != "H-AI" and m != "AI-H" and m != "AI-AI":
            print("Please type one of (H-H/H-AI/AI-H/AI-AI)")
        else:
            break
    return n, b, blocs, s, d1, d2, t, a, m

def main():
	# n, b, blocs, s, d1, d2, t, a, m = userInput()
	g = Game(recommend=True)
	a = True
	m = 'AI-AI'
	m = m.split('-')
	p1_h = "e2"
	p2_h = "e2"
	g.play(algo=a, player_b=m[0], player_w=m[1], p1_heuristic=p1_h, p2_heuristic=p2_h)
	


if __name__ == "__main__":
	main()