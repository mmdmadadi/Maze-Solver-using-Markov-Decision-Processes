import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
##########################################################TURNING IMAGE INTO MATRIX

def get_maze_array(image_file):
    # Open the image file and convert it to grayscale
    maze = Image.open(image_file).convert('L')
    
    # Get the size of the maze image
    cols, rows = maze.size
    
    # Convert the image to a binary matrix
    binary_matrix = maze.point(lambda p: p > 127 and 1)
    
    # Resize the binary matrix (optional)
    binary_matrix = binary_matrix.resize((cols//i, rows//i), Image.NEAREST)
    
    # Get the new size of the resized matrix
    cols, rows = binary_matrix.size
    
    # Convert the binary matrix to a numpy array
    maze_array = np.array(binary_matrix)
    
    return maze_array
#########################################################





##########################################################SCALING FACTOR

i = 1
mazearray = get_maze_array('maze.png')
rows, columns = mazearray.shape
print(rows)
for r in range(rows):
    print(i)
    if mazearray[r][0] == 0:
        i = i + 1
    if mazearray[r][0] == 1:
        break
i = i-1    
#########################################################





##########################################################DEFINING START AND END OF THE PATH

mazearray = get_maze_array('maze.png')
rows, columns = mazearray.shape
for r in range(rows):
    if mazearray[r][0] == 1:
        mazearray[r][0] = 2
        startrow = r
    if mazearray[r][-1] == 1:
        mazearray[r][-1] = 3
        endrow = r
#########################################################    



#mazearray[1][0] = 2
#mazearray[-2][-1] = 3
# print('')
# print('maze array:')
# print('')
# print('start state is 2')
# print('start state is 3')
# print(mazearray)

plt.matshow(mazearray)
plt.axis('off')
plt.show()



#RL:
#########################################################
#Hyperparameters
SMALL_ENOUGH = 0.005
GAMMA = 0.9
NOISE = 0.15
#########################################################





##########################################################Define all states

all_states=[]
for row in range(len(mazearray)) : 
    for col in range(len(mazearray[0])):
        if mazearray[row][col] == 1:
            all_states.append((row,col))
        elif mazearray[row][col] == 2:
            all_states.append((row,col))
        elif mazearray[row][col] == 3:
            all_states.append((row,col))
#print(all_states)
#########################################################





##########################################################Define rewards for all states

rewards=np.zeros((len(mazearray),len(mazearray[0])))
for row in range(len(mazearray)) : 
    for col in range(len(mazearray[0])):
        if mazearray[row][col] == 1:
            rewards[row][col]=0
        # elif mazearray[row][col] == 0:
        #     rewards[row][col]=-1
        elif mazearray[row][col] == 2:
            rewards[row][col]=0
        elif mazearray[row][col] == 3:
            rewards[row][col]=1

#########################################################





##########################################################Dictionnary of possible actions for maze cells

actions={}
for row in range(len(mazearray)) : 
    for col in range(len(mazearray[0])):
       if mazearray[row][col] == 1:
           actions[(row,col)] = ('D','R','L','U')
       if mazearray[row][col] == 2:
           actions[(row,col)] = ('D','R','L','U')
#########################################################





##########################################################Define an initial policy

policy={}
for s in actions.keys():
    policy[s]=np.random.choice(actions[s])
# print(policy)
#########################################################





##########################################################Define initial value function

V={} 
for row in range(len(mazearray)) : 
    for col in range(len(mazearray[0])):
        if mazearray[row][col] == 1:
            V[(row, col)] = 0
        # elif mazearray[row][col] == 0:
        #     V[(row, col)] = -1
        elif mazearray[row][col] == 2:
            V[(row, col)] = 0
        elif mazearray[row][col] == 3:
            V[(row, col)] = 10
#########################################################





##########################################################DEFINING POSSIBLE ACTIONS FOR EACH STATE AND ENVIRONMENT

def env(s, a):
    # print(s, a)
    if a == 'U' :
        if (s[0]-1, s[1]) in all_states:
            nxt = [s[0]-1, s[1]]
        else :
            nxt = [s[0], s[1]]
    if a == 'D':
        if (s[0]+1, s[1]) in all_states:
             nxt = [s[0]+1, s[1]]
        else:
             nxt = [s[0], s[1]]
    if a == 'L':
        if (s[0], s[1]-1) in all_states:
            nxt = [s[0], s[1]-1]
        else:
            nxt = [s[0], s[1]]
    if a == 'R':
        if (s[0], s[1]+1) in all_states:
            nxt = [s[0], s[1]+1]
        else:
            nxt = [s[0], s[1]]
    
    return nxt, rewards[nxt[0]][nxt[1]]
#########################################################





##########################################################VALUE ITERATION

iteration=0
while True:
    biggest_change=0
    for s in all_states:
        if s in actions.keys():
            max_value= float('-inf')
            for a in actions[s]:
                total =0 
                (nxt, reward) = env(s, a)
                remaining_actions = [new_action for new_action in actions[s] if new_action != a]
                (nxt2, reward2) = env(s, remaining_actions[0])
                (nxt3, reward3) = env(s, remaining_actions[1])            
                (nxt4, reward4) = env(s, remaining_actions[2])
                total +=0.85*(reward + GAMMA * V[tuple(nxt)]) + 0.05*(reward2 + GAMMA * V[tuple(nxt2)]) + 0.05*(reward3 + GAMMA * V[tuple(nxt3)]) + 0.05*(reward4 + GAMMA * V[tuple(nxt4)])
                if total >= max_value:
                    max_value = total
            V[s] = max_value
            iteration+=1
    if iteration > 300000:
            break
# print('')
# print('States Value Function are: ')
# print('')
# print(V)
#########################################################





##########################################################DEFINING POLICY

for s in all_states:
    if s in actions.keys():
            (nxt, reward) = env(s, actions[s][0])
            (nxt2, reward2) = env(s, actions[s][1])
            (nxt3, reward3) = env(s, actions[s][2])            
            (nxt4, reward4) = env(s, actions[s][3])
            V1={V[tuple(nxt)] : actions[s][0], V[tuple(nxt2)] : actions[s][1], V[tuple(nxt3)] : actions[s][2], V[tuple(nxt4)] : actions[s][3] }
            max_value= float('-inf')
            for Val in V1.keys():
                total = Val
                if total >= max_value:
                    max_value = total
                    policy[s] = V1[max_value]
# print('')
# print('Policy for each state:')
# print('')
# print(policy)
#########################################################





##########################################################DEFINING PATH

start = (1,0)
flag = True

# Get the shape of the matrix
rows, columns = mazearray.shape

# Convert negative indices to positive indices
row_index = rows - 2
column_index = columns - 2

# print('')
# print('states path are:')
# print('')

while flag:
    next_action = policy[start]
    (next_state, reward) = env(start, next_action)
    mazearray[next_state[0]][next_state[1]] = 2
    start = tuple(next_state)
    print(next_state)
    if next_state[0] == endrow & next_state[1] == column_index:
        flag = False
mazearray[-2][-1] = 2
print('')
print('updated maze array:')
print('')
print('path is 2:')
print('')
print(mazearray)
#########################################################            





##########################################################PLOT SOLVED MAZE

# plt.matshow(mazearray)
# plt.axis('off')
# plt.show()
# Define custom colormap
colors = ['black', 'white', 'red']
cmap = ListedColormap(colors)
# Plot the matrix with the custom colormap
plt.imshow(mazearray, cmap=cmap)
plt.axis('off')  # Turn off axis
#plt.colorbar(ticks=[0, 1, 2], format=plt.FuncFormatter(lambda val, loc: colors[int(val)])) # Show colorbar with custom colors
#plt.grid(True)  # Turn off grid lines
plt.savefig('solvedmaze.png', bbox_inches='tight', pad_inches=0)  # Save the image
plt.show()
#########################################################




