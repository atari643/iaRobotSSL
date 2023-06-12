import numpy as np
import rsk
import time
from rsk.simulator import Simulator
import math
# Définition des constantes
NUM_STATES = 1000
NUM_ACTIONS = 5
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
WIN_REWARD = 5
LOSE_REWARD = -5
INTERVALLE_X = (1.84, -0.34, 0.34)

# Initialisation de la table des Q-values
q_table = {}

def regarder_balle(direction_regard,  ball_position, robot_position):
    xball,yball = ball_position[0],ball_position[1]
    xdef,ydef = robot_position[0][0], robot_position[0][1]
    if(xdef-xball==0):
        xdef+=0.0001
    return float(direction_regard*math.pi+math.atan((ydef-yball)/(xdef-xball)))
def zoneAutourBalle(balle_position, robot_positions):
    xball, yball = balle_position
    xadv, yadv = robot_positions[0][0], robot_positions[0][1]
    if (xball - xadv < 0.11 and xball - xadv > -0.11 and yball - yadv < 0.11 and yball - yadv > -0.11):
        return True
    else:
        return False
# Fonction pour obtenir l'état à partir des positions des robots et de la balle
def get_state(robot_positions, ball_position, numrobot, client):
    state = (robot_positions[0][0], robot_positions[0][1], robot_positions[0][2], ball_position[0], ball_position[1])
    return state
distancemoy=1.07
# Fonction pour sélectionner une action à partir de l'état actuel
def select_action(state, ball_position, robot_positions):
    if np.random.uniform() < EPSILON or state not in q_table:
        action = np.array([np.random.uniform(-1.83/2, 1.83/2), np.random.uniform(-1.22/2, 1.22/2), np.random.uniform(-np.pi, np.pi)])
    else:
        # Exploitation : sélectionne l'action avec la plus grande Q-value
        action = np.zeros(3)
        action[:3] = q_table[state]
    return action

# Fonction pour exécuter une action et obtenir la récompense et le nouvel état
def execute_action(action, team, client, robot_positions, ball_position):
    # Mettre à jour la position du robot avant d'exécuter l'action
    robot_positions[0][0] = action[0]
    robot_positions[0][1] = action[1]
    robot_positions[0][2] = action[2]
    arrived = False
    while not arrived:
        arrived=client.robots["blue"][1].goto((action[0], action[1], action[2]), wait=True)
    client.robots["blue"][1].kick(1)
    reward = rewards(ball_position, robot_positions)
    next_state = get_state(robot_positions, ball_position, 1, client)
    return reward, next_state, ball_position, robot_positions

# Fonction pour calculer la récompense
def rewards(position_Balle, robot_positions):
    xball, yball = position_Balle
    xball2,yball2=client.ball
    result = -1
    if(xball!=xball2 or yball!=yball2):
        result+4
    if xball > 1.83/2 and yball < 0.34 and yball > -0.34:
        result+WIN_REWARD
    if(zoneAutourBalle(position_Balle, robot_positions)):
        result+3
    if(robot_positions[0][0]>xball):
        result-3
    if (robot_positions[0][0] < xball):
        result+2
    if(xball<-1.83/2 and yball < 0.34 and yball > -0.34):
        result+LOSE_REWARD
    return result
import pickle

with rsk.Client(host='127.0.0.1', key='') as client:
    # Boucle d'apprentissage
    robot=client.robots["blue"][1] 
    q_table = {}
    import pickle

    with open('q_table.pkl', 'rb') as f:
        try:
            q_table = pickle.load(f)
            print(q_table)
        except:
            print("Le fichier est corrompu.")
    with open('q_table.pkl', 'wb') as f:
        for episode in range(1000):
            # Initialisation de l'état et de la récompense
            ball_position = [0, 0]
            robot_positions = [[robot.position[0], robot.position[1], robot.orientation]]

            state = get_state(robot_positions, ball_position, 1, client)
            reward = 0

            # Boucle d'exécution de l'épisode
            while True:
                try:
                    # Sélection de l'action à partir de l'état actuel
                    action = select_action(state, ball_position, robot_positions)
                    # Exécution de l'action et obtention de la récompense et du nouvel état
                    reward, next_state, ball_position2, robot_positions = execute_action(action, "blue", client, robot_positions, ball_position)
                    # Mise à jour de la table des Q-values
                    if state not in q_table:
                        q_table[state] = {}
                    if (action[0], action[1], action[2]) not in q_table[state]:
                        q_table[state][(action[0], action[1], action[2])] = 0
                    q_table[state][(action[0], action[1], action[2])] += ALPHA * (reward + GAMMA * max(q_table[next_state].values()) - q_table[state][(action[0], action[1], action[2])])
                    state = next_state

                    # Vérification si l'épisode est terminé
                    ball_position[0] = ball_position2[0]
                    ball_position[1] = ball_position2[1]
                    if ball_position[0] == INTERVALLE_X[0] and INTERVALLE_X[1] <= ball_position[1] <= INTERVALLE_X[2]:
                        reward = WIN_REWARD
                        break
                    print("Episode {}: reward = {}".format(episode, reward))
                except:
                    print("error")
            pickle.dump(q_table, f) 
    # Affichage de la récompense obtenue à la fin de l'épisode
    print("Episode {}: reward = {}".format(episode, reward))