import numpy as np
import rsk
import time
from rsk.simulator import Simulator
import math
import atexit
import pickle

# Définition des constantes
NUM_STATES = 1000
NUM_ACTIONS = 5
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
WIN_REWARD = 8
LOSE_REWARD = -8
INTERVALLE_X = (1.84, -0.34, 0.34)

# Initialisation de la table des Q-values
q_table = {}
def save_q_table():
    with open('q_table.pkl', 'wb') as f:
        pickle.dump(q_table, f)



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
    action = np.zeros(3)
    if np.random.uniform() < EPSILON or state not in q_table:
        action = np.array([np.random.uniform(-1.83/2, 1.83/2), np.random.uniform(-1.22/2, 1.22/2), np.random.uniform(-np.pi, np.pi)])
        action_rounded = np.round(action, 2)
        print(action_rounded)
    else:
        # Exploitation : sélectionne l'action avec la plus grande Q-value
        action_values = q_table[state]
        best_action = max(action_values, key=action_values.get)
        action[:3] = best_action
    return action

# Fonction pour exécuter une action et obtenir la récompense et le nouvel état
def execute_action(action, team, client, robot_positions, ball_position):
    # Mettre à jour la position du robot avant d'exécuter l'action
    robot_positions[0][0] = action[0]
    robot_positions[0][1] = action[1]
    robot_positions[0][2] = action[2]
    arrived = False
    try:
        while not arrived:
            arrived=client.robots["blue"][1].goto((action[0], action[1], action[2]), wait=True)
        client.robots["blue"][1].kick(1)
    except:
        print("erreur")
    reward = rewards(ball_position, robot_positions)
    next_state = get_state(robot_positions, ball_position, 1, client)
    return reward, next_state, ball_position, robot_positions

# Fonction pour calculer la récompense
def rewards(position_Balle, robot_positions):
    xball, yball = position_Balle
    xball2,yball2=client.ball
    result = -1
    if(np.around(xball,1)!=np.around(xball2,1) or np.around(yball,1)!=np.around(yball2,1)):
        if(xball>0):
            result=4
        else: 
            result=-4
    elif xball > 1.83/2 and yball < 0.34 and yball > -0.34:
        result=WIN_REWARD
    elif(zoneAutourBalle(position_Balle, robot_positions)):
        result=3
    elif(robot_positions[0][0]>xball):
        result=-3
    elif(xball<-1.83/2 and yball < 0.34 and yball > -0.34):
        result=LOSE_REWARD
    if(client.referee["teams"]["blue"]["robots"]["1"]["penalized"]):
        result=-2
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
    try:
        with open('q_table.pkl', 'wb') as f:
            for episode in range(10):
                # Initialisation de l'état et de la récompense
                client.robots["blue"][1].goto((-0.4582564589942015, -0.00010306401021831266, 6.2796251206407), wait=True)
                ball_position = [0, 0]
                robot_positions = [[robot.position[0], robot.position[1], robot.orientation]]
                state = get_state(robot_positions, ball_position, 1, client)
                reward = 0

                # Boucle d'exécution de l'épisode
                while True:
                    # Sélection de l'action à partir de l'état actuel
                    ball_position = client.ball
                    action = select_action(state, ball_position, robot_positions)
                    # Exécution de l'action et obtention de la récompense et du nouvel état
                    reward, next_state, ball_position2, robot_positions = execute_action(action, "blue", client, robot_positions, ball_position)
                    print(reward)
                    # Mise à jour de la table des Q-values
                    if state not in q_table:
                        q_table[state] = {}
                    if (action[0], action[1], action[2]) not in q_table[state]:
                        q_table[state][(action[0], action[1], action[2])] = 0
                    if next_state not in q_table:
                        q_table[next_state] = {(0.0, 0.0, 0.0):0.0}
                    q_table[state][(action[0], action[1], action[2])] += ALPHA * (reward + GAMMA * max(q_table[next_state].values()) - q_table[state][(action[0], action[1], action[2])])
                    state = next_state

                    # Vérification si l'épisode est terminé
                    ball_position[0] = ball_position2[0]
                    ball_position[1] = ball_position2[1]
                    if reward==8:
                        reward = WIN_REWARD
                        time.sleep(1)
                        break
                    print("Episode {}: reward = {}".format(episode, reward))
                pickle.dump(q_table, f) 
        # Affichage de la récompense obtenue à la fin de l'épisode
        print("Episode {}: reward = {}".format(episode, reward))
    finally:
        save_q_table()