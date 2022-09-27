
### -=-=-=- Reinforment learning DQN -=-=-=- ###
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# Enviroment init

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## -=-=-=- Define DQN model -=-=-=- ##
# Maak een policy met: pi* = argmax Q*, waarbij het neural netwerk getrained gaat worden op de Q*
# Voor het train updating regel, we we’ll use a fact that every Q function for some policy obeys the Bellman equation
# Als elke functie voor een policy de bellman's equation behoorzaamd, is het verschil tussen je input en de bellman equation
# is de temporal difference error. Deze error gaan we met de mean-squarred error minimaliseren.

# -=-=-=- Q-netwerk -=-=-=-
class DQN(nn.Module):
    
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
    
# -=-=-=- Training -=-=-=- ("alexa, play: Eye-of-the-tiger")

# BATCH_SIZE = Hoeveelheid agents. 
# De kans dat het netwerk een random actie doet, begint bij eps_start. En zal naarmate hij meer weet naar eps_end gaan.
# eps_decay houdt de rate bij hoe snel die naar eps_end gaat.
# gamma en target update zijn voor andere functies.
BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05 
EPS_DECAY = 200
TARGET_UPDATE = 10

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render().transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return np.resize(screen).unsqueeze(0)

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
policy_net = DQN(screen_height, screen_width, 4).to(device)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)

