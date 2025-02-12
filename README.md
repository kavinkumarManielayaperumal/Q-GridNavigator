# 🏴‍☠️ Q-GridNavigator: One Piece RL Adventure  

### **📜 Project Overview**  
Q-GridNavigator is a reinforcement learning project that applies **Q-learning** to guide an agent across a **10x10 grid**, inspired by the **One Piece** world. The agent starts in the **East Blue (bottom-left corner)** and must navigate past **obstacles, enemies, and the Red Line** to reach the **final goal (top-right corner)**.

The project implements **reinforcement learning fundamentals**, including:
- **Q-learning** with an **ε-greedy policy**.
- **Alpha (learning rate)** and **discount factor (gamma)** to optimize long-term rewards.
- **Exploration decay** to balance exploration and exploitation.
- **Custom reward system** for state transitions.

---

### **🌍 The Grid World**
- **Size:** `10x10`
- **Agent's Movements:** `Up`, `Down`, `Left`, `Right`
- **Key Challenge:** A **wall (Red Line)** in the middle restricts movement, forcing strategic decision-making.

#### **📍 Inspired by One Piece Map**
- The **agent starts in the East Blue (bottom-left corner).**
- It must **cross the Red Line** (a single accessible grid in the middle).
- **Dangers include**: Sea monsters, Marines, and obstacles.
- The **goal is to reach the Grand Line (top-right corner).**

---

### **🏆 Reward System**
| Action/State | Reward |
|-------------|--------|
| 🏁 Reaching the Goal | `+20` |
| 💀 Entering a "Hell State" (dangerous zone) | `-5` |
| 🚶 Every move (living penalty) | `-0.01` |
| 🌊 Crossing the Red Line | `+10` (bonus for strategic movement) |

This reward system ensures the agent **avoids unnecessary moves, learns optimal paths, and overcomes obstacles**.

---

### **⚙️ Learning Parameters**
| Parameter | Value |
|-----------|-------|
| Learning Rate (α) | `0.1` |
| Discount Factor (γ) | `0.9` |
| Exploration Strategy | `ε-Greedy` |
| Exploration Decay | `Exponential Decay` |

---

### **🚀 How It Works**
1. The agent **starts in the bottom-left corner (East Blue).**
2. It **learns optimal paths** by trial and error.
3. Encounters **obstacles, sea monsters, Marines**.
4. **Crosses the Red Line through a narrow passage**.
5. **Reaches the goal (Grand Line)** while minimizing penalties.
6. Over time, the Q-table **converges** to an optimal navigation strategy.

---

### **📦 Installation & Setup**
1. Clone this repository:
   ```bash
   git clone https://github.com/kavinkumarManielayaperumal/Q-GridNavigator.git
   cd Q-GridNavigator
🎮 Visualization with Pygame
The grid-based environment is visualized using Pygame.
The agent moves step-by-step, updating its Q-values.
Each tile represents different states (goal, danger, normal).
The Red Line is marked as a special grid to cross.
📊 Results & Visualization
The Q-table updates over time to reflect the optimal policy.
The agent progressively gets smarter, avoiding penalties.
The path taken by the agent is displayed in real-time.
🎯 Future Improvements
Introduce multi-agent systems (Straw Hat Crew AI).
Add deep Q-learning for more complex environments.
Implement more complex enemy AI that adapts to the agent.
🖊️ Author
Developed by @kavinkumarManielayaperumal
Inspired by One Piece and Reinforcement Learning.
If you like it, ⭐ this repo!