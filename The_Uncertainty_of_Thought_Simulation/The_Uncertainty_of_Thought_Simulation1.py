import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle

# --- Global Configuration ---
L = 100.0  # Space size
T_MAX = 200  # Max time steps
SEED = 42
np.random.seed(SEED)

class PhysicsConfig:
    dt = 1.0
    friction = 0.90         
    metabolic_rate = 0.5    
    cat_inertia = 0.8       
    cat_gaze_radius = 15.0  
    doom_heat_rate = 5.0    
    doom_cool_rate = 1.0    
    
    # Heisenberg Parameters
    h_bar_cog = 2.0         
    coupling_lambda = 5.0   

class Stimulus:
    def __init__(self, stype, pos, risk_reward=None):
        self.type = stype   
        self.pos = np.array(pos, dtype=float)
        self.risk_reward = risk_reward 

class Cat:
    def __init__(self, start_pos):
        self.pos = np.array(start_pos, dtype=float)
        self.velocity = np.zeros(2)
        
    # --- 【关键修改】猫现在需要知道仓鼠的速度来进行预判 ---
    def track(self, hamster_pos, hamster_vel):
        # 预测时间窗 (Look-ahead)
        prediction_time = 15.0 
        
        # 简单的线性预测：未来位置 = 当前位置 + 速度 * 时间
        predicted_target = hamster_pos + hamster_vel * prediction_time
        
        # 限制预测目标不出界（猫也知道墙的存在）
        predicted_target = np.clip(predicted_target, 0, L)

        # 猫朝向预测点移动 (拦截逻辑)
        direction = predicted_target - self.pos
        dist = np.linalg.norm(direction)
        
        if dist > 0:
            # 稍微提高一点猫的速度 (2.0 -> 2.5) 以体现压迫感
            target_v = (direction / dist) * 2.5 
            alpha = 1.0 - PhysicsConfig.cat_inertia
            self.velocity = alpha * target_v + (1 - alpha) * self.velocity
            self.pos += self.velocity
            
        return self.pos

class Hamster:
    def __init__(self, start_pos, is_heisenberg=True):
        self.pos = np.array(start_pos, dtype=float)
        self.velocity = np.zeros(2)
        self.comfort = 50.0   
        self.doom_bar = 0.0   
        self.is_heisenberg = is_heisenberg 
        self.alive = True
        self.trajectory = [self.pos.copy()]
        self.ratios = []      
        self.entropies = []   

    def bayesian_decision(self, stimulus):
        vec = stimulus.pos - self.pos
        dist = np.linalg.norm(vec)
        if dist == 0: return np.zeros(2)
        direction = vec / dist

        if stimulus.type == 'Cat':
            # 简单的反向逃逸
            return -direction * 3.0 
        
        # (简化掉其他场景，专注实验1的追逃)
        if stimulus.type == 'Food': return direction * 2.0
        return np.zeros(2)

    def update(self, stimulus, cat_pos=None):
        if not self.alive: return

        # 1. Calculate Ratio
        ratio = 0.0
        if stimulus.type == 'Cat' and cat_pos is not None:
            dist_to_cat = np.linalg.norm(self.pos - cat_pos)
            if dist_to_cat < PhysicsConfig.cat_gaze_radius * 2.0:
                ratio = PhysicsConfig.cat_gaze_radius / (dist_to_cat + 1.0)
        self.ratios.append(ratio)

        # 2. Heisenberg Mechanism (Exp 1: Standard Noise vs No Noise)
        if self.is_heisenberg:
            uncertainty = PhysicsConfig.h_bar_cog * (1 + PhysicsConfig.coupling_lambda * ratio)
        else:
            uncertainty = 0.0 # Newtonian Hamster is perfectly deterministic

        self.entropies.append(uncertainty)

        # 3. Dynamic Update
        v_inertial = self.velocity * PhysicsConfig.friction
        v_bayes = self.bayesian_decision(stimulus)
        
        # 标准布朗噪音 (实验1暂不引入蛇皮走位，先看基础自由意志的效果)
        noise = np.random.normal(0, uncertainty, 2)
        
        alpha = 0.3
        self.velocity = (1 - alpha) * v_inertial + alpha * v_bayes + noise
        
        speed = np.linalg.norm(self.velocity)
        if speed > 4.0:
            self.velocity = (self.velocity / speed) * 4.0

        self.pos += self.velocity
        
        # Bounce
        for i in range(2):
            if self.pos[i] < 0 or self.pos[i] > L:
                self.velocity[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 0, L)

        self.trajectory.append(self.pos.copy())

        # 4. State Update
        self.comfort -= PhysicsConfig.metabolic_rate
        if stimulus.type == 'Cat' and cat_pos is not None:
            dist_to_cat = np.linalg.norm(self.pos - cat_pos)
            if dist_to_cat < PhysicsConfig.cat_gaze_radius:
                self.doom_bar += PhysicsConfig.doom_heat_rate
            else:
                self.doom_bar = max(0, self.doom_bar - PhysicsConfig.doom_cool_rate)
            
            if self.doom_bar >= 100.0:
                self.alive = False 

# --- Experiment Runner ---

def run_experiment(scenario, is_heisenberg=True):
    # 统一接口，专注跑 Scenario C (Cat)
    hamster = Hamster([10, 10], is_heisenberg=is_heisenberg)
    cat = Cat([80, 80])
    stimulus = Stimulus('Cat', cat.pos) 
    
    for t in range(T_MAX):
        if not hamster.alive: break
        
        # --- 【关键修复】这里传入仓鼠速度供猫预判 ---
        cat.track(hamster.pos, hamster.velocity)
        stimulus.pos = cat.pos 
        
        hamster.update(stimulus, cat.pos)
                
    return hamster, cat

# --- Plotting Functions ---
def plot_results():
    fig = plt.figure(figsize=(12, 10))
    plt.style.use('dark_background') 
    
    # 1. Trajectory Comparison
    ax1 = fig.add_subplot(2, 2, 1)
    # Run Newtonian (Deterministic)
    h_newton, c_newton = run_experiment('C', is_heisenberg=False)
    traj_n = np.array(h_newton.trajectory)
    ax1.plot(traj_n[:,0], traj_n[:,1], 'gray', linewidth=1, linestyle='--', label='Newtonian Path')
    # Draw Cat Gaze for Newton
    circle_n = Circle(c_newton.pos, PhysicsConfig.cat_gaze_radius, color='red', alpha=0.3)
    ax1.add_patch(circle_n)
    ax1.set_title("Experiment 1a: Newtonian Hamster\n(Predictable = Dead)")
    ax1.set_xlim(0, L); ax1.set_ylim(0, L)
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    # Run Heisenberg (Stochastic)
    h_heis, c_heis = run_experiment('C', is_heisenberg=True)
    traj_h = np.array(h_heis.trajectory)
    ax2.plot(traj_h[:,0], traj_h[:,1], 'cyan', linewidth=2, label='Heisenberg Path')
    # Draw Cat Gaze for Heisenberg
    circle_h = Circle(c_heis.pos, PhysicsConfig.cat_gaze_radius, color='red', alpha=0.3)
    ax2.add_patch(circle_h)
    ax2.set_title("Experiment 1b: Heisenberg Hamster\n(Unpredictable = Survives)")
    ax2.set_xlim(0, L); ax2.set_ylim(0, L)
    ax2.legend()

    # 2. Survival Analysis (The Core Proof)
    ax3 = fig.add_subplot(2, 1, 2)
    
    n_samples = 50
    steps_newton = []
    steps_heisenberg = []
    
    print("Running Survival Simulation (Newtonian)...")
    for _ in range(n_samples):
        h = Hamster([10, 10], is_heisenberg=False) 
        cat = Cat([80, 80])
        stimulus = Stimulus('Cat', cat.pos)
        steps = T_MAX
        for t in range(T_MAX):
            if not h.alive: 
                steps = t
                break
            # --- 【关键修复】循环中也要传入速度 ---
            cat.track(h.pos, h.velocity)
            stimulus.pos = cat.pos
            h.update(stimulus, cat.pos)
        steps_newton.append(steps)
        
    print("Running Survival Simulation (Heisenberg)...")
    for _ in range(n_samples):
        h = Hamster([10, 10], is_heisenberg=True) 
        cat = Cat([80, 80])
        stimulus = Stimulus('Cat', cat.pos)
        steps = T_MAX
        for t in range(T_MAX):
            if not h.alive: 
                steps = t
                break
            # --- 【关键修复】循环中也要传入速度 ---
            cat.track(h.pos, h.velocity)
            stimulus.pos = cat.pos
            h.update(stimulus, cat.pos)
        steps_heisenberg.append(steps)

    # Plot Kaplan-Meier
    time_points = np.arange(T_MAX)
    survival_newton = [np.mean(np.array(steps_newton) > t) for t in time_points]
    survival_heisenberg = [np.mean(np.array(steps_heisenberg) > t) for t in time_points]
    
    ax3.plot(time_points, survival_newton, 'gray', linestyle='--', linewidth=2, label='Newtonian (Deterministic)')
    ax3.plot(time_points, survival_heisenberg, 'orange', linewidth=3, label='Heisenberg (Free Will)')
    ax3.fill_between(time_points, survival_newton, survival_heisenberg, color='orange', alpha=0.1)
    ax3.set_xlabel("Time Steps")
    ax3.set_ylabel("Survival Probability")
    ax3.set_title("Experiment 1 Result: Free Will as a Counter-Prediction Mechanism")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_results()
