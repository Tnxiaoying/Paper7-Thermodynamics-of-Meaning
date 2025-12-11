import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
from scipy.stats import norm

# --- 全局配置 ---
L = 100.0  # 空间大小
T_MAX = 200  # 最大时间步
SEED = 42
np.random.seed(SEED)

class PhysicsConfig:
    dt = 1.0
    friction = 0.90 # 惯性保留系数
    metabolic_rate = 0.5 # 舒适度自然衰减 (饥饿)
    cat_inertia = 0.8 # 猫的惯性 (0~1, 越大越慢)
    cat_gaze_radius = 15.0
    doom_heat_rate = 5.0 # 被盯住时的加热速度
    doom_cool_rate = 1.0 # 逃离时的冷却速度
    
    # Heisenberg Parameters
    h_bar_cog = 2.0 # 认知普朗克常数 (基础底噪)
    coupling_lambda = 5.0 # 耦合系数 (压力->噪音的转化率)

class Stimulus:
    def __init__(self, type, pos, risk_reward=None):
        self.type = type # 'Food', 'Hand', 'Cat'
        self.pos = np.array(pos, dtype=float)
        self.risk_reward = risk_reward # (prob_good, reward, penalty) for Hand

class Cat:
    def __init__(self, start_pos):
        self.pos = np.array(start_pos, dtype=float)
        self.velocity = np.zeros(2)
        
    def track(self, hamster_pos):
        # 猫试图预测并移动，但有巨大的惯性
        # 猫总是看向仓鼠当前位置（带滞后）
        direction = hamster_pos - self.pos
        dist = np.linalg.norm(direction)
        
        if dist > 0:
            target_v = (direction / dist) * 2.0 # 猫的最大速度较慢
            # 惯性更新：v_new = (1-inertia)*v_target + inertia*v_old
            alpha = 1.0 - PhysicsConfig.cat_inertia
            self.velocity = alpha * target_v + (1 - alpha) * self.velocity
            self.pos += self.velocity
            
        return self.pos

class Hamster:
    def __init__(self, start_pos, is_heisenberg=True):
        self.pos = np.array(start_pos, dtype=float)
        self.velocity = np.zeros(2)
        self.comfort = 50.0 # 初始舒适度
        self.doom_bar = 0.0 # 毁灭进度条
        self.is_heisenberg = is_heisenberg # 是否开启测不准机制
        self.alive = True
        self.trajectory = [self.pos.copy()]
        self.ratios = [] # 记录压力值
        self.entropies = [] # 记录瞬时熵(噪音幅度)

    def bayesian_decision(self, stimulus):
        """根据刺激源类型决定目标向量"""
        vec = stimulus.pos - self.pos
        dist = np.linalg.norm(vec)
        if dist == 0: return np.zeros(2)
        direction = vec / dist

        if stimulus.type == 'Food':
            # 场景1: 确定性吸引
            return direction * 2.0 # 全速前进
        
        elif stimulus.type == 'Hand':
            # 场景2: 概率博弈
            # 效用函数: U = p_good * log(C+R) + p_bad * log(C-P)
            # 简化逻辑：如果舒适度低，冒险(靠近)；舒适度高，保守(远离)
            p_good, reward, penalty = stimulus.risk_reward
            
            # 饥饿感驱动冒险
            risk_tolerance = 100.0 / (self.comfort + 1.0) 
            
            if risk_tolerance > 1.5: # 饿了，去吃
                return direction * 1.5
            else: # 饱了，躲避
                return -direction * 1.0

        elif stimulus.type == 'Cat':
            # 场景3: 逃逸
            return -direction * 3.0 # 全力逃跑
            
        return np.zeros(2)

    def update(self, stimulus, cat_pos=None):
        if not self.alive: return

        # 1. 计算观测比 (Ratio)
        ratio = 0.0
        if stimulus.type == 'Cat' and cat_pos is not None:
            dist_to_cat = np.linalg.norm(self.pos - cat_pos)
            if dist_to_cat < PhysicsConfig.cat_gaze_radius * 2.0:
                ratio = PhysicsConfig.cat_gaze_radius / (dist_to_cat + 1.0)
        self.ratios.append(ratio)

        # 2. 计算基础不确定性
        if self.is_heisenberg:
            uncertainty = PhysicsConfig.h_bar_cog * (1 + PhysicsConfig.coupling_lambda * ratio)
        else:
            uncertainty = 0.0 

        self.entropies.append(uncertainty)

        # 3. 动力学更新
        v_inertial = self.velocity * PhysicsConfig.friction
        v_bayes = self.bayesian_decision(stimulus)
        
        # --- 【实验 2 核心修正：超光速爆发 (Super Burst)】 ---
        current_max_speed = 4.0 
        
        if self.is_heisenberg and ratio > 0.5:
            # 只有当压力大时才爆发
            # 设定倍率：确保速度能一步跨出 15.0 的半径
            # 原速度约 3~4，乘以 5 倍左右即可达到 15~20
            burst_multiplier = 5.0 
            
            # 【关键】：不仅噪音变大，逃跑的主观意愿(v_bayes)也要同步变大！
            # 否则就是原地乱抖
            v_bayes_burst = v_bayes * burst_multiplier
            noise = np.random.normal(0, uncertainty, 2) * burst_multiplier
            
            # 速度上限设置为 20.0，确保能一步跳出猫的视野 (R=15)
            current_max_speed = 20.0 
            
            # 临时覆盖 v_bayes
            v_bayes = v_bayes_burst
        else:
            # 普通模式
            noise = np.random.normal(0, uncertainty, 2)
            current_max_speed = 4.0

        # 融合速度
        alpha = 0.3
        self.velocity = (1 - alpha) * v_inertial + alpha * v_bayes + noise
        
        # 应用动态速度限制
        speed = np.linalg.norm(self.velocity)
        if speed > current_max_speed:
            self.velocity = (self.velocity / speed) * current_max_speed

        # 位置更新
        self.pos += self.velocity
        
        # 边界反弹
        for i in range(2):
            if self.pos[i] < 0 or self.pos[i] > L:
                self.velocity[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 0, L)

        self.trajectory.append(self.pos.copy())

        # 4. 状态更新
        self.comfort -= PhysicsConfig.metabolic_rate
        if stimulus.type == 'Cat' and cat_pos is not None:
            dist_to_cat = np.linalg.norm(self.pos - cat_pos)
            if dist_to_cat < PhysicsConfig.cat_gaze_radius:
                self.doom_bar += PhysicsConfig.doom_heat_rate
            else:
                self.doom_bar = max(0, self.doom_bar - PhysicsConfig.doom_cool_rate)
            
            if self.doom_bar >= 100.0:
                self.alive = False

# --- 实验运行器 ---

def run_experiment(scenario):
    hamster = Hamster([10, 10])
    cat = None
    stimulus = None
    
    if scenario == 'A': # Food
        stimulus = Stimulus('Food', [80, 80])
    elif scenario == 'B': # Hand
        stimulus = Stimulus('Hand', [50, 50], risk_reward=(0.5, 30, 30))
    elif scenario == 'C': # Cat
        cat = Cat([80, 80])
        stimulus = Stimulus('Cat', cat.pos) # 初始位置
    
    for t in range(T_MAX):
        if not hamster.alive: break
        
        # 更新外界
        if scenario == 'C':
            cat.track(hamster.pos)
            stimulus.pos = cat.pos # 刺激源是猫
        
        # 更新仓鼠
        hamster.update(stimulus, cat.pos if cat else None)
        
        # 手的周期性出现 (场景B)
        if scenario == 'B':
            if t % 50 < 25: # 手伸进来了
                stimulus.pos = [50, 50]
            else: # 手拿走了
                stimulus.pos = [-100, -100] # 移出屏幕
                
    return hamster, cat

# --- 绘图函数 ---
def plot_results():
    fig = plt.figure(figsize=(18, 10))
    plt.style.use('dark_background')
    
    # 1. 轨迹对比 (A vs C)
    ax1 = fig.add_subplot(2, 3, 1)
    h_a, _ = run_experiment('A')
    traj_a = np.array(h_a.trajectory)
    ax1.plot(traj_a[:,0], traj_a[:,1], 'cyan', linewidth=2, label='Path')
    ax1.scatter([80], [80], c='lime', marker='*', s=200, label='Food')
    ax1.set_title("Scenario A: Deterministic (Food)\nLaminar Flow")
    ax1.set_xlim(0, L); ax1.set_ylim(0, L)
    ax1.legend()

    ax2 = fig.add_subplot(2, 3, 2)
    h_c, c_c = run_experiment('C')
    traj_c = np.array(h_c.trajectory)
    ax2.plot(traj_c[:,0], traj_c[:,1], 'magenta', linewidth=1, label='Hamster (Heisenberg)')
    # 画出猫的最终注视范围
    circle = Circle(c_c.pos, PhysicsConfig.cat_gaze_radius, color='red', alpha=0.3, label='Cat Gaze')
    ax2.add_patch(circle)
    ax2.set_title("Scenario C: Adversarial (Cat)\nChaotic/High Entropy")
    ax2.set_xlim(0, L); ax2.set_ylim(0, L)
    ax2.legend()

    # 2. 相空间 (Ratio vs Entropy) - 场景 C 数据
    ax3 = fig.add_subplot(2, 3, 3)
    ratios = np.array(h_c.ratios)
    entropies = np.array(h_c.entropies)
    # 简单的散点图 + 拟合
    sns.regplot(x=ratios, y=entropies, ax=ax3, scatter_kws={'alpha':0.5, 'color':'yellow'}, line_kws={'color':'red'})
    ax3.set_xlabel("Observation Ratio (Pressure)")
    ax3.set_ylabel("Entropy / Noise Injection (Delta p)")
    ax3.set_title("Phase Portrait: The Uncertainty Principle")

    # 3. 生存曲线 (Kaplan-Meier Simulation)
    ax4 = fig.add_subplot(2, 1, 2)
    
    n_samples = 50
    steps_newton = []
    steps_heisenberg = []
    
    # 跑50次牛顿仓鼠
    for _ in range(n_samples):
        h, _ = run_experiment('C')
        # 强制变成牛顿仓鼠
        h = Hamster([10, 10], is_heisenberg=False) 
        cat = Cat([80, 80])
        stimulus = Stimulus('Cat', cat.pos)
        steps = T_MAX
        for t in range(T_MAX):
            if not h.alive: 
                steps = t
                break
            cat.track(h.pos); stimulus.pos = cat.pos
            h.update(stimulus, cat.pos)
        steps_newton.append(steps)
        
    # 跑50次海森堡仓鼠
    for _ in range(n_samples):
        h, _ = run_experiment('C') # 默认为 Heisenberg
        steps = T_MAX
        for t in range(len(h.trajectory)): # 复用上面的逻辑太慢，这里直接取之前跑的结果逻辑，或者重跑
             # 为了严谨，重写简易循环
             pass 
        # (为了代码简洁，实际上我们直接在上面循环里跑)
        
    # 重新快速模拟一组 Heisenberg
    for _ in range(n_samples):
        h = Hamster([10, 10], is_heisenberg=True) 
        cat = Cat([80, 80])
        stimulus = Stimulus('Cat', cat.pos)
        steps = T_MAX
        for t in range(T_MAX):
            if not h.alive: 
                steps = t
                break
            cat.track(h.pos); stimulus.pos = cat.pos
            h.update(stimulus, cat.pos)
        steps_heisenberg.append(steps)

    # 绘制生存率
    time_points = np.arange(T_MAX)
    survival_newton = [np.mean(np.array(steps_newton) > t) for t in time_points]
    survival_heisenberg = [np.mean(np.array(steps_heisenberg) > t) for t in time_points]
    
    ax4.plot(time_points, survival_newton, 'gray', linestyle='--', label='Newtonian Hamster (Deterministic)')
    ax4.plot(time_points, survival_heisenberg, 'orange', linewidth=3, label='Heisenberg Hamster (Stochastic)')
    ax4.fill_between(time_points, survival_newton, survival_heisenberg, color='orange', alpha=0.2)
    ax4.set_xlabel("Time Steps")
    ax4.set_ylabel("Survival Probability")
    ax4.set_title("Survival Analysis: Free Will as a Survival Mechanism")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_results()
