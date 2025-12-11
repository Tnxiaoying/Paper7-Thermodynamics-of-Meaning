import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle

# --- Global Configuration ---
L = 100.0  
T_MAX = 200  
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
    def __init__(self, stype, pos):
        self.type = stype   
        self.pos = np.array(pos, dtype=float)

class Cat:
    def __init__(self, start_pos):
        self.pos = np.array(start_pos, dtype=float)
        self.velocity = np.zeros(2)
        
    def track(self, hamster_pos, hamster_vel):
        # 1. 预测性拦截 (Predictive Interception)
        # 预测时间窗
        prediction_time = 15.0 
        predicted_target = hamster_pos + hamster_vel * prediction_time
        predicted_target = np.clip(predicted_target, 0, L)

        # 2. 移动逻辑
        direction = predicted_target - self.pos
        dist = np.linalg.norm(direction)
        
        if dist > 0:
            target_v = (direction / dist) * 2.5 # 速度较快，产生威胁
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

    def update(self, stimulus, cat_pos=None):
        if not self.alive: return

        # 1. 计算 Ratio
        ratio = 0.0
        if stimulus.type == 'Cat' and cat_pos is not None:
            dist_to_cat = np.linalg.norm(self.pos - cat_pos)
            if dist_to_cat < PhysicsConfig.cat_gaze_radius * 2.0:
                ratio = PhysicsConfig.cat_gaze_radius / (dist_to_cat + 1.0)
        self.ratios.append(ratio)

        # 2. 计算 Uncertainty
        if self.is_heisenberg:
            uncertainty = PhysicsConfig.h_bar_cog * (1 + PhysicsConfig.coupling_lambda * ratio)
        else:
            uncertainty = 0.0

        self.entropies.append(uncertainty)

        # 3. 动力学更新
        v_inertial = self.velocity * PhysicsConfig.friction
        
        # 基础逃逸方向
        vec_to_cat = self.pos - stimulus.pos
        dist = np.linalg.norm(vec_to_cat)
        if dist > 0:
            escape_dir = vec_to_cat / dist
        else:
            escape_dir = np.random.randn(2)
            escape_dir /= np.linalg.norm(escape_dir)
            
        v_bayes = escape_dir * 3.0 # 基础逃逸速度
        
        current_max_speed = 4.0 

        # --- 【实验 3 核心：蛇皮走位 (Snake Burst)】 ---
        if self.is_heisenberg and ratio > 0.5:
            # 爆发倍率
            burst_multiplier = 4.0 
            
            # 计算切线方向 (用于左右横跳)
            tangent_dir = np.array([-escape_dir[1], escape_dir[0]])
            
            # 构造结构化噪音：
            # 主要噪音加在切线方向 (Lateral Noise) -> 导致蛇皮走位
            # 次要噪音加在纵向 (Longitudinal) -> 保持逃逸速度
            
            noise_lat = tangent_dir * np.random.normal(0, uncertainty * 2.0) # 侧向大幅抖动
            noise_long = escape_dir * np.random.normal(0, uncertainty * 0.5) # 纵向小幅抖动
            
            noise = (noise_lat + noise_long) * burst_multiplier
            
            # 逃逸意愿也同步爆发
            v_bayes = v_bayes * burst_multiplier
            
            # 允许突破速度上限
            current_max_speed = 20.0 
        else:
            # 普通模式
            noise = np.random.normal(0, uncertainty, 2)
            current_max_speed = 4.0

        # 融合
        alpha = 0.3
        self.velocity = (1 - alpha) * v_inertial + alpha * v_bayes + noise
        
        # 限速
        speed = np.linalg.norm(self.velocity)
        if speed > current_max_speed:
            self.velocity = (self.velocity / speed) * current_max_speed

        # 移动
        self.pos += self.velocity
        
        # 边界
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

# --- 运行器 ---
def run_experiment(is_heisenberg):
    hamster = Hamster([10, 10], is_heisenberg=is_heisenberg)
    cat = Cat([80, 80])
    stimulus = Stimulus('Cat', cat.pos)
    
    for t in range(T_MAX):
        if not hamster.alive: break
        
        # 必须传入速度，确保猫能预判！
        cat.track(hamster.pos, hamster.velocity)
        stimulus.pos = cat.pos
        
        hamster.update(stimulus, cat.pos)
                
    return hamster, cat

# --- 绘图 ---
def plot_results():
    fig = plt.figure(figsize=(18, 10))
    plt.style.use('dark_background')
    
    # 1. 轨迹：Newtonian (死于预判)
    ax1 = fig.add_subplot(2, 3, 1)
    h_n, c_n = run_experiment(is_heisenberg=False)
    traj_n = np.array(h_n.trajectory)
    ax1.plot(traj_n[:,0], traj_n[:,1], 'gray', linestyle='--', label='Newtonian Path')
    ax1.add_patch(Circle(c_n.pos, PhysicsConfig.cat_gaze_radius, color='red', alpha=0.3))
    ax1.set_title("Newtonian Hamster\n(Linear Escape = Intercepted)")
    ax1.set_xlim(0, L); ax1.set_ylim(0, L); ax1.legend()

    # 2. 轨迹：Heisenberg Snake (战术规避)
    ax2 = fig.add_subplot(2, 3, 2)
    h_h, c_h = run_experiment(is_heisenberg=True)
    traj_h = np.array(h_h.trajectory)
    ax2.plot(traj_h[:,0], traj_h[:,1], 'cyan', linewidth=2, label='Heisenberg (Snake)')
    ax2.add_patch(Circle(c_h.pos, PhysicsConfig.cat_gaze_radius, color='red', alpha=0.3))
    ax2.set_title("Heisenberg Hamster\n(Lateral Noise = Survival)")
    ax2.set_xlim(0, L); ax2.set_ylim(0, L); ax2.legend()

    # 3. 相平面
    ax3 = fig.add_subplot(2, 3, 3)
    ratios = np.array(h_h.ratios)
    entropies = np.array(h_h.entropies)
    if len(ratios) > 0:
        sns.regplot(x=ratios, y=entropies, ax=ax3, scatter_kws={'alpha':0.5, 'color':'yellow'}, line_kws={'color':'red'})
    ax3.set_title("Uncertainty Principle")
    ax3.set_xlabel("Ratio"); ax3.set_ylabel("Entropy")

    # 4. 生存分析
    ax4 = fig.add_subplot(2, 1, 2)
    n_samples = 50
    
    steps_n, steps_h = [], []
    
    for _ in range(n_samples):
        h, _ = run_experiment(False)
        steps_n.append(len(h.trajectory))
        
    for _ in range(n_samples):
        h, _ = run_experiment(True)
        steps_h.append(len(h.trajectory))
        
    time_points = np.arange(T_MAX)
    surv_n = [np.mean(np.array(steps_n) > t) for t in time_points]
    surv_h = [np.mean(np.array(steps_h) > t) for t in time_points]
    
    # 调整绘图顺序，确保灰色线如果不为0能被看见
    ax4.plot(time_points, surv_n, 'gray', linestyle='--', linewidth=2, label='Newtonian (Deterministic)')
    ax4.plot(time_points, surv_h, 'orange', linewidth=3, label='Heisenberg (Free Will)')
    ax4.fill_between(time_points, surv_n, surv_h, color='orange', alpha=0.1)
    
    ax4.set_title("Survival Analysis: Tactical Uncertainty vs Predictive Predator")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_results()
