"""
통합 실험 워크플로우
1단계: 하이퍼파라미터 튜닝
2단계: 최적 파라미터로 시나리오 비교 실험
"""

import numpy as np
import torch
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List
from traffic_env import TrafficEnvironment, FixedTimeController
from dqn_agent import DQNAgent, DoubleDQNAgent


def find_best_hyperparameters(tuning_results_path: str) -> Dict:
    """
    하이퍼파라미터 튜닝 결과에서 최적 파라미터 찾기
    
    Args:
        tuning_results_path: 튜닝 결과 JSON 파일 경로
        
    Returns:
        최적 하이퍼파라미터 딕셔너리
    """
    print("\n" + "="*60)
    print("🔍 최적 하이퍼파라미터 선택")
    print("="*60)
    
    with open(tuning_results_path, 'r') as f:
        results = json.load(f)
    
    # 각 실험에서 최적값 찾기
    best_params = {
        'state_dim': 7,
        'action_dim': 2
    }
    
    # Learning Rate에서 최고 성능
    lr_results = results['Learning_Rate']
    best_lr = max(lr_results.items(), key=lambda x: x[1]['avg_reward'])
    best_params['learning_rate'] = float(best_lr[0].split('_')[1])
    print(f"✅ 최적 Learning Rate: {best_params['learning_rate']}")
    print(f"   평균 Reward: {best_lr[1]['avg_reward']:.2f}")
    print(f"   표준편차: {best_lr[1]['std_reward']:.2f}")
    
    # Discount Factor에서 최고 성능
    gamma_results = results['Discount_Factor']
    best_gamma = max(gamma_results.items(), key=lambda x: x[1]['avg_reward'])
    best_params['gamma'] = float(best_gamma[0].split('_')[1])
    print(f"\n✅ 최적 Discount Factor: {best_params['gamma']}")
    print(f"   평균 Reward: {best_gamma[1]['avg_reward']:.2f}")
    print(f"   표준편차: {best_gamma[1]['std_reward']:.2f}")
    
    # Batch Size에서 최고 성능
    batch_results = results['Batch_Size']
    best_batch = max(batch_results.items(), key=lambda x: x[1]['avg_reward'])
    best_params['batch_size'] = int(best_batch[0].split('_')[1])
    print(f"\n✅ 최적 Batch Size: {best_params['batch_size']}")
    print(f"   평균 Reward: {best_batch[1]['avg_reward']:.2f}")
    print(f"   표준편차: {best_batch[1]['std_reward']:.2f}")
    
    # Buffer Size에서 최고 성능
    buffer_results = results['Buffer_Size']
    best_buffer = max(buffer_results.items(), key=lambda x: x[1]['avg_reward'])
    best_params['buffer_capacity'] = int(best_buffer[0].split('_')[1])
    print(f"\n✅ 최적 Buffer Size: {best_params['buffer_capacity']}")
    print(f"   평균 Reward: {best_buffer[1]['avg_reward']:.2f}")
    print(f"   표준편차: {best_buffer[1]['std_reward']:.2f}")
    
    # 기타 고정 파라미터
    best_params['epsilon_start'] = 1.0
    best_params['epsilon_end'] = 0.01
    best_params['epsilon_decay'] = 0.995
    best_params['target_update_freq'] = 100
    
    print("\n" + "="*60)
    print("📋 최종 선택된 하이퍼파라미터")
    print("="*60)
    for key, value in best_params.items():
        print(f"{key:25s}: {value}")
    print("="*60)
    
    return best_params


class ExperimentLogger:
    """실험 로그 관리 클래스"""
    
    def __init__(self, log_path: str = './results/experiment_log.txt'):
        self.log_path = log_path
        self.start_time = datetime.now()
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else '.', exist_ok=True)
        
        # 로그 파일 초기화
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("강화학습 교통 신호등 제어 실험 로그\n")
            f.write("="*80 + "\n")
            f.write(f"실험 시작 시간: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
    
    def log(self, message: str, level: str = "INFO"):
        """로그 메시지 기록"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        # 콘솔에도 출력
        print(log_entry.strip())
    
    def log_section(self, title: str):
        """섹션 헤더 기록"""
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"{title}\n")
            f.write("="*80 + "\n")
        print(f"\n{'='*80}\n{title}\n{'='*80}")
    
    def log_hyperparameters(self, params: Dict):
        """하이퍼파라미터 기록"""
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write("\n[하이퍼파라미터 설정]\n")
            f.write("-"*80 + "\n")
            for key, value in params.items():
                f.write(f"  {key:25s}: {value}\n")
            f.write("-"*80 + "\n\n")
    
    def log_scenario_start(self, scenario: str, algorithm: str, num_episodes: int):
        """시나리오 학습 시작 기록"""
        self.log(f"시나리오 '{scenario}' - {algorithm} 학습 시작 (에피소드: {num_episodes})")
    
    def log_scenario_result(self, scenario: str, algorithm: str, results: Dict):
        """시나리오 결과 기록"""
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n[시나리오: {scenario} - {algorithm} 결과]\n")
            f.write("-"*80 + "\n")
            f.write(f"  평균 Reward: {np.mean(results.get('episode_rewards', [])):.2f}\n")
            f.write(f"  평균 대기시간: {np.mean(results.get('avg_waiting_times', [])):.2f}초\n")
            f.write(f"  평균 처리 차량: {np.mean(results.get('total_vehicles_passed', [])):.1f}대\n")
            if 'max_queue_lengths' in results:
                f.write(f"  평균 최대 대기: {np.mean(results.get('max_queue_lengths', [])):.1f}대\n")
            f.write("-"*80 + "\n\n")
    
    def log_training_progress(self, scenario: str, algorithm: str, episode: int, 
                             total_episodes: int, avg_reward: float, avg_waiting: float, epsilon: float):
        """학습 진행 상황 기록"""
        if episode % 500 == 0 or episode == total_episodes - 1:
            self.log(f"{scenario} - {algorithm}: Episode {episode+1}/{total_episodes} | "
                    f"Reward: {avg_reward:.2f} | 대기시간: {avg_waiting:.2f}초 | ε: {epsilon:.4f}")
    
    def log_final_summary(self, summary_results: Dict):
        """최종 요약 기록"""
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("최종 실험 결과 요약\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"하이퍼파라미터 소스: {summary_results.get('params_source', 'unknown').upper()}\n")
            if 'hyperparameters' in summary_results:
                f.write("\n[사용된 하이퍼파라미터]\n")
                for key, value in summary_results['hyperparameters'].items():
                    f.write(f"  {key:25s}: {value}\n")
            
            f.write("\n[시나리오별 성능 비교]\n")
            f.write("-"*80 + "\n")
            f.write(f"{'시나리오':<20} {'알고리즘':<20} {'평균 대기시간':>15} {'평균 Reward':>15} {'개선율':>12}\n")
            f.write("-"*80 + "\n")
            
            for scenario, data in summary_results['scenarios'].items():
                baseline_wait = data['baseline']['avg_waiting_time']
                for algo in ['dqn', 'ddqn', 'baseline']:
                    wait_time = data[algo]['avg_waiting_time']
                    reward = data[algo]['avg_reward']
                    improvement = ((baseline_wait - wait_time) / baseline_wait * 100) if algo != 'baseline' else 0.0
                    
                    algo_name = {
                        'dqn': 'DQN',
                        'ddqn': 'Double DQN',
                        'baseline': 'Baseline'
                    }[algo]
                    
                    f.write(f"{scenario:<20} {algo_name:<20} {wait_time:>15.2f}초 {reward:>15.2f} {improvement:>11.2f}%\n")
            
            f.write("-"*80 + "\n")
    
    def log_graph_generation(self, graph_name: str):
        """그래프 생성 기록"""
        self.log(f"그래프 생성 완료: {graph_name}")
    
    def log_completion(self):
        """실험 완료 기록"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("실험 완료\n")
            f.write("="*80 + "\n")
            f.write(f"시작 시간: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"총 소요 시간: {duration}\n")
            f.write("="*80 + "\n")
        
        self.log(f"실험 완료! 총 소요 시간: {duration}")


def load_default_hyperparameters() -> Dict:
    """
    하이퍼파라미터 튜닝 결과가 없을 때 기본값 사용
    
    Returns:
        기본 하이퍼파라미터
    """
    print("\n⚠️  하이퍼파라미터 튜닝 결과를 찾을 수 없습니다.")
    print("   기본 파라미터를 사용합니다.\n")
    
    return {
        'state_dim': 7,
        'action_dim': 2,
        'learning_rate': 0.001,
        'gamma': 0.95,
        'batch_size': 64,
        'buffer_capacity': 10000,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'target_update_freq': 100
    }


def train_agent_with_params(
    agent,
    env: TrafficEnvironment,
    num_episodes: int,
    scenario: str,
    save_dir: str,
    logger: ExperimentLogger = None
) -> Dict:
    """최적 파라미터로 에이전트 학습"""
    from tqdm import tqdm
    
    os.makedirs(save_dir, exist_ok=True)
    env.set_scenario(scenario)
    
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'avg_waiting_times': [],
        'total_vehicles_passed': [],
        'losses': [],
        'epsilons': [],
        'max_queue_lengths': []
    }
    
    algorithm_name = "DQN" if isinstance(agent, DQNAgent) and not isinstance(agent, DoubleDQNAgent) else "Double DQN"
    
    print(f"\n🚦 학습 시작: {scenario} 시나리오")
    print(f"   총 에피소드: {num_episodes}")
    
    for episode in tqdm(range(num_episodes), desc=f"Training {scenario}"):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        step_count = 0
        
        done = False
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()
            
            episode_reward += reward
            episode_loss += loss
            step_count += 1
            state = next_state
        
        agent.decay_epsilon()
        
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(step_count)
        history['avg_waiting_times'].append(info['avg_waiting_time'])
        history['total_vehicles_passed'].append(info['total_vehicles_passed'])
        history['losses'].append(episode_loss / step_count if step_count > 0 else 0)
        history['epsilons'].append(agent.epsilon)
        history['max_queue_lengths'].append(info['max_queue_length'])
        
        # 주기적 출력 및 로그
        if (episode + 1) % 500 == 0:
            recent_rewards = history['episode_rewards'][-100:]
            recent_waiting = history['avg_waiting_times'][-100:]
            print(f"\n   Episode {episode+1}/{num_episodes}")
            print(f"   평균 Reward: {np.mean(recent_rewards):.2f}")
            print(f"   평균 대기시간: {np.mean(recent_waiting):.2f}초")
            print(f"   ε: {agent.epsilon:.4f}")
            
            if logger:
                logger.log_training_progress(
                    scenario, algorithm_name, episode, num_episodes,
                    np.mean(recent_rewards), np.mean(recent_waiting), agent.epsilon
                )
    
    # 최종 모델 저장
    final_path = os.path.join(save_dir, f'agent_{scenario}_optimized.pt')
    agent.save(final_path)
    print(f"✅ 모델 저장: {final_path}")
    if logger:
        logger.log(f"모델 저장 완료: {final_path}")
    
    return history


def evaluate_agent_optimized(
    agent,
    env: TrafficEnvironment,
    num_episodes: int,
    scenario: str
) -> Dict:
    """최적화된 에이전트 평가"""
    env.set_scenario(scenario)
    
    results = {
        'episode_rewards': [],
        'avg_waiting_times': [],
        'total_vehicles_passed': [],
        'max_queue_lengths': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        done = False
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
        
        results['episode_rewards'].append(episode_reward)
        results['avg_waiting_times'].append(info['avg_waiting_time'])
        results['total_vehicles_passed'].append(info['total_vehicles_passed'])
        results['max_queue_lengths'].append(info['max_queue_length'])
    
    return results


def plot_training_curves(history: Dict, save_path: str = None, window_size: int = 100):
    """학습 곡선 시각화 (이동 평균선 포함)"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = np.arange(1, len(history['episode_rewards']) + 1)
    
    # 이동 평균 계산 함수
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # Episode Rewards
    axes[0, 0].plot(episodes, history['episode_rewards'], alpha=0.3, color='blue', label='Raw')
    if len(history['episode_rewards']) >= window_size:
        ma_rewards = moving_average(history['episode_rewards'], window_size)
        ma_episodes = np.arange(window_size, len(history['episode_rewards']) + 1)
        axes[0, 0].plot(ma_episodes, ma_rewards, color='blue', linewidth=2, label=f'MA({window_size})')
    axes[0, 0].set_title('에피소드별 Reward', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('에피소드')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Average Waiting Time
    axes[0, 1].plot(episodes, history['avg_waiting_times'], alpha=0.3, color='red', label='Raw')
    if len(history['avg_waiting_times']) >= window_size:
        ma_waiting = moving_average(history['avg_waiting_times'], window_size)
        ma_episodes = np.arange(window_size, len(history['avg_waiting_times']) + 1)
        axes[0, 1].plot(ma_episodes, ma_waiting, color='red', linewidth=2, label=f'MA({window_size})')
    axes[0, 1].set_title('에피소드별 평균 대기시간', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('에피소드')
    axes[0, 1].set_ylabel('대기시간 (초)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss
    axes[1, 0].plot(episodes, history['losses'], alpha=0.3, color='green', label='Raw')
    if len(history['losses']) >= window_size:
        ma_losses = moving_average(history['losses'], window_size)
        ma_episodes = np.arange(window_size, len(history['losses']) + 1)
        axes[1, 0].plot(ma_episodes, ma_losses, color='green', linewidth=2, label=f'MA({window_size})')
    axes[1, 0].set_title('학습 Loss', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('에피소드')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Epsilon
    axes[1, 1].plot(episodes, history['epsilons'], color='purple', linewidth=2)
    axes[1, 1].set_title('탐험률 (ε)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('에피소드')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 그래프 저장: {save_path}")
    
    plt.close()


def plot_scenario_comparison(summary_results: Dict, save_path: str = None):
    """시나리오별 알고리즘 성능 비교 그래프"""
    scenarios = list(summary_results['scenarios'].keys())
    algorithms = ['dqn', 'ddqn', 'baseline']
    algo_labels = ['DQN', 'Double DQN', 'Baseline']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # 데이터 준비
    waiting_times = {algo: [] for algo in algorithms}
    rewards = {algo: [] for algo in algorithms}
    
    for scenario in scenarios:
        for algo in algorithms:
            waiting_times[algo].append(summary_results['scenarios'][scenario][algo]['avg_waiting_time'])
            rewards[algo].append(summary_results['scenarios'][scenario][algo]['avg_reward'])
    
    # 그래프 생성
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(scenarios))
    width = 0.25
    
    # 대기시간 비교
    for i, (algo, label) in enumerate(zip(algorithms, algo_labels)):
        axes[0].bar(x + i*width, waiting_times[algo], width, label=label, color=colors[i], alpha=0.8)
    
    axes[0].set_xlabel('시나리오', fontsize=12)
    axes[0].set_ylabel('평균 대기시간 (초)', fontsize=12)
    axes[0].set_title('시나리오별 평균 대기시간 비교', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=15, ha='right')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Reward 비교
    for i, (algo, label) in enumerate(zip(algorithms, algo_labels)):
        axes[1].bar(x + i*width, rewards[algo], width, label=label, color=colors[i], alpha=0.8)
    
    axes[1].set_xlabel('시나리오', fontsize=12)
    axes[1].set_ylabel('평균 Reward', fontsize=12)
    axes[1].set_title('시나리오별 평균 Reward 비교', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=15, ha='right')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 그래프 저장: {save_path}")
    
    plt.close()


def plot_improvement_comparison(summary_results: Dict, save_path: str = None):
    """알고리즘별 개선율 비교 그래프"""
    scenarios = list(summary_results['scenarios'].keys())
    algorithms = ['dqn', 'ddqn']
    algo_labels = ['DQN', 'Double DQN']
    colors = ['#2E86AB', '#A23B72']
    
    # 개선율 계산
    improvements = {algo: [] for algo in algorithms}
    
    for scenario in scenarios:
        baseline_wait = summary_results['scenarios'][scenario]['baseline']['avg_waiting_time']
        for algo in algorithms:
            algo_wait = summary_results['scenarios'][scenario][algo]['avg_waiting_time']
            improvement = ((baseline_wait - algo_wait) / baseline_wait) * 100
            improvements[algo].append(improvement)
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    for i, (algo, label) in enumerate(zip(algorithms, algo_labels)):
        bars = ax.bar(x + i*width, improvements[algo], width, label=label, color=colors[i], alpha=0.8)
        # 값 표시
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('시나리오', fontsize=12)
    ax.set_ylabel('개선율 (%)', fontsize=12)
    ax.set_title('Baseline 대비 대기시간 개선율', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 그래프 저장: {save_path}")
    
    plt.close()


def plot_heatmap_comparison(summary_results: Dict, save_path: str = None):
    """시나리오별 알고리즘 성능 히트맵"""
    scenarios = list(summary_results['scenarios'].keys())
    algorithms = ['baseline', 'dqn', 'ddqn']
    algo_labels = ['Baseline', 'DQN', 'Double DQN']
    
    # 대기시간 데이터 준비
    waiting_matrix = []
    for scenario in scenarios:
        row = []
        for algo in algorithms:
            row.append(summary_results['scenarios'][scenario][algo]['avg_waiting_time'])
        waiting_matrix.append(row)
    
    waiting_matrix = np.array(waiting_matrix)
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(waiting_matrix, cmap='RdYlGn_r', aspect='auto')
    
    # 축 설정
    ax.set_xticks(np.arange(len(algo_labels)))
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_xticklabels(algo_labels)
    ax.set_yticklabels([s.replace('_', ' ').title() for s in scenarios])
    
    # 값 표시
    for i in range(len(scenarios)):
        for j in range(len(algorithms)):
            text = ax.text(j, i, f'{waiting_matrix[i, j]:.1f}초',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('시나리오별 알고리즘 성능 비교 (평균 대기시간)', fontsize=14, fontweight='bold', pad=20)
    
    # 컬러바
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('평균 대기시간 (초)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 그래프 저장: {save_path}")
    
    plt.close()


def plot_comprehensive_dashboard(summary_results: Dict, save_path: str = None):
    """종합 대시보드 그래프"""
    scenarios = list(summary_results['scenarios'].keys())
    algorithms = ['dqn', 'ddqn', 'baseline']
    algo_labels = ['DQN', 'Double DQN', 'Baseline']
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. 시나리오별 대기시간 비교 (바 차트)
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(scenarios))
    width = 0.25
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, (algo, label) in enumerate(zip(algorithms, algo_labels)):
        waiting_times = [summary_results['scenarios'][s][algo]['avg_waiting_time'] for s in scenarios]
        ax1.bar(x + i*width, waiting_times, width, label=label, color=colors[i], alpha=0.8)
    
    ax1.set_xlabel('시나리오', fontsize=11)
    ax1.set_ylabel('평균 대기시간 (초)', fontsize=11)
    ax1.set_title('시나리오별 평균 대기시간 비교', fontsize=12, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=15, ha='right', fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. 시나리오별 Reward 비교
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (algo, label) in enumerate(zip(algorithms, algo_labels)):
        rewards = [summary_results['scenarios'][s][algo]['avg_reward'] for s in scenarios]
        ax2.bar(x + i*width, rewards, width, label=label, color=colors[i], alpha=0.8)
    
    ax2.set_xlabel('시나리오', fontsize=11)
    ax2.set_ylabel('평균 Reward', fontsize=11)
    ax2.set_title('시나리오별 평균 Reward 비교', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=15, ha='right', fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 개선율 비교
    ax3 = fig.add_subplot(gs[1, 0])
    improvements = {'dqn': [], 'ddqn': []}
    for scenario in scenarios:
        baseline_wait = summary_results['scenarios'][scenario]['baseline']['avg_waiting_time']
        for algo in ['dqn', 'ddqn']:
            algo_wait = summary_results['scenarios'][scenario][algo]['avg_waiting_time']
            improvement = ((baseline_wait - algo_wait) / baseline_wait) * 100
            improvements[algo].append(improvement)
    
    x_imp = np.arange(len(scenarios))
    width_imp = 0.35
    for i, (algo, label) in enumerate(zip(['dqn', 'ddqn'], ['DQN', 'Double DQN'])):
        bars = ax3.bar(x_imp + i*width_imp, improvements[algo], width_imp, 
                       label=label, color=colors[i], alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax3.set_xlabel('시나리오', fontsize=11)
    ax3.set_ylabel('개선율 (%)', fontsize=11)
    ax3.set_title('Baseline 대비 대기시간 개선율', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_imp + width_imp/2)
    ax3.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=15, ha='right', fontsize=9)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # 4. 히트맵
    ax4 = fig.add_subplot(gs[1, 1])
    waiting_matrix = []
    for scenario in scenarios:
        row = []
        for algo in algorithms:
            row.append(summary_results['scenarios'][scenario][algo]['avg_waiting_time'])
        waiting_matrix.append(row)
    waiting_matrix = np.array(waiting_matrix)
    
    im = ax4.imshow(waiting_matrix, cmap='RdYlGn_r', aspect='auto')
    ax4.set_xticks(np.arange(len(algo_labels)))
    ax4.set_yticks(np.arange(len(scenarios)))
    ax4.set_xticklabels(algo_labels)
    ax4.set_yticklabels([s.replace('_', ' ').title() for s in scenarios], fontsize=9)
    
    for i in range(len(scenarios)):
        for j in range(len(algorithms)):
            ax4.text(j, i, f'{waiting_matrix[i, j]:.1f}',
                    ha="center", va="center", color="black", fontweight='bold', fontsize=9)
    
    ax4.set_title('시나리오별 알고리즘 성능 히트맵', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax4, label='평균 대기시간 (초)')
    
    # 5. 알고리즘별 평균 성능 (전체 시나리오 평균)
    ax5 = fig.add_subplot(gs[2, :])
    algo_avg_waiting = {}
    algo_std_waiting = {}
    
    for algo in algorithms:
        waits = [summary_results['scenarios'][s][algo]['avg_waiting_time'] for s in scenarios]
        algo_avg_waiting[algo] = np.mean(waits)
        algo_std_waiting[algo] = np.std(waits)
    
    algo_names = [algo_labels[algorithms.index(a)] for a in algorithms]
    avg_values = [algo_avg_waiting[a] for a in algorithms]
    std_values = [algo_std_waiting[a] for a in algorithms]
    
    bars = ax5.bar(algo_names, avg_values, yerr=std_values, capsize=5, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for i, (bar, avg, std) in enumerate(zip(bars, avg_values, std_values)):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + std + 0.5,
                f'{avg:.2f}±{std:.2f}초',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax5.set_ylabel('평균 대기시간 (초)', fontsize=11)
    ax5.set_title('전체 시나리오 평균 성능 비교 (평균 ± 표준편차)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 전체 제목
    fig.suptitle('강화학습 교통 신호등 제어 - 종합 성능 분석', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 그래프 저장: {save_path}")
    
    plt.close()


def compare_with_baseline(
    env: TrafficEnvironment,
    scenario: str,
    num_episodes: int = 100
) -> Dict:
    """Baseline과 비교"""
    env.set_scenario(scenario)
    controller = FixedTimeController(cycle_time=30)
    
    results = {
        'episode_rewards': [],
        'avg_waiting_times': [],
        'total_vehicles_passed': [],
        'max_queue_lengths': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        controller.reset()
        episode_reward = 0
        
        done = False
        while not done:
            action = controller.get_action(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
        
        results['episode_rewards'].append(episode_reward)
        results['avg_waiting_times'].append(info['avg_waiting_time'])
        results['total_vehicles_passed'].append(info['total_vehicles_passed'])
        results['max_queue_lengths'].append(info['max_queue_length'])
    
    return results


def run_integrated_experiments(
    use_tuned_params: bool = True,
    tuning_results_path: str = './results/hyperparameter_tuning_results.json',
    num_train_episodes: int = 2000,
    num_eval_episodes: int = 100
):
    """
    통합 실험 실행
    
    Args:
        use_tuned_params: 튜닝된 파라미터 사용 여부
        tuning_results_path: 튜닝 결과 파일 경로
        num_train_episodes: 학습 에피소드 수
        num_eval_episodes: 평가 에피소드 수
    """
    # 로거 초기화
    logger = ExperimentLogger('./results/experiment_log.txt')
    
    logger.log_section("통합 실험 워크플로우 시작")
    logger.log(f"학습 에피소드: {num_train_episodes}, 평가 에피소드: {num_eval_episodes}")
    
    print("\n" + "="*70)
    print("🚀 통합 실험 워크플로우 시작")
    print("="*70)
    
    # 1단계: 하이퍼파라미터 로드
    logger.log_section("1단계: 하이퍼파라미터 로드")
    if use_tuned_params and os.path.exists(tuning_results_path):
        logger.log(f"튜닝된 파라미터 사용: {tuning_results_path}")
        best_params = find_best_hyperparameters(tuning_results_path)
        params_source = "tuned"
    else:
        logger.log("기본 파라미터 사용")
        best_params = load_default_hyperparameters()
        params_source = "default"
    
    logger.log_hyperparameters(best_params)
    
    # 환경 생성
    env = TrafficEnvironment()
    logger.log("환경 생성 완료")
    
    # 시나리오 리스트
    scenarios = ['normal', 'morning_rush', 'evening_rush', 'congestion', 'night']
    logger.log(f"테스트 시나리오: {', '.join(scenarios)}")
    
    # 전체 결과 저장
    all_results = {
        'hyperparameters': best_params,
        'params_source': params_source,
        'scenarios': {}
    }
    
    # 2단계: 각 시나리오에서 실험
    logger.log_section("2단계: 시나리오별 실험")
    for scenario in scenarios:
        logger.log_section(f"시나리오: {scenario}")
        print("\n" + "="*70)
        print(f"📍 시나리오: {scenario}")
        print("="*70)
        
        scenario_results = {}
        
        # DQN 학습
        logger.log_scenario_start(scenario, "DQN", num_train_episodes)
        print("\n[1/4] DQN 에이전트 학습 (최적 파라미터)")
        dqn_agent = DQNAgent(**best_params)
        dqn_history = train_agent_with_params(
            dqn_agent, env, num_train_episodes, scenario,
            f'./models/optimized/dqn_{scenario}',
            logger=logger
        )
        scenario_results['dqn_training'] = dqn_history
        logger.log(f"DQN 학습 완료 - 최종 Reward: {np.mean(dqn_history['episode_rewards'][-100:]):.2f}, "
                  f"최종 대기시간: {np.mean(dqn_history['avg_waiting_times'][-100:]):.2f}초")
        
        # DQN 평가
        logger.log(f"DQN 평가 시작 ({num_eval_episodes} 에피소드)")
        print(f"\n[2/4] DQN 평가 ({num_eval_episodes} 에피소드)")
        dqn_eval = evaluate_agent_optimized(
            dqn_agent, env, num_eval_episodes, scenario
        )
        scenario_results['dqn_eval'] = dqn_eval
        logger.log_scenario_result(scenario, "DQN", dqn_eval)
        
        # Double DQN 학습
        logger.log_scenario_start(scenario, "Double DQN", num_train_episodes)
        print("\n[3/4] Double DQN 에이전트 학습 (최적 파라미터)")
        ddqn_agent = DoubleDQNAgent(**best_params)
        ddqn_history = train_agent_with_params(
            ddqn_agent, env, num_train_episodes, scenario,
            f'./models/optimized/ddqn_{scenario}',
            logger=logger
        )
        scenario_results['ddqn_training'] = ddqn_history
        logger.log(f"Double DQN 학습 완료 - 최종 Reward: {np.mean(ddqn_history['episode_rewards'][-100:]):.2f}, "
                  f"최종 대기시간: {np.mean(ddqn_history['avg_waiting_times'][-100:]):.2f}초")
        
        # Double DQN 평가
        logger.log(f"Double DQN 평가 시작 ({num_eval_episodes} 에피소드)")
        print(f"\n[4/4] Double DQN 평가 ({num_eval_episodes} 에피소드)")
        ddqn_eval = evaluate_agent_optimized(
            ddqn_agent, env, num_eval_episodes, scenario
        )
        scenario_results['ddqn_eval'] = ddqn_eval
        logger.log_scenario_result(scenario, "Double DQN", ddqn_eval)
        
        # Baseline 평가
        logger.log(f"Baseline 평가 시작 ({num_eval_episodes} 에피소드)")
        print(f"\n[Baseline] 고정 주기 신호등 평가")
        baseline_eval = compare_with_baseline(env, scenario, num_eval_episodes)
        scenario_results['baseline_eval'] = baseline_eval
        logger.log_scenario_result(scenario, "Baseline", baseline_eval)
        
        # 결과 요약 출력
        print("\n" + "-"*70)
        print(f"📊 {scenario} 시나리오 결과 요약")
        print("-"*70)
        print(f"{'알고리즘':<20} {'평균 Reward':>15} {'평균 대기시간':>15}")
        print("-"*70)
        print(f"{'DQN (최적화)':<20} {np.mean(dqn_eval['episode_rewards']):>15.2f} "
              f"{np.mean(dqn_eval['avg_waiting_times']):>15.2f}초")
        print(f"{'Double DQN (최적화)':<20} {np.mean(ddqn_eval['episode_rewards']):>15.2f} "
              f"{np.mean(ddqn_eval['avg_waiting_times']):>15.2f}초")
        print(f"{'Baseline (고정 30초)':<20} {np.mean(baseline_eval['episode_rewards']):>15.2f} "
              f"{np.mean(baseline_eval['avg_waiting_times']):>15.2f}초")
        print("-"*70)
        
        # 개선율 계산
        baseline_waiting = np.mean(baseline_eval['avg_waiting_times'])
        dqn_waiting = np.mean(dqn_eval['avg_waiting_times'])
        ddqn_waiting = np.mean(ddqn_eval['avg_waiting_times'])
        
        dqn_improvement = ((baseline_waiting - dqn_waiting) / baseline_waiting) * 100
        ddqn_improvement = ((baseline_waiting - ddqn_waiting) / baseline_waiting) * 100
        
        print(f"\n💡 대기시간 개선율:")
        print(f"   DQN: {dqn_improvement:>6.2f}%")
        print(f"   Double DQN: {ddqn_improvement:>6.2f}%")
        
        # 학습 곡선 시각화
        logger.log("학습 곡선 그래프 생성 중...")
        print(f"\n📊 학습 곡선 그래프 생성 중...")
        plot_training_curves(
            dqn_history,
            f'./results/plots/dqn_{scenario}_training.png'
        )
        logger.log_graph_generation(f'dqn_{scenario}_training.png')
        plot_training_curves(
            ddqn_history,
            f'./results/plots/ddqn_{scenario}_training.png'
        )
        logger.log_graph_generation(f'ddqn_{scenario}_training.png')
        
        all_results['scenarios'][scenario] = scenario_results
    
    # 3단계: 결과 저장
    logger.log_section("3단계: 결과 저장")
    print("\n" + "="*70)
    print("💾 결과 저장")
    print("="*70)
    
    # JSON 저장 (history 제외한 요약만)
    summary_results = {
        'hyperparameters': all_results['hyperparameters'],
        'params_source': all_results['params_source'],
        'scenarios': {}
    }
    
    for scenario, data in all_results['scenarios'].items():
        summary_results['scenarios'][scenario] = {
            'dqn': {
                'avg_reward': float(np.mean(data['dqn_eval']['episode_rewards'])),
                'avg_waiting_time': float(np.mean(data['dqn_eval']['avg_waiting_times'])),
                'avg_vehicles_passed': float(np.mean(data['dqn_eval']['total_vehicles_passed']))
            },
            'ddqn': {
                'avg_reward': float(np.mean(data['ddqn_eval']['episode_rewards'])),
                'avg_waiting_time': float(np.mean(data['ddqn_eval']['avg_waiting_times'])),
                'avg_vehicles_passed': float(np.mean(data['ddqn_eval']['total_vehicles_passed']))
            },
            'baseline': {
                'avg_reward': float(np.mean(data['baseline_eval']['episode_rewards'])),
                'avg_waiting_time': float(np.mean(data['baseline_eval']['avg_waiting_times'])),
                'avg_vehicles_passed': float(np.mean(data['baseline_eval']['total_vehicles_passed']))
            }
        }
    
    results_path = './results/integrated_experiment_results.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(summary_results, f, indent=4)
    
    print(f"✅ 결과 저장: {results_path}")
    logger.log(f"JSON 결과 저장 완료: {results_path}")
    
    # 최종 요약 테이블
    print("\n" + "="*70)
    print("🎯 최종 실험 결과 요약")
    print("="*70)
    print(f"\n사용된 하이퍼파라미터: {params_source.upper()}")
    print(f"Learning Rate: {best_params['learning_rate']}")
    print(f"Gamma: {best_params['gamma']}")
    print(f"Batch Size: {best_params['batch_size']}")
    print(f"Buffer Size: {best_params['buffer_capacity']}")
    
    print("\n" + "-"*70)
    print(f"{'시나리오':<20} {'알고리즘':<20} {'평균 대기시간':>15} {'개선율':>12}")
    print("-"*70)
    
    for scenario in scenarios:
        data = summary_results['scenarios'][scenario]
        baseline_wait = data['baseline']['avg_waiting_time']
        
        for algo in ['dqn', 'ddqn', 'baseline']:
            wait_time = data[algo]['avg_waiting_time']
            improvement = ((baseline_wait - wait_time) / baseline_wait * 100) if algo != 'baseline' else 0.0
            
            algo_name = {
                'dqn': 'DQN',
                'ddqn': 'Double DQN',
                'baseline': 'Baseline'
            }[algo]
            
            print(f"{scenario:<20} {algo_name:<20} {wait_time:>15.2f}초 {improvement:>11.2f}%")
    
    print("-"*70)
    
    # 추가 시각화 생성
    logger.log_section("4단계: 추가 시각화 생성")
    print("\n📊 추가 시각화 그래프 생성 중...")
    
    # 1. 시나리오별 알고리즘 성능 비교
    plot_scenario_comparison(
        summary_results,
        './results/plots/scenario_comparison.png'
    )
    logger.log_graph_generation('scenario_comparison.png')
    
    # 2. 알고리즘별 개선율 비교
    plot_improvement_comparison(
        summary_results,
        './results/plots/improvement_comparison.png'
    )
    logger.log_graph_generation('improvement_comparison.png')
    
    # 3. 히트맵 비교
    plot_heatmap_comparison(
        summary_results,
        './results/plots/performance_heatmap.png'
    )
    logger.log_graph_generation('performance_heatmap.png')
    
    # 4. 종합 대시보드
    plot_comprehensive_dashboard(
        summary_results,
        './results/plots/comprehensive_dashboard.png'
    )
    logger.log_graph_generation('comprehensive_dashboard.png')
    
    print("\n📊 생성된 그래프 파일:")
    print("   학습 곡선:")
    print("   - ./results/plots/dqn_*_training.png")
    print("   - ./results/plots/ddqn_*_training.png")
    print("   성능 비교:")
    print("   - ./results/plots/scenario_comparison.png")
    print("   - ./results/plots/improvement_comparison.png")
    print("   - ./results/plots/performance_heatmap.png")
    print("   - ./results/plots/comprehensive_dashboard.png")
    
    # 최종 요약 로그 기록
    logger.log_final_summary(summary_results)
    
    # 실험 완료 로그
    logger.log_completion()
    
    print("\n✨ 모든 실험 완료!")
    print(f"📝 상세 로그: {logger.log_path}")
    print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='통합 실험 워크플로우')
    parser.add_argument(
        '--use-tuned',
        action='store_true',
        help='하이퍼파라미터 튜닝 결과 사용 (기본: 기본값 사용)'
    )
    parser.add_argument(
        '--tuning-results',
        type=str,
        default='./results/hyperparameter_tuning_results.json',
        help='하이퍼파라미터 튜닝 결과 파일 경로'
    )
    parser.add_argument(
        '--train-episodes',
        type=int,
        default=2000,
        help='학습 에피소드 수'
    )
    parser.add_argument(
        '--eval-episodes',
        type=int,
        default=100,
        help='평가 에피소드 수'
    )
    
    args = parser.parse_args()
    
    run_integrated_experiments(
        use_tuned_params=args.use_tuned,
        tuning_results_path=args.tuning_results,
        num_train_episodes=args.train_episodes,
        num_eval_episodes=args.eval_episodes
    )