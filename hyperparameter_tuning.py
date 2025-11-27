"""
하이퍼파라미터 튜닝 실험 스크립트
Learning Rate, Discount Factor, Batch Size, Replay Buffer Size 비교
"""

import numpy as np
import torch
from traffic_env import TrafficEnvironment
from dqn_agent import DQNAgent
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def train_with_params(
    env: TrafficEnvironment,
    params: dict,
    num_episodes: int = 1000,
    scenario: str = 'normal',
    seed: int = 42
) -> dict:
    """
    특정 하이퍼파라미터로 학습
    
    Args:
        env: 환경
        params: 하이퍼파라미터 딕셔너리
        num_episodes: 학습 에피소드 수
        scenario: 시나리오
        seed: 랜덤 시드
        
    Returns:
        학습 결과
    """
    # 시드 고정
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 환경 설정
    env.set_scenario(scenario)
    
    # 에이전트 생성
    agent = DQNAgent(
        state_dim=params['state_dim'],
        action_dim=params['action_dim'],
        learning_rate=params.get('learning_rate', 0.001),
        gamma=params.get('gamma', 0.95),
        epsilon_start=params.get('epsilon_start', 1.0),
        epsilon_end=params.get('epsilon_end', 0.01),
        epsilon_decay=params.get('epsilon_decay', 0.995),
        buffer_capacity=params.get('buffer_capacity', 10000),
        batch_size=params.get('batch_size', 64),
        target_update_freq=params.get('target_update_freq', 100)
    )
    
    # 학습 기록
    history = {
        'episode_rewards': [],
        'avg_waiting_times': [],
        'total_vehicles_passed': [],
        'losses': [],
        'max_queue_lengths': []
    }
    
    # 학습
    for episode in range(num_episodes):
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
        history['avg_waiting_times'].append(info['avg_waiting_time'])
        history['total_vehicles_passed'].append(info['total_vehicles_passed'])
        history['losses'].append(episode_loss / step_count if step_count > 0 else 0)
        history['max_queue_lengths'].append(info['max_queue_length'])
    
    # 최종 성능 (마지막 100 에피소드 평균)
    final_performance = {
        'avg_reward': np.mean(history['episode_rewards'][-100:]),
        'avg_waiting_time': np.mean(history['avg_waiting_times'][-100:]),
        'avg_vehicles_passed': np.mean(history['total_vehicles_passed'][-100:]),
        'avg_loss': np.mean(history['losses'][-100:])
    }
    
    return {
        'history': history,
        'final_performance': final_performance,
        'params': params
    }


def experiment_learning_rate(
    env: TrafficEnvironment,
    learning_rates: list = [0.0001, 0.001, 0.01],
    num_episodes: int = 1000,
    seeds: list = [42, 123, 456]
) -> dict:
    """
    실험 A: Learning Rate 비교
    
    Args:
        env: 환경
        learning_rates: 테스트할 learning rate 리스트
        num_episodes: 학습 에피소드 수
        seeds: 랜덤 시드 리스트 (3회 반복)
        
    Returns:
        실험 결과
    """
    print("\n" + "="*60)
    print("실험 A: Learning Rate 비교")
    print("="*60)
    
    results = {}
    
    # 기본 파라미터
    base_params = {
        'state_dim': 7,
        'action_dim': 2,
        'gamma': 0.95,
        'buffer_capacity': 10000,
        'batch_size': 64,
        'target_update_freq': 100
    }
    
    for lr in learning_rates:
        print(f"\n[Learning Rate = {lr}]")
        lr_results = []
        
        for seed_idx, seed in enumerate(seeds):
            print(f"  실행 {seed_idx + 1}/3 (seed={seed})...")
            
            params = base_params.copy()
            params['learning_rate'] = lr
            
            result = train_with_params(
                env, params, num_episodes, scenario='normal', seed=seed
            )
            lr_results.append(result)
        
        # 3회 평균 계산
        avg_result = {
            'avg_reward': np.mean([r['final_performance']['avg_reward'] for r in lr_results]),
            'avg_waiting_time': np.mean([r['final_performance']['avg_waiting_time'] for r in lr_results]),
            'avg_vehicles_passed': np.mean([r['final_performance']['avg_vehicles_passed'] for r in lr_results]),
            'std_reward': np.std([r['final_performance']['avg_reward'] for r in lr_results]),
            'all_runs': lr_results
        }
        
        results[f'lr_{lr}'] = avg_result
        
        print(f"  ✅ 평균 Reward: {avg_result['avg_reward']:.2f} (±{avg_result['std_reward']:.2f})")
        print(f"     평균 대기시간: {avg_result['avg_waiting_time']:.2f}초")
    
    return results


def experiment_discount_factor(
    env: TrafficEnvironment,
    gammas: list = [0.90, 0.95, 0.99],
    num_episodes: int = 1000,
    seeds: list = [42, 123, 456]
) -> dict:
    """
    실험 B: Discount Factor 비교
    """
    print("\n" + "="*60)
    print("실험 B: Discount Factor (γ) 비교")
    print("="*60)
    
    results = {}
    
    base_params = {
        'state_dim': 7,
        'action_dim': 2,
        'learning_rate': 0.001,
        'buffer_capacity': 10000,
        'batch_size': 64,
        'target_update_freq': 100
    }
    
    for gamma in gammas:
        print(f"\n[Gamma = {gamma}]")
        gamma_results = []
        
        for seed_idx, seed in enumerate(seeds):
            print(f"  실행 {seed_idx + 1}/3 (seed={seed})...")
            
            params = base_params.copy()
            params['gamma'] = gamma
            
            result = train_with_params(
                env, params, num_episodes, scenario='normal', seed=seed
            )
            gamma_results.append(result)
        
        avg_result = {
            'avg_reward': np.mean([r['final_performance']['avg_reward'] for r in gamma_results]),
            'avg_waiting_time': np.mean([r['final_performance']['avg_waiting_time'] for r in gamma_results]),
            'avg_vehicles_passed': np.mean([r['final_performance']['avg_vehicles_passed'] for r in gamma_results]),
            'std_reward': np.std([r['final_performance']['avg_reward'] for r in gamma_results]),
            'all_runs': gamma_results
        }
        
        results[f'gamma_{gamma}'] = avg_result
        
        print(f"  ✅ 평균 Reward: {avg_result['avg_reward']:.2f} (±{avg_result['std_reward']:.2f})")
        print(f"     평균 대기시간: {avg_result['avg_waiting_time']:.2f}초")
    
    return results


def experiment_batch_size(
    env: TrafficEnvironment,
    batch_sizes: list = [32, 64, 128],
    num_episodes: int = 1000,
    seeds: list = [42, 123, 456]
) -> dict:
    """
    실험 C: Batch Size 비교
    """
    print("\n" + "="*60)
    print("실험 C: Batch Size 비교")
    print("="*60)
    
    results = {}
    
    base_params = {
        'state_dim': 7,
        'action_dim': 2,
        'learning_rate': 0.001,
        'gamma': 0.95,
        'buffer_capacity': 10000,
        'target_update_freq': 100
    }
    
    for batch_size in batch_sizes:
        print(f"\n[Batch Size = {batch_size}]")
        bs_results = []
        
        for seed_idx, seed in enumerate(seeds):
            print(f"  실행 {seed_idx + 1}/3 (seed={seed})...")
            
            params = base_params.copy()
            params['batch_size'] = batch_size
            
            result = train_with_params(
                env, params, num_episodes, scenario='normal', seed=seed
            )
            bs_results.append(result)
        
        avg_result = {
            'avg_reward': np.mean([r['final_performance']['avg_reward'] for r in bs_results]),
            'avg_waiting_time': np.mean([r['final_performance']['avg_waiting_time'] for r in bs_results]),
            'avg_vehicles_passed': np.mean([r['final_performance']['avg_vehicles_passed'] for r in bs_results]),
            'std_reward': np.std([r['final_performance']['avg_reward'] for r in bs_results]),
            'all_runs': bs_results
        }
        
        results[f'batch_{batch_size}'] = avg_result
        
        print(f"  ✅ 평균 Reward: {avg_result['avg_reward']:.2f} (±{avg_result['std_reward']:.2f})")
        print(f"     평균 대기시간: {avg_result['avg_waiting_time']:.2f}초")
    
    return results


def experiment_buffer_size(
    env: TrafficEnvironment,
    buffer_sizes: list = [5000, 10000, 20000],
    num_episodes: int = 1000,
    seeds: list = [42, 123, 456]
) -> dict:
    """
    실험 D: Replay Buffer Size 비교
    """
    print("\n" + "="*60)
    print("실험 D: Replay Buffer Size 비교")
    print("="*60)
    
    results = {}
    
    base_params = {
        'state_dim': 7,
        'action_dim': 2,
        'learning_rate': 0.001,
        'gamma': 0.95,
        'batch_size': 64,
        'target_update_freq': 100
    }
    
    for buffer_size in buffer_sizes:
        print(f"\n[Buffer Size = {buffer_size}]")
        buf_results = []
        
        for seed_idx, seed in enumerate(seeds):
            print(f"  실행 {seed_idx + 1}/3 (seed={seed})...")
            
            params = base_params.copy()
            params['buffer_capacity'] = buffer_size
            
            result = train_with_params(
                env, params, num_episodes, scenario='normal', seed=seed
            )
            buf_results.append(result)
        
        avg_result = {
            'avg_reward': np.mean([r['final_performance']['avg_reward'] for r in buf_results]),
            'avg_waiting_time': np.mean([r['final_performance']['avg_waiting_time'] for r in buf_results]),
            'avg_vehicles_passed': np.mean([r['final_performance']['avg_vehicles_passed'] for r in buf_results]),
            'std_reward': np.std([r['final_performance']['avg_reward'] for r in buf_results]),
            'all_runs': buf_results
        }
        
        results[f'buffer_{buffer_size}'] = avg_result
        
        print(f"  ✅ 평균 Reward: {avg_result['avg_reward']:.2f} (±{avg_result['std_reward']:.2f})")
        print(f"     평균 대기시간: {avg_result['avg_waiting_time']:.2f}초")
    
    return results


def plot_hyperparameter_comparison(results: dict, param_name: str, save_path: str):
    """하이퍼파라미터 비교 그래프"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    param_values = list(results.keys())
    rewards = [results[k]['avg_reward'] for k in param_values]
    waiting_times = [results[k]['avg_waiting_time'] for k in param_values]
    std_rewards = [results[k]['std_reward'] for k in param_values]
    
    # 파라미터 이름 정리
    param_labels = [k.split('_', 1)[1] for k in param_values]
    
    # Reward 비교
    axes[0].bar(range(len(param_values)), rewards, yerr=std_rewards, capsize=5)
    axes[0].set_xticks(range(len(param_values)))
    axes[0].set_xticklabels(param_labels)
    axes[0].set_ylabel('Average Reward (last 100 episodes)')
    axes[0].set_title(f'Reward Comparison - {param_name}')
    axes[0].grid(True, alpha=0.3)
    
    # 대기시간 비교
    axes[1].bar(range(len(param_values)), waiting_times)
    axes[1].set_xticks(range(len(param_values)))
    axes[1].set_xticklabels(param_labels)
    axes[1].set_ylabel('Average Waiting Time (seconds)')
    axes[1].set_title(f'Waiting Time Comparison - {param_name}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 그래프 저장: {save_path}")
    plt.close()


def save_hyperparameter_results(all_results: dict, filepath: str):
    """결과 저장 (history 제외하고 요약만)"""
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            # 'all_runs'에서 'history' 제거 (너무 큼)
            filtered = {}
            for k, v in obj.items():
                if k == 'all_runs':
                    filtered[k] = [
                        {
                            'final_performance': run['final_performance'],
                            'params': run['params']
                        }
                        for run in v
                    ]
                else:
                    filtered[k] = convert_to_serializable(v)
            return filtered
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    print(f"\n💾 결과 저장: {filepath}")


def print_summary_table(all_results: dict):
    """결과 요약 테이블 출력"""
    print("\n" + "="*80)
    print("📊 하이퍼파라미터 튜닝 결과 요약")
    print("="*80)
    
    for experiment_name, results in all_results.items():
        print(f"\n[{experiment_name}]")
        print("-" * 80)
        print(f"{'파라미터':<20} {'평균 Reward':<20} {'평균 대기시간(초)':<20} {'표준편차':<15}")
        print("-" * 80)
        
        for param_key, data in results.items():
            param_label = param_key.split('_', 1)[1]
            print(f"{param_label:<20} {data['avg_reward']:<20.2f} "
                  f"{data['avg_waiting_time']:<20.2f} {data['std_reward']:<15.2f}")
        print("-" * 80)


if __name__ == "__main__":
    # 환경 생성
    env = TrafficEnvironment()
    
    print("="*60)
    print("🔬 하이퍼파라미터 튜닝 실험 시작")
    print("="*60)
    print("각 설정당 3회 실행 (seeds: 42, 123, 456)")
    print("학습 에피소드: 1000")
    
    # 모든 실험 실행
    all_results = {}
    
    # 실험 A: Learning Rate
    all_results['Learning_Rate'] = experiment_learning_rate(
        env,
        learning_rates=[0.0001, 0.001, 0.01],
        num_episodes=1000
    )
    
    # 실험 B: Discount Factor
    all_results['Discount_Factor'] = experiment_discount_factor(
        env,
        gammas=[0.90, 0.95, 0.99],
        num_episodes=1000
    )
    
    # 실험 C: Batch Size
    all_results['Batch_Size'] = experiment_batch_size(
        env,
        batch_sizes=[32, 64, 128],
        num_episodes=1000
    )
    
    # 실험 D: Buffer Size
    all_results['Buffer_Size'] = experiment_buffer_size(
        env,
        buffer_sizes=[5000, 10000, 20000],
        num_episodes=1000
    )
    
    # 결과 요약 출력
    print_summary_table(all_results)
    
    # 결과 저장
    save_hyperparameter_results(
        all_results,
        './results/hyperparameter_tuning_results.json'
    )
    
    # 그래프 생성
    plot_hyperparameter_comparison(
        all_results['Learning_Rate'],
        'Learning Rate',
        './results/plots/hyperparameter_lr.png'
    )
    
    plot_hyperparameter_comparison(
        all_results['Discount_Factor'],
        'Discount Factor',
        './results/plots/hyperparameter_gamma.png'
    )
    
    plot_hyperparameter_comparison(
        all_results['Batch_Size'],
        'Batch Size',
        './results/plots/hyperparameter_batch.png'
    )
    
    plot_hyperparameter_comparison(
        all_results['Buffer_Size'],
        'Buffer Size',
        './results/plots/hyperparameter_buffer.png'
    )
    
    print("\n" + "="*60)
    print("✨ 모든 하이퍼파라미터 실험 완료!")
    print("="*60)