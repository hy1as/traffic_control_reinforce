"""
빠른 테스트용 간단한 실험 스크립트
적은 에피소드로 빠르게 동작 확인
"""

import numpy as np
import torch
from traffic_env import TrafficEnvironment, FixedTimeController
from dqn_agent import DQNAgent, DoubleDQNAgent
import json
import os


def quick_test():
    """빠른 동작 테스트"""
    print("="*60)
    print("🚦 빠른 테스트 시작")
    print("="*60)
    
    # 환경 및 에이전트 생성
    env = TrafficEnvironment()
    env.set_scenario('normal')
    
    STATE_DIM = 7
    ACTION_DIM = 2
    
    agent = DQNAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        learning_rate=0.001,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_capacity=1000,  # 테스트용으로 작게
        batch_size=32,
        target_update_freq=10
    )
    
    print(f"디바이스: {agent.device}")
    print(f"상태 차원: {STATE_DIM}")
    print(f"행동 차원: {ACTION_DIM}")
    
    # 짧은 학습 (10 에피소드)
    print("\n📚 학습 시작 (10 에피소드)")
    
    history = {
        'episode_rewards': [],
        'avg_waiting_times': [],
        'losses': []
    }
    
    for episode in range(10):
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
        history['losses'].append(episode_loss / step_count if step_count > 0 else 0)
        
        print(f"Episode {episode+1}/10 - Reward: {episode_reward:.2f}, "
              f"대기시간: {info['avg_waiting_time']:.2f}초, "
              f"ε: {agent.epsilon:.4f}")
    
    # 평가 (5 에피소드)
    print("\n📊 평가 시작 (5 에피소드)")
    
    eval_results = {
        'episode_rewards': [],
        'avg_waiting_times': []
    }
    
    for episode in range(5):
        state = env.reset()
        episode_reward = 0
        
        done = False
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
        
        eval_results['episode_rewards'].append(episode_reward)
        eval_results['avg_waiting_times'].append(info['avg_waiting_time'])
        
        print(f"Eval {episode+1}/5 - Reward: {episode_reward:.2f}, "
              f"대기시간: {info['avg_waiting_time']:.2f}초")
    
    print(f"\n평균 평가 Reward: {np.mean(eval_results['episode_rewards']):.2f}")
    print(f"평균 대기시간: {np.mean(eval_results['avg_waiting_times']):.2f}초")
    
    # Baseline 비교
    print("\n🔧 Baseline (고정 주기 30초) 평가 (5 에피소드)")
    
    controller = FixedTimeController(cycle_time=30)
    baseline_results = {
        'episode_rewards': [],
        'avg_waiting_times': []
    }
    
    for episode in range(5):
        state = env.reset()
        controller.reset()
        episode_reward = 0
        
        done = False
        while not done:
            action = controller.get_action(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
        
        baseline_results['episode_rewards'].append(episode_reward)
        baseline_results['avg_waiting_times'].append(info['avg_waiting_time'])
    
    print(f"Baseline 평균 Reward: {np.mean(baseline_results['episode_rewards']):.2f}")
    print(f"Baseline 평균 대기시간: {np.mean(baseline_results['avg_waiting_times']):.2f}초")
    
    # 결과 저장
    os.makedirs('./results', exist_ok=True)
    
    test_results = {
        'training': history,
        'evaluation': eval_results,
        'baseline': baseline_results
    }
    
    with open('./results/quick_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=4)
    
    print("\n✅ 테스트 완료!")
    print("결과 저장: ./results/quick_test_results.json")
    
    print("\n" + "="*60)
    print("🎉 모든 기능이 정상 동작합니다!")
    print("="*60)


if __name__ == "__main__":
    quick_test()