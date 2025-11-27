"""
빠른 하이퍼파라미터 튜닝 테스트
각 설정당 50 에피소드만 학습하여 빠르게 동작 확인
"""

import numpy as np
import torch
from traffic_env import TrafficEnvironment
from dqn_agent import DQNAgent
import json
import os


def quick_hyperparameter_test():
    """빠른 하이퍼파라미터 테스트"""
    print("="*60)
    print("🔬 빠른 하이퍼파라미터 테스트")
    print("="*60)
    print("각 설정당 50 에피소드만 학습")
    
    env = TrafficEnvironment()
    env.set_scenario('normal')
    
    STATE_DIM = 7
    ACTION_DIM = 2
    NUM_EPISODES = 50
    
    results = {}
    
    # 실험 A: Learning Rate
    print("\n[실험 A] Learning Rate 비교")
    lr_results = {}
    
    for lr in [0.0001, 0.001, 0.01]:
        print(f"\n  Learning Rate = {lr}")
        
        agent = DQNAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=lr,
            gamma=0.95,
            buffer_capacity=1000,
            batch_size=32
        )
        
        rewards = []
        waiting_times = []
        
        for episode in range(NUM_EPISODES):
            state = env.reset()
            episode_reward = 0
            
            done = False
            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.update()
                episode_reward += reward
                state = next_state
            
            agent.decay_epsilon()
            rewards.append(episode_reward)
            waiting_times.append(info['avg_waiting_time'])
        
        # 마지막 20 에피소드 평균
        avg_reward = np.mean(rewards[-20:])
        avg_waiting = np.mean(waiting_times[-20:])
        
        lr_results[f'lr_{lr}'] = {
            'avg_reward': avg_reward,
            'avg_waiting_time': avg_waiting
        }
        
        print(f"    평균 Reward: {avg_reward:.2f}")
        print(f"    평균 대기시간: {avg_waiting:.2f}초")
    
    results['Learning_Rate'] = lr_results
    
    # 실험 B: Discount Factor
    print("\n[실험 B] Discount Factor 비교")
    gamma_results = {}
    
    for gamma in [0.90, 0.95, 0.99]:
        print(f"\n  Gamma = {gamma}")
        
        agent = DQNAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=0.001,
            gamma=gamma,
            buffer_capacity=1000,
            batch_size=32
        )
        
        rewards = []
        waiting_times = []
        
        for episode in range(NUM_EPISODES):
            state = env.reset()
            episode_reward = 0
            
            done = False
            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.update()
                episode_reward += reward
                state = next_state
            
            agent.decay_epsilon()
            rewards.append(episode_reward)
            waiting_times.append(info['avg_waiting_time'])
        
        avg_reward = np.mean(rewards[-20:])
        avg_waiting = np.mean(waiting_times[-20:])
        
        gamma_results[f'gamma_{gamma}'] = {
            'avg_reward': avg_reward,
            'avg_waiting_time': avg_waiting
        }
        
        print(f"    평균 Reward: {avg_reward:.2f}")
        print(f"    평균 대기시간: {avg_waiting:.2f}초")
    
    results['Discount_Factor'] = gamma_results
    
    # 실험 C: Batch Size
    print("\n[실험 C] Batch Size 비교")
    batch_results = {}
    
    for batch_size in [16, 32, 64]:
        print(f"\n  Batch Size = {batch_size}")
        
        agent = DQNAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            learning_rate=0.001,
            gamma=0.95,
            buffer_capacity=1000,
            batch_size=batch_size
        )
        
        rewards = []
        waiting_times = []
        
        for episode in range(NUM_EPISODES):
            state = env.reset()
            episode_reward = 0
            
            done = False
            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.update()
                episode_reward += reward
                state = next_state
            
            agent.decay_epsilon()
            rewards.append(episode_reward)
            waiting_times.append(info['avg_waiting_time'])
        
        avg_reward = np.mean(rewards[-20:])
        avg_waiting = np.mean(waiting_times[-20:])
        
        batch_results[f'batch_{batch_size}'] = {
            'avg_reward': avg_reward,
            'avg_waiting_time': avg_waiting
        }
        
        print(f"    평균 Reward: {avg_reward:.2f}")
        print(f"    평균 대기시간: {avg_waiting:.2f}초")
    
    results['Batch_Size'] = batch_results
    
    # 결과 요약
    print("\n" + "="*60)
    print("📊 결과 요약")
    print("="*60)
    
    for exp_name, exp_results in results.items():
        print(f"\n[{exp_name}]")
        print("-" * 60)
        for param_key, metrics in exp_results.items():
            param_label = param_key.split('_', 1)[1]
            print(f"{param_label:<15} Reward: {metrics['avg_reward']:>8.2f}  "
                  f"대기시간: {metrics['avg_waiting_time']:>6.2f}초")
        print("-" * 60)
    
    # 결과 저장
    os.makedirs('./results', exist_ok=True)
    with open('./results/quick_hyperparameter_test.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n✅ 테스트 완료!")
    print("결과 저장: ./results/quick_hyperparameter_test.json")
    
    # 최적 파라미터 추천
    print("\n" + "="*60)
    print("🎯 추천 파라미터 (테스트 기반)")
    print("="*60)
    
    # Learning Rate에서 최고 성능
    best_lr = max(results['Learning_Rate'].items(), 
                  key=lambda x: x[1]['avg_reward'])
    print(f"Learning Rate: {best_lr[0].split('_')[1]} "
          f"(Reward: {best_lr[1]['avg_reward']:.2f})")
    
    # Gamma에서 최고 성능
    best_gamma = max(results['Discount_Factor'].items(), 
                     key=lambda x: x[1]['avg_reward'])
    print(f"Discount Factor: {best_gamma[0].split('_')[1]} "
          f"(Reward: {best_gamma[1]['avg_reward']:.2f})")
    
    # Batch Size에서 최고 성능
    best_batch = max(results['Batch_Size'].items(), 
                     key=lambda x: x[1]['avg_reward'])
    print(f"Batch Size: {best_batch[0].split('_')[1]} "
          f"(Reward: {best_batch[1]['avg_reward']:.2f})")
    
    print("\n⚠️  주의: 50 에피소드 테스트 결과이므로 참고용입니다.")
    print("   정확한 결과는 hyperparameter_tuning.py 실행 필요")
    print("="*60)


if __name__ == "__main__":
    quick_hyperparameter_test()