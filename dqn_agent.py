"""
DQN Agent 구현
Deep Q-Network와 Double DQN 알고리즘
PyTorch 기반
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import Tuple, List


class QNetwork(nn.Module):
    """Q-Network (신경망 구조)"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Args:
            state_dim: 상태 공간 차원 (7)
            action_dim: 행동 공간 차원 (2)
            hidden_dim: 은닉층 차원
        """
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """순전파"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class ReplayBuffer:
    """Experience Replay Buffer"""
    
    def __init__(self, capacity: int = 10000):
        """
        Args:
            capacity: 버퍼 최대 크기
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """경험 저장"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """배치 샘플링"""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)
        )
    
    def __len__(self) -> int:
        """현재 버퍼 크기"""
        return len(self.buffer)


class DQNAgent:
    """DQN Agent"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: str = None
    ):
        """
        Args:
            state_dim: 상태 공간 차원
            action_dim: 행동 공간 차원
            learning_rate: 학습률
            gamma: 할인율
            epsilon_start: 초기 탐색률
            epsilon_end: 최종 탐색률
            epsilon_decay: 탐색률 감소율
            buffer_capacity: Replay Buffer 크기
            batch_size: 배치 크기
            target_update_freq: Target Network 업데이트 주기
            device: 디바이스 (cpu/cuda)
        """
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # 네트워크 초기화
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate
        )
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # ε-greedy 파라미터
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 학습 카운터
        self.update_counter = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        행동 선택 (ε-greedy)
        
        Args:
            state: 현재 상태
            training: 학습 모드 여부
            
        Returns:
            선택된 행동
        """
        if training and random.random() < self.epsilon:
            # 탐색: 랜덤 행동
            return random.randint(0, self.action_dim - 1)
        else:
            # 활용: Q-value 최대화 행동
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(
                    self.device
                )
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=1).item()
            return action
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """경험 저장"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> float:
        """
        Q-Network 업데이트
        
        Returns:
            loss 값
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # 배치 샘플링
        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(self.batch_size)
        )
        
        # Tensor 변환
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 현재 Q-value
        current_q_values = self.q_network(states).gather(1, actions)
        
        # 목표 Q-value (DQN)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Loss 계산
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Target Network 업데이트
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        """ε 감소"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """모델 저장"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath: str):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


class DoubleDQNAgent(DQNAgent):
    """Double DQN Agent"""
    
    def update(self) -> float:
        """
        Q-Network 업데이트 (Double DQN 방식)
        
        Returns:
            loss 값
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # 배치 샘플링
        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample(self.batch_size)
        )
        
        # Tensor 변환
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 현재 Q-value
        current_q_values = self.q_network(states).gather(1, actions)
        
        # 목표 Q-value (Double DQN)
        with torch.no_grad():
            # Main Network로 최적 행동 선택
            next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            
            # Target Network로 Q-value 평가
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        
        # Loss 계산
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Target Network 업데이트
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()