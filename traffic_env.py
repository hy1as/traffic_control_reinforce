"""
교차로 환경 (Traffic Intersection Environment)
4방향 교차로를 시뮬레이션하는 환경
"""

import numpy as np
from collections import deque
from typing import Tuple, Dict, List


class TrafficEnvironment:
    """4거리 교차로 시뮬레이션 환경"""
    
    def __init__(
        self,
        max_queue_length: int = 30,
        yellow_phase_duration: int = 2,
        min_green_duration: int = 5,
        arrival_rates: Dict[str, float] = None
    ):
        """
        Args:
            max_queue_length: 각 차선의 최대 대기 차량 수
            yellow_phase_duration: 노란불 지속 시간 (초)
            min_green_duration: 최소 신호 유지 시간 (초)
            arrival_rates: 각 방향별 차량 도착률 (대/초)
        """
        self.max_queue_length = max_queue_length
        self.yellow_phase_duration = yellow_phase_duration
        self.min_green_duration = min_green_duration
        
        # 기본 도착률 설정 (평시)
        if arrival_rates is None:
            arrival_rates = {
                'north': 0.2,
                'south': 0.2,
                'east': 0.2,
                'west': 0.2
            }
        self.arrival_rates = arrival_rates
        
        # 차선 큐 초기화
        self.queues = {
            'north': deque(),
            'south': deque(),
            'east': deque(),
            'west': deque()
        }
        
        # 상태 변수
        self.current_phase = 0  # 0: 북/남 초록, 1: 동/서 초록
        self.phase_duration = 0  # 현재 신호 지속 시간
        self.yellow_phase_active = False
        self.yellow_phase_counter = 0
        
        # 통계
        self.total_waiting_time = 0
        self.total_vehicles_passed = 0
        self.current_step = 0
        self.time_of_day = 1  # 0: 야간, 1: 평시, 2: 출퇴근
        
    def reset(self) -> np.ndarray:
        """환경 초기화"""
        # 모든 큐 비우기
        for direction in self.queues:
            self.queues[direction].clear()
        
        # 상태 변수 초기화
        self.current_phase = 0
        self.phase_duration = 0
        self.yellow_phase_active = False
        self.yellow_phase_counter = 0
        
        # 통계 초기화
        self.total_waiting_time = 0
        self.total_vehicles_passed = 0
        self.current_step = 0
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        환경 한 스텝 진행
        
        Args:
            action: 0 (신호 유지) 또는 1 (신호 변경)
            
        Returns:
            next_state: 다음 상태
            reward: 보상
            done: 에피소드 종료 여부
            info: 추가 정보
        """
        self.current_step += 1
        
        # 1. 신호 제어 처리
        if action == 1 and not self.yellow_phase_active:
            # 신호 변경 요청
            if self.phase_duration >= self.min_green_duration:
                # 노란불 시작
                self.yellow_phase_active = True
                self.yellow_phase_counter = 0
        
        # 2. 노란불 처리
        if self.yellow_phase_active:
            self.yellow_phase_counter += 1
            if self.yellow_phase_counter >= self.yellow_phase_duration:
                # 노란불 종료, 신호 전환
                self.current_phase = 1 - self.current_phase
                self.phase_duration = 0
                self.yellow_phase_active = False
        
        # 3. 신규 차량 생성 (Poisson 분포)
        self._generate_vehicles()
        
        # 4. 차량 통과 처리 (초록불 방향만)
        if not self.yellow_phase_active:
            self._process_green_lanes()
        
        # 5. 대기 시간 누적
        self._accumulate_waiting_time()
        
        # 6. 신호 지속 시간 증가
        if not self.yellow_phase_active:
            self.phase_duration += 1
        
        # 7. 보상 계산
        reward = self._calculate_reward(action)
        
        # 8. 다음 상태 및 종료 조건
        next_state = self._get_state()
        done = self.current_step >= 1000  # 에피소드 길이
        
        # 9. 추가 정보
        info = {
            'total_waiting_time': self.total_waiting_time,
            'total_vehicles_passed': self.total_vehicles_passed,
            'avg_waiting_time': (
                self.total_waiting_time / max(1, self.total_vehicles_passed)
            ),
            'queue_lengths': self._get_queue_lengths(),
            'max_queue_length': max(self._get_queue_lengths().values())
        }
        
        return next_state, reward, done, info
    
    def _generate_vehicles(self):
        """각 차선에 확률적으로 차량 생성"""
        for direction in self.queues:
            # Poisson 분포로 차량 수 결정
            num_vehicles = np.random.poisson(self.arrival_rates[direction])
            
            for _ in range(num_vehicles):
                if len(self.queues[direction]) < self.max_queue_length:
                    # 차량 추가 (대기 시간 0으로 시작)
                    self.queues[direction].append(0)
    
    def _process_green_lanes(self):
        """초록불 차선의 차량 통과 처리"""
        if self.current_phase == 0:
            # 북/남 초록
            active_directions = ['north', 'south']
        else:
            # 동/서 초록
            active_directions = ['east', 'west']
        
        for direction in active_directions:
            if len(self.queues[direction]) > 0:
                # 맨 앞 차량 통과 (1초당 1대)
                self.queues[direction].popleft()
                self.total_vehicles_passed += 1
    
    def _accumulate_waiting_time(self):
        """모든 대기 차량의 대기 시간 누적"""
        for direction in self.queues:
            for i in range(len(self.queues[direction])):
                self.queues[direction][i] += 1
                self.total_waiting_time += 1
    
    def _calculate_reward(self, action: int) -> float:
        """보상 계산"""
        # 기본 페널티: 대기 차량 수
        total_waiting = sum(len(q) for q in self.queues.values())
        reward = -1.0 * total_waiting
        
        # 신호 변경 비용
        if action == 1:
            reward -= 5.0
        
        # 혼잡 가중 페널티 (특정 방향 15대 이상)
        for direction in self.queues:
            queue_length = len(self.queues[direction])
            if queue_length > 15:
                reward -= (queue_length - 15) * 2.0
        
        # 처리 보너스 (이번 스텝에 차량 통과한 경우)
        # 이전 스텝 대기 차량 수와 비교 필요하지만 단순화
        
        return reward
    
    def _get_state(self) -> np.ndarray:
        """현재 상태 반환 (정규화된 7차원 벡터)"""
        # 1-4: 각 차선 대기 차량 수 (0~1 정규화)
        queue_lengths = [
            min(len(self.queues['north']) / self.max_queue_length, 1.0),
            min(len(self.queues['south']) / self.max_queue_length, 1.0),
            min(len(self.queues['east']) / self.max_queue_length, 1.0),
            min(len(self.queues['west']) / self.max_queue_length, 1.0)
        ]
        
        # 5: 현재 신호 상태 (0 또는 1)
        phase = float(self.current_phase)
        
        # 6: 신호 지속 시간 (0~1 정규화, 최대 60초 가정)
        duration = min(self.phase_duration / 60.0, 1.0)
        
        # 7: 시간대 정보 (0~1 정규화, 0~2 범위를 0~1로)
        time_of_day = self.time_of_day / 2.0
        
        state = np.array(
            queue_lengths + [phase, duration, time_of_day],
            dtype=np.float32
        )
        
        return state
    
    def _get_queue_lengths(self) -> Dict[str, int]:
        """현재 각 차선의 대기 차량 수 반환"""
        return {
            direction: len(self.queues[direction])
            for direction in self.queues
        }
    
    def set_scenario(self, scenario: str):
        """시나리오별 차량 생성률 설정"""
        scenarios = {
            'normal': {
                'north': 0.2, 'south': 0.2,
                'east': 0.2, 'west': 0.2
            },
            'morning_rush': {
                'north': 0.6, 'south': 0.6,
                'east': 0.15, 'west': 0.15
            },
            'evening_rush': {
                'north': 0.15, 'south': 0.15,
                'east': 0.6, 'west': 0.6
            },
            'congestion': {
                'north': 0.8, 'south': 0.8,
                'east': 0.7, 'west': 0.7
            },
            'night': {
                'north': 0.05, 'south': 0.05,
                'east': 0.05, 'west': 0.05
            }
        }
        
        if scenario in scenarios:
            self.arrival_rates = scenarios[scenario]
            
            # 시간대 정보 업데이트
            if scenario == 'night':
                self.time_of_day = 0
            elif scenario in ['morning_rush', 'evening_rush']:
                self.time_of_day = 2
            else:
                self.time_of_day = 1


class FixedTimeController:
    """비교 실험용 고정 주기 신호등"""
    
    def __init__(self, cycle_time: int = 30):
        """
        Args:
            cycle_time: 각 방향 신호 유지 시간 (초)
        """
        self.cycle_time = cycle_time
        self.current_time = 0
        self.current_phase = 0
    
    def get_action(self, state: np.ndarray) -> int:
        """고정 주기로 신호 변경"""
        self.current_time += 1
        
        if self.current_time >= self.cycle_time:
            self.current_time = 0
            self.current_phase = 1 - self.current_phase
            return 1  # 신호 변경
        
        return 0  # 신호 유지
    
    def reset(self):
        """컨트롤러 초기화"""
        self.current_time = 0
        self.current_phase = 0