"""
교통 신호등 제어 시연 스크립트
ASCII 아트로 교차로 상황을 시각화하고 학습된 모델이 신호를 조정하는 것을 보여줌
"""

import os
import sys
import time
import json
import numpy as np
import torch
from collections import deque
from traffic_env import TrafficEnvironment, FixedTimeController
from dqn_agent import DQNAgent, DoubleDQNAgent


class DemoEnvironment:
    """저장된 가상 데이터를 사용하는 시연용 환경"""
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: 저장된 교통 데이터 JSON 파일 경로
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.scenario = self.data['scenario']
        self.arrival_rates = self.data['arrival_rates']
        self.sequence = self.data['sequence']
        self.num_steps = len(self.sequence)
        
        # 환경 초기화 (도착률 0으로 설정하여 자동 생성 방지)
        self.env = TrafficEnvironment(
            arrival_rates={'north': 0, 'south': 0, 'east': 0, 'west': 0}
        )
        self.env.set_scenario(self.scenario)
        
        # 시뮬레이션 상태
        self.current_step = 0
        self.step_index = 0  # 시퀀스 인덱스
        
    def reset(self):
        """환경 초기화"""
        self.current_step = 0
        self.step_index = 0
        return self.env.reset()
    
    def step(self, action: int):
        """
        한 스텝 진행 (저장된 데이터 사용)
        
        Args:
            action: 0 (신호 유지) 또는 1 (신호 변경)
            
        Returns:
            next_state, reward, done, info
        """
        if self.step_index >= len(self.sequence):
            # 시퀀스 끝
            next_state = self.env._get_state()
            done = True
            info = self._get_info()
            return next_state, 0.0, done, info
        
        # 저장된 도착 데이터 사용
        arrivals = self.sequence[self.step_index]['arrivals']
        
        # 차량 도착 처리 (저장된 데이터 사용)
        # 환경의 _generate_vehicles를 대체
        for direction in ['north', 'south', 'east', 'west']:
            num_vehicles = arrivals[direction]
            for _ in range(num_vehicles):
                if len(self.env.queues[direction]) < self.env.max_queue_length:
                    self.env.queues[direction].append(0)
        
        # 신호 제어 처리 (환경의 step 로직을 직접 구현)
        self.env.current_step += 1
        
        # 1. 신호 제어 처리
        if action == 1 and not self.env.yellow_phase_active:
            if self.env.phase_duration >= self.env.min_green_duration:
                self.env.yellow_phase_active = True
                self.env.yellow_phase_counter = 0
        
        # 2. 노란불 처리
        if self.env.yellow_phase_active:
            self.env.yellow_phase_counter += 1
            if self.env.yellow_phase_counter >= self.env.yellow_phase_duration:
                self.env.current_phase = 1 - self.env.current_phase
                self.env.phase_duration = 0
                self.env.yellow_phase_active = False
        
        # 3. 차량 통과 처리 (초록불 방향만)
        if not self.env.yellow_phase_active:
            if self.env.current_phase == 0:
                active_directions = ['north', 'south']
            else:
                active_directions = ['east', 'west']
            
            for direction in active_directions:
                if len(self.env.queues[direction]) > 0:
                    self.env.queues[direction].popleft()
                    self.env.total_vehicles_passed += 1
        
        # 4. 대기 시간 누적
        for direction in self.env.queues:
            for i in range(len(self.env.queues[direction])):
                self.env.queues[direction][i] += 1
                self.env.total_waiting_time += 1
        
        # 5. 신호 지속 시간 증가
        if not self.env.yellow_phase_active:
            self.env.phase_duration += 1
        
        # 6. 보상 계산
        reward = self._calculate_reward(action)
        
        # 7. 다음 상태 및 종료 조건
        next_state = self.env._get_state()
        done = self.env.current_step >= 1000 or self.step_index >= len(self.sequence) - 1
        
        # 8. 추가 정보
        info = self._get_info()
        
        self.step_index += 1
        
        return next_state, reward, done, info
    
    def _calculate_reward(self, action: int) -> float:
        """보상 계산 (환경과 동일한 로직)"""
        total_waiting = sum(len(q) for q in self.env.queues.values())
        reward = -1.0 * total_waiting
        
        if action == 1:
            reward -= 5.0
        
        for direction in self.env.queues:
            queue_length = len(self.env.queues[direction])
            if queue_length > 15:
                reward -= (queue_length - 15) * 2.0
        
        return reward
    
    def _get_state(self):
        """현재 상태 반환"""
        return self.env._get_state()
    
    def _get_info(self):
        """추가 정보 반환"""
        return {
            'total_waiting_time': self.env.total_waiting_time,
            'total_vehicles_passed': self.env.total_vehicles_passed,
            'avg_waiting_time': (
                self.env.total_waiting_time / max(1, self.env.total_vehicles_passed)
            ),
            'queue_lengths': self.env._get_queue_lengths(),
            'max_queue_length': max(self.env._get_queue_lengths().values())
        }
    
    def get_current_state_info(self):
        """현재 상태 정보 반환"""
        return {
            'step': self.current_step,
            'phase': self.env.current_phase,
            'phase_duration': self.env.phase_duration,
            'yellow_active': self.env.yellow_phase_active,
            'queue_lengths': self.env._get_queue_lengths(),
            'arrival_rates': self.arrival_rates
        }


def clear_screen():
    """화면 지우기"""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_display_width(text: str) -> int:
    """
    문자열의 실제 표시 폭 계산 (한글=2칸, 영문/숫자=1칸)
    
    Args:
        text: 계산할 문자열
        
    Returns:
        실제 표시 폭
    """
    width = 0
    for char in text:
        # 한글 범위 (AC00-D7A3)
        if 0xAC00 <= ord(char) <= 0xD7A3:
            width += 2
        # 한자 범위 (4E00-9FFF)
        elif 0x4E00 <= ord(char) <= 0x9FFF:
            width += 2
        # 일본어 히라가나/가타카나
        elif 0x3040 <= ord(char) <= 0x309F or 0x30A0 <= ord(char) <= 0x30FF:
            width += 2
        else:
            width += 1
    return width


def pad_to_width(text: str, target_width: int, align: str = 'left') -> str:
    """
    문자열을 지정된 폭으로 맞춤 (한글 폭 고려)
    
    Args:
        text: 맞출 문자열
        target_width: 목표 폭 (바이트 수)
        align: 정렬 방식 ('left', 'right', 'center')
        
    Returns:
        맞춘 문자열
    """
    current_width = get_display_width(text)
    
    if current_width >= target_width:
        return text
    
    padding = target_width - current_width
    
    if align == 'left':
        return text + ' ' * padding
    elif align == 'right':
        return ' ' * padding + text
    else:  # center
        left_pad = padding // 2
        right_pad = padding - left_pad
        return ' ' * left_pad + text + ' ' * right_pad


def get_display_width(text: str) -> int:
    """
    문자열의 실제 표시 폭 계산 (한글=2칸, 영문/숫자=1칸)
    
    Args:
        text: 계산할 문자열
        
    Returns:
        실제 표시 폭
    """
    width = 0
    for char in text:
        # 한글, 한자, 일본어 등은 2칸
        if ord(char) >= 0xAC00 and ord(char) <= 0xD7A3:  # 한글 범위
            width += 2
        elif ord(char) >= 0x4E00 and ord(char) <= 0x9FFF:  # 한자 범위
            width += 2
        else:
            width += 1
    return width


def pad_to_width(text: str, target_width: int, align: str = 'left') -> str:
    """
    문자열을 지정된 폭으로 맞춤 (한글 폭 고려)
    
    Args:
        text: 맞출 문자열
        target_width: 목표 폭
        align: 정렬 방식 ('left', 'right', 'center')
        
    Returns:
        맞춘 문자열
    """
    current_width = get_display_width(text)
    
    if current_width >= target_width:
        return text
    
    padding = target_width - current_width
    
    if align == 'left':
        return text + ' ' * padding
    elif align == 'right':
        return ' ' * padding + text
    else:  # center
        left_pad = padding // 2
        right_pad = padding - left_pad
        return ' ' * left_pad + text + ' ' * right_pad


def draw_intersection(env_info: dict, action: int = None, q_values: list = None):
    """
    ASCII 아트로 교차로 시각화
    
    Args:
        env_info: 환경 정보 딕셔너리
        action: 선택된 행동 (None이면 표시 안 함)
        q_values: Q-value 리스트 (None이면 표시 안 함)
    """
    queues = env_info['queue_lengths']
    phase = env_info['phase']
    phase_duration = env_info['phase_duration']
    yellow = env_info['yellow_active']
    step = env_info['step']
    
    # 신호 상태 결정 (ASCII 문자 사용)
    if yellow:
        north_south_signal = "[Y]"
        east_west_signal = "[R]"
    elif phase == 0:
        north_south_signal = "[G]"
        east_west_signal = "[R]"
    else:
        north_south_signal = "[R]"
        east_west_signal = "[G]"
    
    # 차량 표시 (최대 15대까지만 표시)
    def format_queue(direction: str, max_display: int = 15):
        count = queues[direction]
        if count == 0:
            return "." * max_display
        elif count <= max_display:
            return "O" * count + "." * (max_display - count)
        else:
            return "O" * max_display + f" +{count-max_display}"
    
    # 교차로 그리기
    output = []
    output.append("=" * 70)
    output.append(f"  교통 신호등 제어 시뮬레이션 - Step {step:4d} | 신호 지속: {phase_duration:3d}초")
    output.append("=" * 70)
    output.append("")
    
    # 북쪽 차선
    north_queue = format_queue('north')
    output.append(f"        {north_south_signal} 북쪽: {north_queue} ({queues['north']:2d}대)")
    output.append("")
    output.append("        │")
    output.append("        │")
    
    # 중앙 교차로
    west_queue = format_queue('west')
    east_queue = format_queue('east')
    
    output.append(f"서쪽: {west_queue} ({queues['west']:2d}대) ────┼──── 동쪽: {east_queue} ({queues['east']:2d}대)")
    output.append(f"        {east_west_signal}")
    output.append("        │")
    output.append("        │")
    
    # 남쪽 차선
    south_queue = format_queue('south')
    output.append(f"        {north_south_signal} 남쪽: {south_queue} ({queues['south']:2d}대)")
    output.append("")
    
    # 통계 정보 (고정 폭)
    total_waiting = sum(queues.values())
    output.append("-" * 70)
    stats_line = f"총 대기 차량: {total_waiting:3d}대  |  북/남: {queues['north']+queues['south']:2d}대  |  동/서: {queues['east']+queues['west']:2d}대"
    stats_padded = pad_to_width(stats_line, 68)
    output.append(f"  {stats_padded}")
    
    # 행동 정보 (고정 폭)
    if action is not None:
        action_text = "신호 변경" if action == 1 else "신호 유지"
        action_line = f"선택된 행동: {action_text}"
        action_padded = pad_to_width(action_line, 68)
        output.append(f"  {action_padded}")
    
    # Q-value 정보 (고정 폭)
    if q_values is not None:
        q_value_line = f"Q-value: 유지={q_values[0]:7.2f}  변경={q_values[1]:7.2f}"
        q_value_padded = pad_to_width(q_value_line, 68)
        output.append(f"  {q_value_padded}")
    
    output.append("=" * 70)
    
    return "\n".join(output)


def draw_comparison_view(
    rl_info: dict,
    baseline_info: dict,
    step: int,
    rl_action: int = None,
    rl_q_values: list = None
):
    """
    고정 신호와 학습된 모델을 나란히 비교하는 시각화
    
    Args:
        rl_info: 강화학습 모델 환경 정보
        baseline_info: 고정 신호 환경 정보
        step: 현재 스텝
        rl_action: 강화학습 모델의 행동
        rl_q_values: 강화학습 모델의 Q-value
    """
    def format_queue_small(queues, direction, max_display=8):
        count = queues[direction]
        if count == 0:
            return "." * max_display
        elif count <= max_display:
            return "O" * count + "." * (max_display - count)
        else:
            return "O" * max_display + f"+{count-max_display}"
    
    def get_signal(phase, yellow, is_ns):
        if yellow:
            return "[Y]"
        elif (phase == 0 and is_ns) or (phase == 1 and not is_ns):
            return "[G]"
        else:
            return "[R]"
    
    rl_queues = rl_info['queue_lengths']
    baseline_queues = baseline_info['queue_lengths']
    
    rl_ns_signal = get_signal(rl_info['phase'], rl_info['yellow_active'], True)
    rl_ew_signal = get_signal(rl_info['phase'], rl_info['yellow_active'], False)
    baseline_ns_signal = get_signal(baseline_info['phase'], baseline_info['yellow_active'], True)
    baseline_ew_signal = get_signal(baseline_info['phase'], baseline_info['yellow_active'], False)
    
    output = []
    output.append("=" * 140)
    output.append(f"  교통 신호등 제어 성능 비교 - Step {step:4d}")
    output.append("=" * 140)
    output.append("")
    
    # 헤더 (고정 폭)
    header_left = pad_to_width('강화학습 모델 (RL)', 69, 'center')
    header_right = pad_to_width('고정 신호 (Baseline)', 69, 'center')
    output.append(f"{header_left} │ {header_right}")
    output.append("-" * 140)
    
    # 북쪽 차선 (고정 폭)
    rl_north = format_queue_small(rl_queues, 'north')
    baseline_north = format_queue_small(baseline_queues, 'north')
    rl_north_line = f"{rl_ns_signal} 북: {rl_north} ({rl_queues['north']:2d}대)"
    baseline_north_line = f"{baseline_ns_signal} 북: {baseline_north} ({baseline_queues['north']:2d}대)"
    rl_north_padded = pad_to_width(rl_north_line, 69)
    baseline_north_padded = pad_to_width(baseline_north_line, 69)
    output.append(f"{rl_north_padded} │ {baseline_north_padded}")
    output.append("")
    
    # 중앙 교차로
    rl_west = format_queue_small(rl_queues, 'west')
    rl_east = format_queue_small(rl_queues, 'east')
    baseline_west = format_queue_small(baseline_queues, 'west')
    baseline_east = format_queue_small(baseline_queues, 'east')
    
    rl_center_line = f"서: {rl_west} ({rl_queues['west']:2d}대) ──┼── 동: {rl_east} ({rl_queues['east']:2d}대)"
    baseline_center_line = f"서: {baseline_west} ({baseline_queues['west']:2d}대) ──┼── 동: {baseline_east} ({baseline_queues['east']:2d}대)"
    rl_center_padded = pad_to_width(rl_center_line, 69)
    baseline_center_padded = pad_to_width(baseline_center_line, 69)
    output.append(f"{rl_center_padded} │ {baseline_center_padded}")
    output.append(f"{'':<69} │ ")
    rl_signal_padded = pad_to_width(rl_ew_signal, 69, 'center')
    baseline_signal_padded = pad_to_width(baseline_ew_signal, 69, 'center')
    output.append(f"{rl_signal_padded} │ {baseline_signal_padded}")
    output.append("")
    
    # 남쪽 차선
    rl_south = format_queue_small(rl_queues, 'south')
    baseline_south = format_queue_small(baseline_queues, 'south')
    rl_south_line = f"{rl_ns_signal} 남: {rl_south} ({rl_queues['south']:2d}대)"
    baseline_south_line = f"{baseline_ns_signal} 남: {baseline_south} ({baseline_queues['south']:2d}대)"
    rl_south_padded = pad_to_width(rl_south_line, 69)
    baseline_south_padded = pad_to_width(baseline_south_line, 69)
    output.append(f"{rl_south_padded} │ {baseline_south_padded}")
    output.append("")
    
    # 통계 비교
    rl_total = sum(rl_queues.values())
    baseline_total = sum(baseline_queues.values())
    improvement = ((baseline_total - rl_total) / max(baseline_total, 1)) * 100 if baseline_total > 0 else 0
    
    output.append("-" * 140)
    
    # 통계 비교 (고정 폭)
    rl_total_line = f"총 대기 차량: {rl_total:3d}대"
    baseline_total_line = f"총 대기 차량: {baseline_total:3d}대"
    rl_total_padded = pad_to_width(rl_total_line, 69)
    baseline_total_padded = pad_to_width(baseline_total_line, 69)
    output.append(f"{rl_total_padded} │ {baseline_total_padded}")
    
    # 개선율 표시 (고정 폭)
    if improvement > 0:
        improvement_line = f"개선율: {improvement:>5.1f}% 감소"
        improvement_padded = pad_to_width(improvement_line, 69)
        output.append(f"{improvement_padded} │ {'':<69}")
    elif improvement < 0:
        improvement_line = f"개선율: {abs(improvement):>5.1f}% 증가"
        improvement_padded = pad_to_width(improvement_line, 69)
        output.append(f"{improvement_padded} │ {'':<69}")
    else:
        improvement_line = f"개선율: 동일"
        improvement_padded = pad_to_width(improvement_line, 69)
        output.append(f"{improvement_padded} │ {'':<69}")
    
    # 행동 정보 (고정 폭)
    if rl_action is not None:
        action_text = "신호 변경" if rl_action == 1 else "신호 유지"
        rl_action_line = f"행동: {action_text}"
        baseline_action_line = f"행동: 고정 주기"
        rl_action_padded = pad_to_width(rl_action_line, 69)
        baseline_action_padded = pad_to_width(baseline_action_line, 69)
        output.append(f"{rl_action_padded} │ {baseline_action_padded}")
    
    # Q-value 정보 (고정 폭)
    if rl_q_values is not None:
        q_value_line = f"Q-value: 유지={rl_q_values[0]:6.2f} 변경={rl_q_values[1]:6.2f}"
        q_value_padded = pad_to_width(q_value_line, 69)
        output.append(f"{q_value_padded} │ {'':<69}")
    
    # 신호 지속 시간 (고정 폭)
    rl_duration_line = f"신호 지속: {rl_info['phase_duration']:3d}초"
    baseline_duration_line = f"신호 지속: {baseline_info['phase_duration']:3d}초"
    rl_duration_padded = pad_to_width(rl_duration_line, 69)
    baseline_duration_padded = pad_to_width(baseline_duration_line, 69)
    output.append(f"{rl_duration_padded} │ {baseline_duration_padded}")
    
    output.append("=" * 140)
    
    return "\n".join(output)


def run_comparison_demo(
    scenario: str,
    model_path: str,
    data_path: str = None,
    agent_type: str = 'dqn',
    speed: float = 1.0,
    max_steps: int = None,
    baseline_cycle: int = 30
):
    """
    고정 신호와 학습된 모델을 나란히 비교하는 시연
    
    Args:
        scenario: 시나리오 이름
        model_path: 모델 파일 경로
        data_path: 교통 데이터 파일 경로
        agent_type: 'dqn' 또는 'ddqn'
        speed: 시뮬레이션 속도
        max_steps: 최대 스텝 수
        baseline_cycle: 고정 신호 주기 (초)
    """
    # 데이터 경로 설정
    if data_path is None:
        data_path = f'./demo_data/{scenario}_traffic_data.json'
    
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        print(f"   먼저 'python generate_demo_data.py --scenario {scenario}'를 실행하세요.")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    # 환경 및 에이전트 초기화
    print(f"\n🚀 성능 비교 시연 시작: {scenario} 시나리오")
    print(f"   모델: {model_path}")
    print(f"   데이터: {data_path}")
    print(f"   알고리즘: {agent_type.upper()}")
    print(f"   고정 신호 주기: {baseline_cycle}초")
    print("\n준비 중...")
    time.sleep(1)
    
    # 두 환경 생성 (동일한 데이터 사용)
    rl_env = DemoEnvironment(data_path)
    baseline_env = DemoEnvironment(data_path)
    
    # 에이전트 및 컨트롤러 초기화
    rl_agent = load_model(model_path, agent_type)
    baseline_controller = FixedTimeController(cycle_time=baseline_cycle)
    
    # 시뮬레이션 시작
    rl_state = rl_env.reset()
    baseline_state = baseline_env.reset()
    baseline_controller.reset()
    
    step_count = 0
    
    rl_stats = {
        'total_waiting_time': 0,
        'total_vehicles_passed': 0,
        'signal_changes': 0,
        'total_reward': 0
    }
    
    baseline_stats = {
        'total_waiting_time': 0,
        'total_vehicles_passed': 0,
        'signal_changes': 0,
        'total_reward': 0
    }
    
    try:
        while True:
            clear_screen()
            
            # 강화학습 모델 행동 선택
            rl_action = rl_agent.select_action(rl_state, training=False)
            
            # Q-value 계산
            with torch.no_grad():
                state_tensor = torch.FloatTensor(rl_state).unsqueeze(0).to(rl_agent.device)
                rl_q_values = rl_agent.q_network(state_tensor).cpu().numpy()[0]
            
            # 고정 신호 행동 선택
            baseline_action = baseline_controller.get_action(baseline_state)
            
            # 환경 스텝
            rl_next_state, rl_reward, rl_done, rl_info = rl_env.step(rl_action)
            baseline_next_state, baseline_reward, baseline_done, baseline_info = baseline_env.step(baseline_action)
            
            # 통계 업데이트
            step_count += 1
            rl_stats['total_waiting_time'] = rl_info.get('total_waiting_time', 0)
            rl_stats['total_vehicles_passed'] = rl_info.get('total_vehicles_passed', 0)
            rl_stats['total_reward'] += rl_reward
            if rl_action == 1:
                rl_stats['signal_changes'] += 1
            
            baseline_stats['total_waiting_time'] = baseline_info.get('total_waiting_time', 0)
            baseline_stats['total_vehicles_passed'] = baseline_info.get('total_vehicles_passed', 0)
            baseline_stats['total_reward'] += baseline_reward
            if baseline_action == 1:
                baseline_stats['signal_changes'] += 1
            
            # 현재 상태 정보
            rl_env_info = rl_env.get_current_state_info()
            rl_env_info['step'] = step_count
            
            baseline_env_info = baseline_env.get_current_state_info()
            baseline_env_info['step'] = step_count
            
            # 비교 시각화
            comparison_display = draw_comparison_view(
                rl_env_info, baseline_env_info, step_count, rl_action, rl_q_values.tolist()
            )
            print(comparison_display)
            
            # 누적 통계 (고정 폭)
            print(f"\n  누적 통계:")
            header_left = pad_to_width('강화학습 모델', 28, 'center')
            header_right = pad_to_width('고정 신호', 28, 'center')
            print(f"  {header_left} │ {header_right}")
            print(f"  {'-'*28} │ {'-'*28}")
            
            rl_avg_waiting = (rl_stats['total_waiting_time'] / max(1, rl_stats['total_vehicles_passed']))
            baseline_avg_waiting = (baseline_stats['total_waiting_time'] / max(1, baseline_stats['total_vehicles_passed']))
            waiting_improvement = ((baseline_avg_waiting - rl_avg_waiting) / max(baseline_avg_waiting, 0.1)) * 100
            
            # 통계 출력 (고정 폭)
            rl_vehicles_line = f"통과 차량: {rl_stats['total_vehicles_passed']:4d}대"
            baseline_vehicles_line = f"통과 차량: {baseline_stats['total_vehicles_passed']:4d}대"
            rl_vehicles_padded = pad_to_width(rl_vehicles_line, 28)
            baseline_vehicles_padded = pad_to_width(baseline_vehicles_line, 28)
            print(f"  {rl_vehicles_padded} │ {baseline_vehicles_padded}")
            
            rl_waiting_line = f"평균 대기시간: {rl_avg_waiting:6.2f}초"
            baseline_waiting_line = f"평균 대기시간: {baseline_avg_waiting:6.2f}초"
            rl_waiting_padded = pad_to_width(rl_waiting_line, 28)
            baseline_waiting_padded = pad_to_width(baseline_waiting_line, 28)
            print(f"  {rl_waiting_padded} │ {baseline_waiting_padded}")
            
            if waiting_improvement > 0:
                improvement_line = f"대기시간 개선: {waiting_improvement:>5.1f}% 감소"
                improvement_padded = pad_to_width(improvement_line, 28)
                print(f"  {improvement_padded} │ {'':<28}")
            
            rl_changes_line = f"신호 변경: {rl_stats['signal_changes']:3d}회"
            baseline_changes_line = f"신호 변경: {baseline_stats['signal_changes']:3d}회"
            rl_changes_padded = pad_to_width(rl_changes_line, 28)
            baseline_changes_padded = pad_to_width(baseline_changes_line, 28)
            print(f"  {rl_changes_padded} │ {baseline_changes_padded}")
            
            rl_reward_line = f"총 Reward: {rl_stats['total_reward']:8.2f}"
            baseline_reward_line = f"총 Reward: {baseline_stats['total_reward']:8.2f}"
            rl_reward_padded = pad_to_width(rl_reward_line, 28)
            baseline_reward_padded = pad_to_width(baseline_reward_line, 28)
            print(f"  {rl_reward_padded} │ {baseline_reward_padded}")
            
            print("\n  [Ctrl+C로 종료]")
            
            # 종료 조건
            if rl_done or baseline_done or (max_steps and step_count >= max_steps):
                break
            
            rl_state = rl_next_state
            baseline_state = baseline_next_state
            
            # 속도 제어
            time.sleep(speed)
    
    except KeyboardInterrupt:
        print("\n\n시연이 중단되었습니다.")
    
    # 최종 통계
    clear_screen()
    print("=" * 140)
    print("성능 비교 시연 종료 - 최종 통계")
    print("=" * 140)
    
    rl_avg_waiting = (rl_stats['total_waiting_time'] / max(1, rl_stats['total_vehicles_passed']))
    baseline_avg_waiting = (baseline_stats['total_waiting_time'] / max(1, baseline_stats['total_vehicles_passed']))
    waiting_improvement = ((baseline_avg_waiting - rl_avg_waiting) / max(baseline_avg_waiting, 0.1)) * 100
    
    # 최종 통계 테이블 (고정 폭)
    header = pad_to_width('지표', 25) + pad_to_width('강화학습 모델', 20, 'right') + pad_to_width('고정 신호', 20, 'right') + pad_to_width('개선율', 15, 'right')
    print(f"\n{header}")
    print("-" * 140)
    
    step_line = pad_to_width('총 스텝', 25) + pad_to_width(str(step_count), 20, 'right') + pad_to_width(str(step_count), 20, 'right') + pad_to_width('', 15)
    print(step_line)
    
    vehicles_line = (pad_to_width('통과 차량', 25) + 
                    pad_to_width(f"{rl_stats['total_vehicles_passed']}대", 20, 'right') + 
                    pad_to_width(f"{baseline_stats['total_vehicles_passed']}대", 20, 'right') + 
                    pad_to_width('', 15))
    print(vehicles_line)
    
    waiting_line = (pad_to_width('평균 대기시간', 25) + 
                   pad_to_width(f"{rl_avg_waiting:.2f}초", 20, 'right') + 
                   pad_to_width(f"{baseline_avg_waiting:.2f}초", 20, 'right') + 
                   pad_to_width(f"{waiting_improvement:.1f}%", 15, 'right'))
    print(waiting_line)
    
    changes_line = (pad_to_width('신호 변경 횟수', 25) + 
                   pad_to_width(f"{rl_stats['signal_changes']}회", 20, 'right') + 
                   pad_to_width(f"{baseline_stats['signal_changes']}회", 20, 'right') + 
                   pad_to_width('', 15))
    print(changes_line)
    
    reward_line = (pad_to_width('총 Reward', 25) + 
                  pad_to_width(f"{rl_stats['total_reward']:.2f}", 20, 'right') + 
                  pad_to_width(f"{baseline_stats['total_reward']:.2f}", 20, 'right') + 
                  pad_to_width('', 15))
    print(reward_line)
    print("=" * 140)


def load_model(model_path: str, agent_type: str = 'dqn'):
    """
    학습된 모델 로드
    
    Args:
        model_path: 모델 파일 경로
        agent_type: 'dqn' 또는 'ddqn'
        
    Returns:
        로드된 에이전트
    """
    # 기본 하이퍼파라미터 (학습 시 사용한 값과 동일해야 함)
    params = {
        'state_dim': 7,
        'action_dim': 2,
        'learning_rate': 0.001,
        'gamma': 0.95,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_capacity': 10000,
        'batch_size': 64,
        'target_update_freq': 100
    }
    
    # 에이전트 생성
    if agent_type.lower() == 'ddqn':
        agent = DoubleDQNAgent(**params)
    else:
        agent = DQNAgent(**params)
    
    # 모델 로드
    agent.load(model_path)
    agent.epsilon = 0.0  # 평가 모드 (탐험 없음)
    
    return agent


def run_demo(
    scenario: str,
    model_path: str,
    data_path: str = None,
    agent_type: str = 'dqn',
    speed: float = 1.0,
    max_steps: int = None
):
    """
    시연 실행
    
    Args:
        scenario: 시나리오 이름
        model_path: 모델 파일 경로
        data_path: 교통 데이터 파일 경로 (None이면 자동 생성)
        agent_type: 'dqn' 또는 'ddqn'
        speed: 시뮬레이션 속도 (초 단위 대기 시간)
        max_steps: 최대 스텝 수 (None이면 전체)
    """
    # 데이터 경로 설정
    if data_path is None:
        data_path = f'./demo_data/{scenario}_traffic_data.json'
    
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        print(f"   먼저 'python generate_demo_data.py --scenario {scenario}'를 실행하세요.")
        return
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    # 환경 및 에이전트 초기화
    print(f"\n🚀 시연 시작: {scenario} 시나리오")
    print(f"   모델: {model_path}")
    print(f"   데이터: {data_path}")
    print(f"   알고리즘: {agent_type.upper()}")
    print("\n준비 중...")
    time.sleep(1)
    
    demo_env = DemoEnvironment(data_path)
    agent = load_model(model_path, agent_type)
    
    # 시뮬레이션 시작
    state = demo_env.reset()
    total_reward = 0
    step_count = 0
    
    stats = {
        'total_waiting_time': 0,
        'total_vehicles_passed': 0,
        'signal_changes': 0
    }
    
    try:
        while True:
            # 화면 지우기
            clear_screen()
            
            # 행동 선택
            action = agent.select_action(state, training=False)
            
            # Q-value 계산 (표시용)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                q_values = agent.q_network(state_tensor).cpu().numpy()[0]
            
            # 환경 스텝
            next_state, reward, done, info = demo_env.step(action)
            
            # 통계 업데이트
            total_reward += reward
            step_count += 1
            stats['total_waiting_time'] = info.get('total_waiting_time', 0)
            stats['total_vehicles_passed'] = info.get('total_vehicles_passed', 0)
            if action == 1:
                stats['signal_changes'] += 1
            
            # 현재 상태 정보
            env_info = demo_env.get_current_state_info()
            env_info['step'] = step_count
            
            # 교차로 그리기
            intersection_display = draw_intersection(env_info, action, q_values.tolist())
            print(intersection_display)
            
            # 추가 통계
            print(f"\n  누적 Reward: {total_reward:8.2f}")
            print(f"  통과 차량: {stats['total_vehicles_passed']:4d}대")
            print(f"  신호 변경 횟수: {stats['signal_changes']:3d}회")
            if stats['total_vehicles_passed'] > 0:
                avg_waiting = stats['total_waiting_time'] / stats['total_vehicles_passed']
                print(f"  평균 대기시간: {avg_waiting:6.2f}초")
            
            print("\n  [Ctrl+C로 종료]")
            
            # 종료 조건
            if done or (max_steps and step_count >= max_steps):
                break
            
            state = next_state
            time.sleep(speed)
    
    except KeyboardInterrupt:
        print("\n\n시연이 중단되었습니다.")
    
    # 최종 통계
    clear_screen()
    print("=" * 70)
    print("시연 종료 - 최종 통계")
    print("=" * 70)
    print(f"  총 스텝: {step_count}")
    print(f"  총 Reward: {total_reward:.2f}")
    print(f"  통과 차량: {stats['total_vehicles_passed']}대")
    print(f"  신호 변경 횟수: {stats['signal_changes']}회")
    if stats['total_vehicles_passed'] > 0:
        avg_waiting = stats['total_waiting_time'] / stats['total_vehicles_passed']
        print(f"  평균 대기시간: {avg_waiting:.2f}초")
    print("=" * 70)


def run_demo_no_visualization(
    scenario: str,
    model_path: str,
    data_path: str = None,
    agent_type: str = 'dqn',
    max_steps: int = None
):
    """
    시각화 없이 시연 실행 (빠른 테스트용)
    
    Args:
        scenario: 시나리오 이름
        model_path: 모델 파일 경로
        data_path: 교통 데이터 파일 경로 (None이면 자동 생성)
        agent_type: 'dqn' 또는 'ddqn'
        max_steps: 최대 스텝 수 (None이면 전체)
        
    Returns:
        통계 딕셔너리
    """
    # 데이터 경로 설정
    if data_path is None:
        data_path = f'./demo_data/{scenario}_traffic_data.json'
    
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        return None
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return None
    
    demo_env = DemoEnvironment(data_path)
    agent = load_model(model_path, agent_type)
    
    # 시뮬레이션 시작
    state = demo_env.reset()
    total_reward = 0
    step_count = 0
    
    stats = {
        'total_waiting_time': 0,
        'total_vehicles_passed': 0,
        'signal_changes': 0,
        'max_queue_length': 0
    }
    
    while True:
        # 행동 선택
        action = agent.select_action(state, training=False)
        
        # 환경 스텝
        next_state, reward, done, info = demo_env.step(action)
        
        # 통계 업데이트
        total_reward += reward
        step_count += 1
        stats['total_waiting_time'] = info.get('total_waiting_time', 0)
        stats['total_vehicles_passed'] = info.get('total_vehicles_passed', 0)
        stats['max_queue_length'] = max(stats['max_queue_length'], info.get('max_queue_length', 0))
        if action == 1:
            stats['signal_changes'] += 1
        
        # 종료 조건
        if done or (max_steps and step_count >= max_steps):
            break
        
        state = next_state
    
    # 최종 통계 계산
    stats['total_reward'] = total_reward
    stats['total_steps'] = step_count
    if stats['total_vehicles_passed'] > 0:
        stats['avg_waiting_time'] = stats['total_waiting_time'] / stats['total_vehicles_passed']
    else:
        stats['avg_waiting_time'] = 0.0
    
    return stats


def run_all_scenarios_test(
    agent_type: str = 'dqn',
    max_steps: int = None,
    visualize: bool = False,
    speed: float = 1.0,
    compare: bool = True,
    baseline_cycle: int = 30
):
    """
    모든 시나리오를 순차적으로 테스트
    
    Args:
        agent_type: 'dqn' 또는 'ddqn'
        max_steps: 각 시나리오의 최대 스텝 수
        visualize: True면 시각화, False면 빠른 테스트
        speed: 시각화 모드일 때의 속도
    """
    scenarios = ['normal', 'morning_rush', 'evening_rush', 'congestion', 'night']
    all_results = {}
    
    print("=" * 70)
    print(f"🚀 모든 시나리오 테스트 시작 ({agent_type.upper()})")
    print("=" * 70)
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] {scenario} 시나리오 테스트 중...")
        
        # 모델 경로 자동 설정
        agent_dir = 'ddqn' if agent_type == 'ddqn' else 'dqn'
        model_path = f'./models/optimized/{agent_dir}_{scenario}/agent_{scenario}_optimized.pt'
        
        if visualize:
            # 시각화 모드
            if compare:
                run_comparison_demo(
                    scenario=scenario,
                    model_path=model_path,
                    agent_type=agent_type,
                    speed=speed,
                    max_steps=max_steps,
                    baseline_cycle=baseline_cycle
                )
            else:
                run_demo(
                    scenario=scenario,
                    model_path=model_path,
                    agent_type=agent_type,
                    speed=speed,
                    max_steps=max_steps
                )
            # 시각화 후 통계 수집
            stats = run_demo_no_visualization(
                scenario=scenario,
                model_path=model_path,
                agent_type=agent_type,
                max_steps=max_steps
            )
        else:
            # 빠른 테스트 모드
            stats = run_demo_no_visualization(
                scenario=scenario,
                model_path=model_path,
                agent_type=agent_type,
                max_steps=max_steps
            )
        
        if stats:
            all_results[scenario] = stats
            print(f"✅ {scenario} 완료")
            print(f"   통과 차량: {stats['total_vehicles_passed']}대")
            print(f"   평균 대기시간: {stats['avg_waiting_time']:.2f}초")
            print(f"   신호 변경: {stats['signal_changes']}회")
        else:
            print(f"❌ {scenario} 실패")
            all_results[scenario] = None
    
    # 결과 요약 출력
    print("\n" + "=" * 70)
    print("📊 전체 시나리오 테스트 결과 요약")
    print("=" * 70)
    print(f"{'시나리오':<20} {'통과 차량':>12} {'평균 대기시간':>15} {'신호 변경':>12} {'총 Reward':>12}")
    print("-" * 70)
    
    for scenario in scenarios:
        if all_results.get(scenario):
            stats = all_results[scenario]
            print(f"{scenario:<20} {stats['total_vehicles_passed']:>12}대 "
                  f"{stats['avg_waiting_time']:>14.2f}초 {stats['signal_changes']:>12}회 "
                  f"{stats['total_reward']:>11.2f}")
        else:
            print(f"{scenario:<20} {'실패':>12}")
    
    print("=" * 70)
    
    # 평균 통계
    valid_results = {k: v for k, v in all_results.items() if v is not None}
    if valid_results:
        avg_waiting = sum(s['avg_waiting_time'] for s in valid_results.values()) / len(valid_results)
        total_vehicles = sum(s['total_vehicles_passed'] for s in valid_results.values())
        total_changes = sum(s['signal_changes'] for s in valid_results.values())
        avg_reward = sum(s['total_reward'] for s in valid_results.values()) / len(valid_results)
        
        print(f"\n📈 평균 통계:")
        print(f"   평균 대기시간: {avg_waiting:.2f}초")
        print(f"   총 통과 차량: {total_vehicles}대")
        print(f"   총 신호 변경: {total_changes}회")
        print(f"   평균 Reward: {avg_reward:.2f}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='교통 신호등 제어 시연')
    parser.add_argument(
        '--scenario',
        type=str,
        default=None,
        choices=['normal', 'morning_rush', 'evening_rush', 'congestion', 'night', 'all'],
        help='시나리오 선택 (all이면 모든 시나리오 테스트)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='모델 파일 경로 (지정하지 않으면 자동으로 찾음)'
    )
    parser.add_argument(
        '--agent-type',
        type=str,
        default='dqn',
        choices=['dqn', 'ddqn'],
        help='에이전트 타입'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='시뮬레이션 속도 (초 단위 대기 시간, 기본: 1.0초, 시각화 모드에서만 사용)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=None,
        help='최대 스텝 수 (지정하지 않으면 전체)'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='시각화 없이 빠른 테스트 (모든 시나리오 테스트 시 유용)'
    )
    parser.add_argument(
        '--no-compare',
        dest='compare',
        action='store_false',
        default=True,
        help='비교 모드 비활성화 (단일 모델만 표시, 기본값: 비교 모드 활성화)'
    )
    parser.add_argument(
        '--baseline-cycle',
        type=int,
        default=30,
        help='고정 신호 주기 (초, 기본: 30초)'
    )
    
    args = parser.parse_args()
    
    # 모든 시나리오 테스트
    if args.scenario == 'all' or args.scenario is None:
        run_all_scenarios_test(
            agent_type=args.agent_type,
            max_steps=args.steps,
            visualize=not args.no_visualize,
            speed=args.speed,
            compare=args.compare,
            baseline_cycle=args.baseline_cycle
        )
    else:
        # 단일 시나리오 테스트
        # 모델 경로 자동 설정
        if args.model is None:
            agent_dir = 'ddqn' if args.agent_type == 'ddqn' else 'dqn'
            args.model = f'./models/optimized/{agent_dir}_{args.scenario}/agent_{args.scenario}_optimized.pt'
        
        # 비교 모드인지 확인
        if args.compare:
            run_comparison_demo(
                scenario=args.scenario,
                model_path=args.model,
                agent_type=args.agent_type,
                speed=args.speed,
                max_steps=args.steps,
                baseline_cycle=args.baseline_cycle
            )
        else:
            run_demo(
                scenario=args.scenario,
                model_path=args.model,
                agent_type=args.agent_type,
                speed=args.speed,
                max_steps=args.steps
            )

