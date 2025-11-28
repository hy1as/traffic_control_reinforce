"""
시연용 가상 교통 데이터 생성 스크립트
미리 생성된 교통 패턴을 저장하여 시연 시 일관된 데이터 사용
"""

import numpy as np
import json
import os
from collections import deque
from traffic_env import TrafficEnvironment


def generate_traffic_sequence(env: TrafficEnvironment, num_steps: int, seed: int = 42):
    """
    고정된 시드로 교통 시퀀스 생성
    
    Args:
        env: 교통 환경
        num_steps: 생성할 스텝 수
        seed: 랜덤 시드
        
    Returns:
        교통 시퀀스 리스트 (각 스텝의 차량 도착 정보)
    """
    np.random.seed(seed)
    
    sequence = []
    for step in range(num_steps):
        # 각 방향별 도착 차량 수 (Poisson 분포)
        arrivals = {}
        for direction in ['north', 'south', 'east', 'west']:
            num_vehicles = np.random.poisson(env.arrival_rates[direction])
            arrivals[direction] = num_vehicles
        
        sequence.append({
            'step': step,
            'arrivals': arrivals
        })
    
    return sequence


def save_demo_data(scenario: str, num_steps: int = 500, seed: int = 42):
    """
    시연용 데이터 생성 및 저장
    
    Args:
        scenario: 시나리오 이름
        num_steps: 생성할 스텝 수
        seed: 랜덤 시드
    """
    env = TrafficEnvironment()
    env.set_scenario(scenario)
    
    print(f"📊 {scenario} 시나리오 데이터 생성 중...")
    print(f"   스텝 수: {num_steps}")
    print(f"   시드: {seed}")
    
    sequence = generate_traffic_sequence(env, num_steps, seed)
    
    # 저장 디렉토리 생성
    save_dir = './demo_data'
    os.makedirs(save_dir, exist_ok=True)
    
    # JSON으로 저장
    save_path = os.path.join(save_dir, f'{scenario}_traffic_data.json')
    with open(save_path, 'w') as f:
        json.dump({
            'scenario': scenario,
            'arrival_rates': env.arrival_rates,
            'num_steps': num_steps,
            'seed': seed,
            'sequence': sequence
        }, f, indent=2)
    
    print(f"✅ 데이터 저장 완료: {save_path}")
    
    # 통계 출력
    total_arrivals = {d: sum(s['arrivals'][d] for s in sequence) for d in ['north', 'south', 'east', 'west']}
    print(f"\n📈 생성된 데이터 통계:")
    print(f"   총 도착 차량:")
    for direction, count in total_arrivals.items():
        print(f"     {direction:6s}: {count:4d}대")
    print(f"   총계: {sum(total_arrivals.values())}대")
    
    return save_path


def generate_all_scenarios():
    """모든 시나리오의 시연 데이터 생성"""
    scenarios = ['normal', 'morning_rush', 'evening_rush', 'congestion', 'night']
    
    print("="*60)
    print("시연용 가상 교통 데이터 생성")
    print("="*60)
    
    for scenario in scenarios:
        print(f"\n[{scenario}]")
        save_demo_data(scenario, num_steps=500, seed=42)
    
    print("\n" + "="*60)
    print("✅ 모든 시나리오 데이터 생성 완료!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='시연용 가상 교통 데이터 생성')
    parser.add_argument(
        '--scenario',
        type=str,
        default=None,
        help='생성할 시나리오 (지정하지 않으면 모든 시나리오 생성)'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=500,
        help='생성할 스텝 수'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='랜덤 시드'
    )
    
    args = parser.parse_args()
    
    if args.scenario:
        save_demo_data(args.scenario, args.steps, args.seed)
    else:
        generate_all_scenarios()

