"""
마스터 실험 스크립트
하이퍼파라미터 튜닝 → 최적 파라미터로 시나리오 비교 자동 실행
"""

import os
import sys
import subprocess
import json


def run_command(cmd: str, description: str):
    """명령어 실행"""
    print("\n" + "="*70)
    print(f"🚀 {description}")
    print("="*70)
    print(f"명령어: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ 오류 발생: {description}")
        sys.exit(1)
    
    print(f"\n✅ 완료: {description}")


def check_tuning_results_exist() -> bool:
    """하이퍼파라미터 튜닝 결과 존재 확인"""
    return os.path.exists('./results/hyperparameter_tuning_results.json')


def master_workflow(mode: str = 'full'):
    """
    마스터 워크플로우
    
    Args:
        mode: 'full' (전체), 'quick' (빠른 테스트), 'skip-tuning' (튜닝 스킵)
    """
    print("\n" + "="*70)
    print("🎯 강화학습 교통 신호등 제어 - 마스터 실험 워크플로우")
    print("="*70)
    print(f"실행 모드: {mode.upper()}")
    print("="*70)
    
    if mode == 'quick':
        # 빠른 테스트 모드
        print("\n📝 빠른 테스트 모드")
        print("   - 기본 동작 확인: 10 에피소드")
        print("   - 하이퍼파라미터 테스트: 50 에피소드")
        print("   - 예상 소요 시간: 약 10분\n")
        
        # 1. 기본 동작 확인
        run_command(
            "python quick_test.py",
            "1단계: 기본 동작 확인"
        )
        
        # 2. 하이퍼파라미터 빠른 테스트
        run_command(
            "python quick_hyperparameter_test.py",
            "2단계: 하이퍼파라미터 빠른 테스트"
        )
        
        print("\n" + "="*70)
        print("✨ 빠른 테스트 완료!")
        print("="*70)
        print("\n📌 다음 단계:")
        print("   전체 실험을 실행하려면:")
        print("   python master_experiment.py --mode full")
        
    elif mode == 'skip-tuning':
        # 튜닝 스킵 모드 (기본 파라미터 사용)
        print("\n📝 튜닝 스킵 모드")
        print("   - 기본 하이퍼파라미터 사용")
        print("   - 시나리오 비교 실험만 수행")
        print("   - 예상 소요 시간: 약 4-6시간\n")
        
        # 통합 워크플로우 (기본 파라미터)
        run_command(
            "python integrated_workflow.py --train-episodes 2000 --eval-episodes 100",
            "시나리오 비교 실험 (기본 파라미터)"
        )
        
        print("\n" + "="*70)
        print("✨ 실험 완료!")
        print("="*70)
        
    elif mode == 'full':
        # 전체 실험 모드
        print("\n📝 전체 실험 모드")
        print("   - 1단계: 하이퍼파라미터 튜닝 (실험 A~D)")
        print("   - 2단계: 최적 파라미터로 시나리오 비교")
        print("   - 예상 소요 시간: 약 12-16시간\n")
        
        input("⚠️  전체 실험은 오랜 시간이 걸립니다. 계속하시겠습니까? [Enter]")
        
        # 1단계: 하이퍼파라미터 튜닝
        if not check_tuning_results_exist():
            run_command(
                "python hyperparameter_tuning.py",
                "1단계: 하이퍼파라미터 튜닝 (실험 A~D)"
            )
        else:
            print("\n✅ 하이퍼파라미터 튜닝 결과 이미 존재")
            print("   기존 결과를 사용합니다.")
        
        # 2단계: 최적 파라미터로 시나리오 비교
        run_command(
            "python integrated_workflow.py --use-tuned --train-episodes 2000 --eval-episodes 100",
            "2단계: 시나리오 비교 (최적 파라미터)"
        )
        
        print("\n" + "="*70)
        print("✨ 모든 실험 완료!")
        print("="*70)
        
        # 결과 파일 요약
        print("\n📂 생성된 결과 파일:")
        print("   ./results/hyperparameter_tuning_results.json")
        print("   ./results/integrated_experiment_results.json")
        print("   ./results/plots/hyperparameter_*.png")
        print("   ./models/optimized/")
    
    else:
        print(f"❌ 알 수 없는 모드: {mode}")
        print("   사용 가능한 모드: 'quick', 'skip-tuning', 'full'")
        sys.exit(1)


def print_usage():
    """사용법 출력"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║        강화학습 교통 신호등 제어 - 마스터 실험 스크립트            ║
╚════════════════════════════════════════════════════════════════════╝

사용법:
    python master_experiment.py [--mode MODE]

모드 옵션:
    quick        : 빠른 테스트 (~10분)
                   - 기본 동작 확인
                   - 하이퍼파라미터 빠른 테스트
    
    skip-tuning  : 튜닝 스킵 모드 (~4-6시간)
                   - 기본 파라미터로 시나리오 비교만 수행
                   - 하이퍼파라미터 튜닝 건너뛰기
    
    full         : 전체 실험 (~12-16시간) ⭐ 권장
                   - 하이퍼파라미터 튜닝 (실험 A~D)
                   - 최적 파라미터로 시나리오 비교

예시:
    # 빠른 테스트 (권장)
    python master_experiment.py --mode quick
    
    # 전체 실험
    python master_experiment.py --mode full
    
    # 기본 파라미터로 시나리오만 실험
    python master_experiment.py --mode skip-tuning

실험 워크플로우:
    
    [Mode: quick]
    1. 기본 동작 확인 (10 에피소드)
    2. 하이퍼파라미터 테스트 (50 에피소드)
    
    [Mode: skip-tuning]
    1. 시나리오 비교 (기본 파라미터)
       - normal, morning_rush, evening_rush, congestion, night
       - DQN vs Double DQN vs Baseline
    
    [Mode: full] ⭐⭐⭐
    1. 하이퍼파라미터 튜닝
       - 실험 A: Learning Rate
       - 실험 B: Discount Factor
       - 실험 C: Batch Size
       - 실험 D: Buffer Size
    
    2. 최적 파라미터 자동 선택
    
    3. 시나리오 비교 (최적 파라미터)
       - normal, morning_rush, evening_rush, congestion, night
       - DQN vs Double DQN vs Baseline

결과 파일:
    ./results/hyperparameter_tuning_results.json
    ./results/integrated_experiment_results.json
    ./results/plots/
    ./models/optimized/

╚════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='강화학습 교통 신호등 제어 - 마스터 실험',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='quick',
        choices=['quick', 'skip-tuning', 'full'],
        help='실험 모드 선택'
    )
    
    parser.add_argument(
        '--help-detail',
        action='store_true',
        help='상세 사용법 출력'
    )
    
    args = parser.parse_args()
    
    if args.help_detail:
        print_usage()
    else:
        master_workflow(mode=args.mode)