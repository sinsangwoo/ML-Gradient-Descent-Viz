"""
경사하강법 선형 회귀 시각화 메인 실행 파일

이 스크립트는 다음을 수행함.
1. 선형 관계를 가진 데이터 생성
2. 경사하강법으로 모델 학습
3. 학습 과정을 4가지 그래프로 시각화
"""

from data_generator import LinearDataGenerator
from gradient_descent import GradientDescentRegressor
from visualizer import GradientDescentVisualizer


def main():
    """메인 실행 함수"""
    
    print("="*60)
    print("경사하강법 선형 회귀 시각화 프로그램")
    print("="*60)
    
    # ========================================
    # 1. 데이터 생성
    # ========================================
    print("\n[1단계] 데이터 생성 중...")
    print("-"*60)
    
    # 실제 파라미터 설정
    W_true = 2
    b_true = 5
    
    # 데이터 생성기 초기화
    data_gen = LinearDataGenerator(W_true=W_true, b_true=b_true, seed=42)
    X, y = data_gen.generate_data(n_samples=100, noise_std=1.0)
    
    print(f"✓ 데이터 생성 완료")
    print(f"  - 샘플 개수: {len(X)}")
    print(f"  - X 범위: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  - y 범위: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  - 실제 파라미터: W={W_true}, b={b_true}")
    
    # ========================================
    # 2. 모델 학습
    # ========================================
    print("\n[2단계] 경사하강법으로 모델 학습 중...")
    print("-"*60)
    
    # 하이퍼파라미터 설정
    learning_rate = 0.1
    epochs = 1000
    
    # 모델 생성 및 학습
    model = GradientDescentRegressor(
        learning_rate=learning_rate,
        epochs=epochs,
        random_seed=42
    )
    
    print(f"하이퍼파라미터:")
    print(f"  - 학습률(Learning Rate): {learning_rate}")
    print(f"  - 학습 반복 횟수(Epochs): {epochs}")
    print()
    
    model.fit(X, y, verbose=True)
    
    # 학습 결과 확인
    params = model.get_parameters()
    print(f"\n학습 결과 비교:")
    print(f"  실제 W: {W_true:.4f} | 학습된 W: {params['W']:.4f} | 오차: {abs(W_true - params['W']):.4f}")
    print(f"  실제 b: {b_true:.4f} | 학습된 b: {params['b']:.4f} | 오차: {abs(b_true - params['b']):.4f}")
    
    # ========================================
    # 3. 시각화
    # ========================================
    print("\n[3단계] 학습 과정 시각화...")
    print("-"*60)
    
    # 시각화 객체 생성
    visualizer = GradientDescentVisualizer()
    
    # 모든 그래프 순서대로 출력
    visualizer.visualize_all(X, y, model, W_true, b_true)
    
    # ========================================
    # 4. 예측 테스트
    # ========================================
    print("\n[4단계] 예측 테스트...")
    print("-"*60)
    
    import numpy as np
    X_test = np.array([[0.5], [1.0], [1.5], [2.0]])
    y_pred = model.predict(X_test)
    
    print("새로운 데이터에 대한 예측:")
    for x, y_p in zip(X_test, y_pred):
        y_true_calc = b_true + W_true * x[0]
        print(f"  X={x[0]:.1f} → 예측: {y_p[0]:.3f} | 실제: {y_true_calc:.3f} | 오차: {abs(y_p[0] - y_true_calc):.3f}")
    
    print("\n" + "="*60)
    print("프로그램 실행 완료!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()