import numpy as np

class LinearDataGenerator:
    # 선형 회귀를 위한 데이터 생성 클래스
    
    def __init__(self, W_true=2, b_true=5, seed=42):
        #
        """Parameters
        W_true : float
            실제 가중치 (기울기)
        b_true : float
            실제 편향 (절편)
        seed : int
            난수 시드
        """
        self.W_true = W_true
        self.b_true = b_true
        self.seed = seed
        np.random.seed(seed)
    
    def generate_data(self, n_samples=100, noise_std=1.0):
        """
        선형 관계를 가진 데이터 생성
        
        Parameters:
        -----------
        n_samples : int
            생성할 샘플 개수
        noise_std : float
            노이즈의 표준편차
            
        Returns:
        --------
        X : ndarray, shape (n_samples, 1)
            입력 데이터
        y : ndarray, shape (n_samples, 1)
            타겟 데이터
        """
        X = 2 * np.random.rand(n_samples, 1)
        noise = noise_std * np.random.randn(n_samples, 1)
        y = self.b_true + self.W_true * X + noise
        
        return X, y
    
    def get_true_parameters(self):
        """실제 파라미터 반환"""
        return self.W_true, self.b_true


if __name__ == "__main__":
    # 테스트 코드
    generator = LinearDataGenerator(W_true=2, b_true=5, seed=42)
    X, y = generator.generate_data(n_samples=100)
    
    print("=" * 50)
    print("데이터 생성 완료")
    print("=" * 50)
    print(f"생성된 데이터 개수: {len(X)}")
    print(f"X 범위: [{X.min():.2f}, {X.max():.2f}]")
    print(f"y 범위: [{y.min():.2f}, {y.max():.2f}]")
    print(f"\n실제 파라미터:")
    W_true, b_true = generator.get_true_parameters()
    print(f"  W (가중치): {W_true}")
    print(f"  b (편향): {b_true}")
    print("=" * 50)