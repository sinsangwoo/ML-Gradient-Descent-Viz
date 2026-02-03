import numpy as np

class GradientDescentRegressor:
    """경사하강법을 사용한 선형 회귀 모델 (Enhanced with convergence monitoring)"""
    
    def __init__(self, learning_rate=0.1, epochs=1000, random_seed=None, monitor_convergence=False):
        """
        Parameters:
        -----------
        learning_rate : float
            학습률
        epochs : int
            학습 반복 횟수
        random_seed : int, optional
            가중치 초기화를 위한 난수 시드
        monitor_convergence : bool
            이론적 수렴 속도 모니터링 여부
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed
        self.monitor_convergence = monitor_convergence
        
        # 모델 파라미터
        self.W = None
        self.b = None
        
        # 학습 이력
        self.w_history = []
        self.b_history = []
        self.loss_history = []
        self.gradient_norms = []  # Track gradient magnitudes
        
        # Convergence theory objects (lazy initialization)
        self._convergence_analyzer = None
        self._stability_analyzer = None
        
    def _initialize_parameters(self):
        """가중치와 편향 초기화"""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self.W = np.random.randn(1, 1)
        self.b = np.random.randn(1, 1)
        
    def _compute_loss(self, X, y):
        """
        평균 제곱 오차(MSE) 계산
        
        Parameters:
        -----------
        X : ndarray
            입력 데이터
        y : ndarray
            실제 타겟 값
            
        Returns:
        --------
        loss : float
            평균 제곱 오차
        """
        y_pred = X @ self.W + self.b
        error = y_pred - y
        loss = np.mean(error**2)
        return loss
    
    def _compute_gradients(self, X, y, n_samples):
        """
        손실 함수에 대한 기울기(그래디언트) 계산
        
        Parameters:
        -----------
        X : ndarray
            입력 데이터
        y : ndarray
            실제 타겟 값
        n_samples : int
            샘플 개수
            
        Returns:
        --------
        grad_W : ndarray
            가중치에 대한 기울기
        grad_b : float
            편향에 대한 기울기
        """
        y_pred = X @ self.W + self.b
        error = y_pred - y
        
        grad_W = (2/n_samples) * X.T @ error
        grad_b = (2/n_samples) * np.sum(error)
        
        return grad_W, grad_b
    
    def fit(self, X, y, verbose=True):
        """
        경사하강법으로 모델 학습
        
        Parameters:
        -----------
        X : ndarray, shape (n_samples, 1)
            입력 데이터
        y : ndarray, shape (n_samples, 1)
            타겟 데이터
        verbose : bool
            학습 진행 상황 출력 여부
            
        Returns:
        --------
        self : GradientDescentRegressor
            학습된 모델
        """
        # Initialize convergence monitoring if enabled
        if self.monitor_convergence:
            from theory.convergence_proof import ConvergenceAnalyzer
            from theory.numerical_stability import NumericalStabilityAnalyzer
            
            self._convergence_analyzer = ConvergenceAnalyzer(X, y)
            self._stability_analyzer = NumericalStabilityAnalyzer(dtype=X.dtype)
            
            if verbose:
                print("\n" + "="*70)
                print("THEORETICAL CONVERGENCE ANALYSIS (PRE-TRAINING)")
                print("="*70)
                self._convergence_analyzer.print_analysis()
                
                # Verify learning rate
                verification = self._convergence_analyzer.verify_convergence_guarantee(self.learning_rate)
                print("\n[Learning Rate Verification]")
                if verification['converges']:
                    print(f"  ✓ Learning rate η={self.learning_rate} will converge")
                    print(f"  Convergence rate ρ = {verification['convergence_rate']:.6f}")
                else:
                    print(f"  ✗ WARNING: Learning rate η={self.learning_rate} may diverge!")
                    print(f"  Convergence rate ρ = {verification['convergence_rate']:.6f}")
                    print(f"  Recommended: η < {verification['stability_limit']:.6f}")
                print("="*70 + "\n")
        
        # 파라미터 초기화
        self._initialize_parameters()
        n_samples = len(X)
        
        # 초기 상태 기록
        self.w_history = []
        self.b_history = []
        self.loss_history = []
        self.gradient_norms = []
        
        # 경사하강법 반복
        for step in range(self.epochs + 1):
            # 현재 파라미터 저장
            self.w_history.append(self.W.item())
            self.b_history.append(self.b.item())
            
            # 손실 계산 및 저장
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)
            
            # Numerical stability monitoring
            if self.monitor_convergence and self._stability_analyzer:
                self._stability_analyzer.monitor_loss(loss, step)
            
            # 진행 상황 출력
            if verbose and step % 100 == 0:
                print(f"Step {step:4d} | Loss: {loss:.6f} | W: {self.W.item():.4f} | b: {self.b.item():.4f}")
            
            # 마지막 스텝에서는 업데이트 하지 않음
            if step == self.epochs:
                break
            
            # 그래디언트 계산
            grad_W, grad_b = self._compute_gradients(X, y, n_samples)
            
            # Track gradient norm
            grad_norm = np.sqrt(np.sum(grad_W**2) + grad_b**2)
            self.gradient_norms.append(grad_norm)
            
            # Monitor gradient stability
            if self.monitor_convergence and self._stability_analyzer:
                gradient_vector = np.array([[grad_W.item()], [grad_b]])
                self._stability_analyzer.monitor_gradient(gradient_vector, step)
            
            # Store old parameters for update monitoring
            if self.monitor_convergence:
                theta_old = np.array([[self.W.item()], [self.b.item()]])
            
            # 파라미터 업데이트 (경사하강법)
            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b
            
            # Monitor parameter update
            if self.monitor_convergence and self._stability_analyzer:
                theta_new = np.array([[self.W.item()], [self.b.item()]])
                self._stability_analyzer.monitor_parameter_update(
                    theta_old, theta_new, step, self.learning_rate
                )
        
        if verbose:
            print("\n" + "="*50)
            print("학습 완료!")
            print("="*50)
            print(f"최종 W (가중치): {self.W.item():.4f}")
            print(f"최종 b (편향): {self.b.item():.4f}")
            print(f"최종 Loss: {loss:.6f}")
            print("="*50)
            
            # Print stability analysis if monitoring enabled
            if self.monitor_convergence and self._stability_analyzer:
                print("\n")
                self._stability_analyzer.print_stability_report()
        
        return self
    
    def predict(self, X):
        """
        새로운 데이터에 대한 예측
        
        Parameters:
        -----------
        X : ndarray
            입력 데이터
            
        Returns:
        --------
        y_pred : ndarray
            예측 값
        """
        if self.W is None or self.b is None:
            raise ValueError("모델이 학습되지 않았습니다. fit() 메서드를 먼저 호출하세요.")
        return X @ self.W + self.b
    
    def get_parameters(self):
        """현재 모델 파라미터 반환"""
        return {
            'W': self.W.item() if self.W is not None else None,
            'b': self.b.item() if self.b is not None else None
        }
    
    def get_history(self):
        """학습 이력 반환"""
        return {
            'w_history': self.w_history,
            'b_history': self.b_history,
            'loss_history': self.loss_history,
            'gradient_norms': self.gradient_norms
        }
    
    def get_convergence_analyzer(self):
        """Get convergence analyzer (if monitoring was enabled)"""
        return self._convergence_analyzer
    
    def get_stability_analyzer(self):
        """Get stability analyzer (if monitoring was enabled)"""
        return self._stability_analyzer


if __name__ == "__main__":
    # 테스트 코드
    print("경사하강법 모델 테스트 (with convergence monitoring)\n")
    
    # 간단한 데이터 생성
    np.random.seed(42)
    X_test = 2 * np.random.rand(50, 1)
    y_test = 5 + 2 * X_test + np.random.randn(50, 1)
    
    # 모델 생성 및 학습 (with monitoring)
    model = GradientDescentRegressor(
        learning_rate=0.1, 
        epochs=500, 
        random_seed=42,
        monitor_convergence=True  # Enable convergence monitoring
    )
    model.fit(X_test, y_test, verbose=True)
    
    # 예측
    X_new = np.array([[0.5], [1.0], [1.5]])
    y_pred = model.predict(X_new)
    
    print("\n예측 결과:")
    for i, (x, y) in enumerate(zip(X_new, y_pred)):
        print(f"X = {x[0]:.1f} → 예측 y = {y[0]:.2f}")