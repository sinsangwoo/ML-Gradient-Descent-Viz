import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
import urllib.request
import os
import ssl

class GradientDescentVisualizer:
    """경사하강법 학습 과정 시각화 클래스"""
    
    def __init__(self):
        """폰트 설정 및 초기화"""
        self._setup_font()
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def _setup_font(self):
        """한글 폰트 설정"""
        font_path = 'NanumGothic.ttf'
        
        # 폰트 파일이 없으면 다운로드
        if not os.path.exists(font_path):
            print("나눔고딕 폰트를 다운로드합니다...")
            url = 'https://raw.githubusercontent.com/google/fonts/main/ofl/nanumgothic/NanumGothic-Regular.ttf'
            try:
                context = ssl._create_unverified_context()
                with urllib.request.urlopen(url, context=context) as u, open(font_path, 'wb') as f:
                    f.write(u.read())
                print("다운로드 완료.")
            except Exception as e:
                print(f"폰트 다운로드 중 오류가 발생했습니다: {e}")
                font_path = None
        
        # matplotlib에 폰트 등록
        if font_path:
            self.font_prop = fm.FontProperties(fname=font_path)
            plt.rc('font', family=self.font_prop.get_name())
        else:
            self.font_prop = fm.FontProperties()
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_final_result(self, X, y, W, b, W_true=None, b_true=None):
        """
        최종 학습 결과 시각화
        
        Parameters:
        -----------
        X : ndarray
            입력 데이터
        y : ndarray
            타겟 데이터
        W : float
            학습된 가중치
        b : float
            학습된 편향
        W_true : float, optional
            실제 가중치
        b_true : float, optional
            실제 편향
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(X, y, alpha=0.6, label='원본 데이터')
        plt.plot(X, X * W + b, color='red', linewidth=3, label='학습된 회귀선')
        plt.title('1. 최종 학습 결과', fontsize=18, fontproperties=self.font_prop)
        plt.xlabel('X', fontsize=14)
        plt.ylabel('y', fontsize=14)
        plt.legend(prop=self.font_prop)
        
        # 학습 결과 텍스트 박스
        text = f'Final W: {W:.3f}\nFinal b: {b:.3f}'
        if W_true is not None and b_true is not None:
            text += f'\n\nTrue W: {W_true}\nTrue b: {b_true}'
        plt.text(0.05, 0.95, text,
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_loss_convergence(self, loss_history, epochs):
        """
        손실 함수 수렴 과정 시각화
        
        Parameters:
        -----------
        loss_history : list
            손실 이력
        epochs : int
            총 학습 스텝 수
        """
        plt.figure(figsize=(8, 6))
        plt.plot(range(epochs + 1), loss_history, color='blue', label='Loss')
        plt.title('2. 손실(Loss) 함수의 수렴 과정', fontsize=18, fontproperties=self.font_prop)
        plt.xlabel('학습 스텝 (Steps)', fontsize=14, fontproperties=self.font_prop)
        plt.ylabel('Loss (MSE)', fontsize=14)
        plt.yscale('log')
        plt.legend(prop=self.font_prop)
        plt.annotate('손실이 최솟값으로 수렴', fontproperties=self.font_prop,
                     xy=(epochs, min(loss_history)), xytext=(epochs*0.5, np.median(loss_history)),
                     arrowprops=dict(facecolor='black', shrink=0.05), ha='center')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_contour_path(self, X, y, w_history, b_history, W_true, b_true, final_W, final_b):
        """
        손실 함수 등고선과 최적화 경로 시각화
        
        Parameters:
        -----------
        X : ndarray
            입력 데이터
        y : ndarray
            타겟 데이터
        w_history : list
            가중치 이력
        b_history : list
            편향 이력
        W_true : float
            실제 가중치
        b_true : float
            실제 편향
        final_W : float
            최종 가중치
        final_b : float
            최종 편향
        """
        # 손실 함수 곡면 계산
        W_grid = np.linspace(final_W - 3, final_W + 3, 100)
        b_grid = np.linspace(final_b - 3, final_b + 3, 100)
        W_mesh, b_mesh = np.meshgrid(W_grid, b_grid)
        loss_mesh = np.zeros_like(W_mesh)
        
        for i in range(W_mesh.shape[0]):
            for j in range(W_mesh.shape[1]):
                y_pred_mesh = X * W_mesh[i, j] + b_mesh[i, j]
                loss_mesh[i, j] = np.mean((y_pred_mesh - y)**2)
        
        # 등고선 플롯
        fig, ax = plt.subplots(figsize=(9, 7))
        contour = ax.contourf(W_mesh, b_mesh, loss_mesh, 
                              levels=np.logspace(np.log10(np.min(loss_mesh)+0.01), 
                                               np.log10(np.max(loss_mesh)), 50), 
                              cmap='viridis', alpha=0.8)
        fig.colorbar(contour, ax=ax, label='Loss')
        
        # 최적화 경로
        ax.plot(w_history, b_history, 'r-o', markersize=3, linewidth=1.5, label='(W, b) 이동 경로')
        ax.plot(w_history[0], b_history[0], 'go', markersize=12, label='시작점 (W₀, b₀)')
        ax.plot(W_true, b_true, 'y*', markersize=15, markeredgecolor='black', label='실제 최적점')
        
        ax.set_title('3. 손실 함수 등고선과 최적화 경로', fontsize=18, fontproperties=self.font_prop)
        ax.set_xlabel('가중치 (W)', fontsize=14, fontproperties=self.font_prop)
        ax.set_ylabel('편향 (b)', fontsize=14, fontproperties=self.font_prop)
        ax.legend(prop=self.font_prop)
        ax.text(w_history[10], b_history[10] + 0.5, '기울기의\n반대방향으로 이동', 
                color='white', fontsize=12, ha='center', fontproperties=self.font_prop)
        plt.tight_layout()
        plt.show()
        
        return W_mesh, b_mesh, loss_mesh
    
    def plot_3d_surface(self, W_mesh, b_mesh, loss_mesh, w_history, b_history, loss_history, W_true, b_true):
        """
        3D 손실 함수 곡면과 최적화 경로 시각화
        
        Parameters:
        -----------
        W_mesh : ndarray
            가중치 메쉬 그리드
        b_mesh : ndarray
            편향 메쉬 그리드
        loss_mesh : ndarray
            손실 값 메쉬 그리드
        w_history : list
            가중치 이력
        b_history : list
            편향 이력
        loss_history : list
            손실 이력
        W_true : float
            실제 가중치
        b_true : float
            실제 편향
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 손실 함수 곡면
        ax.plot_surface(W_mesh, b_mesh, loss_mesh, cmap='viridis', alpha=0.6, 
                       edgecolor='none', rstride=5, cstride=5)
        
        # 최적화 경로
        ax.plot(w_history, b_history, loss_history, 'r-o', markersize=3, linewidth=2, label='최적화 경로')
        ax.scatter(w_history[0], b_history[0], loss_history[0], c='g', s=100, label='시작점')
        ax.scatter(W_true, b_true, np.min(loss_mesh), c='yellow', marker='*', 
                  s=200, edgecolor='black', label='실제 최적점')
        
        ax.set_title('4. 3D 손실 함수 곡면(Loss Landscape)', fontsize=18, fontproperties=self.font_prop)
        ax.set_xlabel('가중치 (W)', fontsize=12)
        ax.set_ylabel('편향 (b)', fontsize=12)
        ax.set_zlabel('손실 (Loss)', fontsize=12)
        ax.view_init(elev=30, azim=135)
        ax.legend(prop=self.font_prop)
        plt.tight_layout()
        plt.show()
    
    def visualize_all(self, X, y, model, W_true, b_true):
        """
        모든 시각화를 순서대로 실행
        
        Parameters:
        -----------
        X : ndarray
            입력 데이터
        y : ndarray
            타겟 데이터
        model : GradientDescentRegressor
            학습된 모델
        W_true : float
            실제 가중치
        b_true : float
            실제 편향
        """
        # 모델에서 데이터 추출
        params = model.get_parameters()
        history = model.get_history()
        
        W = params['W']
        b = params['b']
        w_history = history['w_history']
        b_history = history['b_history']
        loss_history = history['loss_history']
        epochs = model.epochs
        
        print("\n시각화를 시작합니다...\n")
        
        # 1. 최종 학습 결과
        print("1/4: 최종 학습 결과 플롯")
        self.plot_final_result(X, y, W, b, W_true, b_true)
        
        # 2. 손실 수렴 과정
        print("2/4: 손실 함수 수렴 과정 플롯")
        self.plot_loss_convergence(loss_history, epochs)
        
        # 3. 등고선과 최적화 경로
        print("3/4: 손실 함수 등고선과 최적화 경로 플롯")
        W_mesh, b_mesh, loss_mesh = self.plot_contour_path(X, y, w_history, b_history, 
                                                            W_true, b_true, W, b)
        
        # 4. 3D 곡면
        print("4/4: 3D 손실 함수 곡면 플롯")
        self.plot_3d_surface(W_mesh, b_mesh, loss_mesh, w_history, b_history, 
                            loss_history, W_true, b_true)
        
        print("\n모든 시각화가 완료되었습니다!")


if __name__ == "__main__":
    # 간단한 테스트
    print("Visualizer 테스트\n")
    
    # 더미 데이터 생성
    np.random.seed(42)
    X_test = 2 * np.random.rand(50, 1)
    y_test = 5 + 2 * X_test + np.random.randn(50, 1)
    
    # 더미 학습 이력
    w_hist = [1.0, 1.5, 1.8, 2.0]
    b_hist = [3.0, 4.0, 4.5, 5.0]
    loss_hist = [10.0, 5.0, 2.0, 1.0]
    
    # 시각화 객체 생성
    viz = GradientDescentVisualizer()
    
    # 개별 플롯 테스트
    print("최종 결과 플롯 테스트...")
    viz.plot_final_result(X_test, y_test, 2.0, 5.0, 2.0, 5.0)