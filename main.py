import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
import urllib.request
import os
import ssl

# --- 1. 한글 폰트 설정 (어떤 환경에서도 동작) ---
# 나눔고딕 폰트 경로 설정
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
    font_prop = fm.FontProperties(fname=font_path)
    plt.rc('font', family=font_prop.get_name())
else:
    font_prop = fm.FontProperties()
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 기본 설정 및 데이터 생성 ---
np.random.seed(42)
W_true = 2
b_true = 5
X = 2 * np.random.rand(100, 1)
y = b_true + W_true * X + np.random.randn(100, 1)

# --- 3. 경사하강법 모델 구현 ---
learning_rate = 0.1
epochs = 1000
W = np.random.randn(1, 1)
b = np.random.randn(1, 1)

w_history = []
b_history = []
loss_history = []
n_samples = len(X)

for step in range(epochs + 1):
    w_history.append(W.item())
    b_history.append(b.item())
    
    y_pred = X @ W + b
    error = y_pred - y
    loss = np.mean(error**2)
    loss_history.append(loss)
    
    if step == epochs:
        break
        
    grad_W = (2/n_samples) * X.T @ error
    grad_b = (2/n_samples) * np.sum(error)
    
    W -= learning_rate * grad_W
    b -= learning_rate * grad_b

print("--- 학습 완료 ---")
final_W, final_b = w_history[-1], b_history[-1]
print(f"최종 W: {final_W:.4f} (실제 값: {W_true})")
print(f"최종 b: {final_b:.4f} (실제 값: {b_true})")

# --- 4. 미적분학적 원리 시각화 (하나씩 순서대로) ---
plt.style.use('seaborn-v0_8-whitegrid')

# ##### 플롯 1: 최종 학습 결과 #####
plt.figure(figsize=(8, 6))
plt.scatter(X, y, alpha=0.6, label='원본 데이터')
plt.plot(X, X * final_W + final_b, color='red', linewidth=3, label='학습된 회귀선')
plt.title('1. 최종 학습 결과', fontsize=18, fontproperties=font_prop)
plt.xlabel('X', fontsize=14)
plt.ylabel('y', fontsize=14)
plt.legend(prop=font_prop)
plt.text(0.05, 0.95, f'Final W: {final_W:.3f}\nFinal b: {final_b:.3f}',
         transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
plt.grid(True)
plt.tight_layout()
plt.show() # 첫 번째 그래프 보이기

# ##### 플롯 2: 손실 수렴 과정 #####
plt.figure(figsize=(8, 6))
plt.plot(range(epochs + 1), loss_history, color='blue', label='Loss')
plt.title('2. 손실(Loss) 함수의 수렴 과정', fontsize=18, fontproperties=font_prop)
plt.xlabel('학습 스텝 (Steps)', fontsize=14, fontproperties=font_prop)
plt.ylabel('Loss (MSE)', fontsize=14)
plt.yscale('log')
plt.legend(prop=font_prop)
plt.annotate('손실이 최솟값으로 수렴', fontproperties=font_prop,
             xy=(epochs, min(loss_history)), xytext=(epochs*0.5, np.median(loss_history)),
             arrowprops=dict(facecolor='black', shrink=0.05), ha='center')
plt.grid(True)
plt.tight_layout()
plt.show() # 두 번째 그래프 보이기

# 손실 함수 곡면을 그리기 위한 데이터 준비
W_grid = np.linspace(final_W - 3, final_W + 3, 100)
b_grid = np.linspace(final_b - 3, final_b + 3, 100)
W_mesh, b_mesh = np.meshgrid(W_grid, b_grid)
loss_mesh = np.zeros_like(W_mesh)
for i in range(W_mesh.shape[0]):
    for j in range(W_mesh.shape[1]):
        y_pred_mesh = X * W_mesh[i, j] + b_mesh[i, j]
        loss_mesh[i, j] = np.mean((y_pred_mesh - y)**2)

# ##### 플롯 3: 손실 함수 등고선과 최적화 경로 #####
fig, ax = plt.subplots(figsize=(9, 7))
contour = ax.contourf(W_mesh, b_mesh, loss_mesh, levels=np.logspace(np.log10(np.min(loss_mesh)+0.01), np.log10(np.max(loss_mesh)), 50), cmap='viridis', alpha=0.8)
fig.colorbar(contour, ax=ax, label='Loss')
ax.plot(w_history, b_history, 'r-o', markersize=3, linewidth=1.5, label='(W, b) 이동 경로')
ax.plot(w_history[0], b_history[0], 'go', markersize=12, label='시작점 (W₀, b₀)')
ax.plot(W_true, b_true, 'y*', markersize=15, markeredgecolor='black', label='실제 최적점')
ax.set_title('3. 손실 함수 등고선과 최적화 경로', fontsize=18, fontproperties=font_prop)
ax.set_xlabel('가중치 (W)', fontsize=14, fontproperties=font_prop)
ax.set_ylabel('편향 (b)', fontsize=14, fontproperties=font_prop)
ax.legend(prop=font_prop)
ax.text(w_history[10], b_history[10] + 0.5, '기울기의\n반대방향으로 이동', color='white', fontsize=12, ha='center', fontproperties=font_prop)
plt.tight_layout()
plt.show() # 세 번째 그래프 보이기

# ##### 플롯 4: 3D 손실 함수 곡면과 가중치 이동 경로 #####
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W_mesh, b_mesh, loss_mesh, cmap='viridis', alpha=0.6, edgecolor='none', rstride=5, cstride=5)
ax.plot(w_history, b_history, loss_history, 'r-o', markersize=3, linewidth=2, label='최적화 경로')
ax.scatter(w_history[0], b_history[0], loss_history[0], c='g', s=100, label='시작점')
ax.scatter(W_true, b_true, np.min(loss_mesh), c='yellow', marker='*', s=200, edgecolor='black', label='실제 최적점')
ax.set_title('4. 3D 손실 함수 곡면(Loss Landscape)', fontsize=18, fontproperties=font_prop)
ax.set_xlabel('가중치 (W)', fontsize=12)
ax.set_ylabel('편향 (b)', fontsize=12)
ax.set_zlabel('손실 (Loss)', fontsize=12)
ax.view_init(elev=30, azim=135)
ax.legend(prop=font_prop)
plt.tight_layout()
plt.show() # 네 번째 그래프 보이기