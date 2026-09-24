import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 데이터 불러오기 
df = pd.read_csv(r'C:\Users\yoon0\OneDrive\바탕 화면\새 폴더\gu05.csv')
df['temp'] = df['3'] # 3번 컬럼(가장 가혹한 조건) 데이터를 선택

#  샘플 데이터 생성
data = {
    'time': np.arange(0, 181),
    # 컬럼 '3' 데이터 일부 발췌 및 선형 보간으로 180초 모사
    'temp': np.interp(np.arange(0, 181), 
                      [0, 10, 30, 60, 90, 137, 180], 
                      [67.66, 75.66, 76.66, 79.66, 81.66, 85.66, 87.66]) 
}
df = pd.DataFrame(data)

# 실제 엑셀 파일 사용 시 아래 주석 해제 및 컬럼명 ('3') 지정
# df['temp'] = df['3'] 

# 2. 알고리즘 파라미터 설정
window_size = 15           # 이동 평균 윈도우 (15초)
critical_temp = 85.0       # 하드웨어 컷오프 임계 온도 (℃)
warning_rate = 1.2         # 알고리즘 경고 임계 상승률 (℃/min) - 데이터에 맞게 조절

# 3. 데이터 전처리 및 특징 추출 (알고리즘 로직 반영)
# 온도를 부드럽게 스무딩 (EWMA 또는 단순 이동 평균)
df['temp_smooth'] = df['temp'].rolling(window=window_size, min_periods=1).mean()

# 온도 상승률(dT/dt) 계산 (15초 간의 변화량을 분당 변화율 ℃/min로 변환)
df['dTdt'] = df['temp_smooth'].diff(periods=window_size) / window_size * 60.0

# 4. 임계점 도달 시간(Index) 찾기
# 가동 초기(0~20초)의 급격한 상승 스파이크를 무시하기 위해 20초 이후부터 탐색
search_range = df['time'] > 20 

# 물리적 컷오프 도달 시간 (절대 온도 85도 돌파)
t_crit_idx = df[search_range & (df['temp_smooth'] >= critical_temp)].index.min()
t_crit = df.loc[t_crit_idx, 'time'] if pd.notna(t_crit_idx) else None

# 알고리즘 경고 도달 시간 (dT/dt 임계치 돌파)
t_warn_idx = df[search_range & (df['dTdt'] >= warning_rate)].index.min()
t_warn = df.loc[t_warn_idx, 'time'] if pd.notna(t_warn_idx) else None

# 5. 그래프 그리기 (이중 축)
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300) # 논문용 고해상도(300 dpi)

color1 = 'tab:blue'
ax1.set_xlabel('Time (sec)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Absolute Temperature (℃)', color=color1, fontsize=12, fontweight='bold')
line1 = ax1.plot(df['time'], df['temp_smooth'], color=color1, linewidth=2, label='Temperature')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.6)

# 두 번째 Y축 (dT/dt)
ax2 = ax1.twinx()  
color2 = 'tab:red'
ax2.set_ylabel('Temp Rate of Change (dT/dt, ℃/min)', color=color2, fontsize=12, fontweight='bold')
line2 = ax2.plot(df['time'], df['dTdt'], color=color2, linestyle='-.', linewidth=2, label='dT/dt')
ax2.tick_params(axis='y', labelcolor=color2)

# 6. 임계선 및 어노테이션 (마커 표기) 추가
if t_crit is not None:
    ax1.axhline(y=critical_temp, color='blue', linestyle=':', alpha=0.7)
    ax1.plot(t_crit, critical_temp, 'bx', markersize=12, markeredgewidth=2)
    ax1.text(t_crit + 5, critical_temp - 1, f'Failure (85℃)\nat {int(t_crit)}s', color='blue')

if t_warn is not None:
    ax2.axhline(y=warning_rate, color='red', linestyle=':', alpha=0.7)
    ax2.plot(t_warn, warning_rate, 'r^', markersize=10)
    ax2.text(t_warn + 5, warning_rate + 0.1, f'Algorithm Warning\nat {int(t_warn)}s', color='red')

# 확보한 예측 골든타임 (Delta T) 화살표 그리기
if t_crit is not None and t_warn is not None and t_crit > t_warn:
    y_arrow = ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.5
    ax1.annotate('', xy=(t_crit, y_arrow), xytext=(t_warn, y_arrow),
                 arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax1.text((t_warn + t_crit)/2, y_arrow + 0.5, f'Golden Time\n$\\Delta$t = {int(t_crit - t_warn)} sec', 
             ha='center', color='green', fontweight='bold')

# 범례 통합
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.title('Thermal Transient Response & Algorithm Warning Trigger', fontsize=14, fontweight='bold')
fig.tight_layout()

# 그래프 저장 
plt.savefig('PHM_Algorithm_Validation.png', format='png', dpi=300)
plt.show()