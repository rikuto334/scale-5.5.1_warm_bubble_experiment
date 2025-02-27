import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
import time
import warnings

# 乱数シードの設定（再現性確保のため）
random_num=1 #必要に応じて乱数シードを変えてください

np.random.seed(random_num)

# 警告を抑制
warnings.filterwarnings('ignore', category=DeprecationWarning)
# matplotlib のバックエンドを設定（GUIなしでも画像保存可能）
matplotlib.use('Agg')

# ---------------------------
# シミュレーション・ファイル設定
# ---------------------------

nofpe = 2              # 使用するPE（プロセッシングエレメント）の数
fny = 2                # y方向の領域分割数
fnx = 1                # x方向の領域分割数
loop = 0               # ループ回数の初期値
varname = 'PREC'       # シミュレーション対象の変数（降水量）

# ファイル名のテンプレート（'######'は後でPE番号等で置換）
init_file = "init_00000101-000000.000.pe######.nc"
sub_init_file = "0000/init_00000101-000000.000.pe######.nc"
org_file = "init_00000101-000000.000.pe######.org.nc"
history_file = "history.pe######.nc"
sub_history_file = "0000/history.pe######.nc"
restart_file = "restart_00000101-010000.000.pe000000.nc"

# その他ファイル名（ループ番号なども含む）
orgfile = f"history-{loop}.pe######.nc"
now_file = f"now.pe######.nc"
temp_file=f"temp.pe######.nc" # 一時バックアップ用
temp2_file=f"temp2.pe######.nc" # さらに別のバックアップ用
# シミュレーション実行ディレクトリ（実行時のパス）
file_path = '/home/rk-nagai/scale-5.5.1/scale-rm/test/tutorial/ideal'#自信のディレクトリに書き換えてください
# PSO（Particle Swarm Optimization）結果出力用のファイル名（ここでは gpyoptfile という名称）
gpyoptfile=f"gpyopt-{loop}.pe######.nc" #PSOの結果を保存するファイル　名前は適当なので必要に応じて書き換えてください

# ---------------------------
# 結果記録用の配列
# ---------------------------

input_history1 = np.zeros((6, 1))  # 各ループごとの input1（MOMY）入力履歴
input_history2 = np.zeros((6, 1))  # 各ループごとの input2（y位置）入力履歴
input_history3 = np.zeros((6, 1))  # 各ループごとの input3（z位置）入力履歴
time_history   = np.zeros((6, 1))  # 各ループの実行時間の記録

sum_gpy = np.zeros(40)   # PSO適用後の累積降水量（各グリッドごと）
sum_no  = np.zeros(40)   # 制御前の累積降水量

n = 10                   # 個体数（ジョブ数）10並列するためにはcore数20以上のPCで実行してください
swarmsize = 10           # 粒子群のサイズ（PSOの個体数）

T = 6                    # 予測区間の長さ
opt_num = 10           # PSOの最大反復回数 必要に応じて変えてください
# PSO用カウンタ（乱数生成器のシード用などに利用）
init_cnt=0
r1_cnt=0

# ---------------------------
# 関数定義：predict
# ---------------------------

def predict(inputs,t):
    """
    各個体（ジョブ）の入力パラメータに基づいてシミュレーションを実行し、
    各ジョブの累積降水量（評価値）を算出する。
      inputs: 各個体の入力パラメータ（3*T 次元：連続変数＋2種類の整数変数）
      t: 予測区間のタイムステップ数
    """
    # 各PEごとに、現在の初期状態(init_file)をバックアップ(now_fileへコピー)
    for pe in range(nofpe):
            now = now_file.replace('######', str(pe).zfill(6))
            init = init_file.replace('######', str(pe).zfill(6))
            subprocess.run(["cp", init, now])
    global sub_history_file,sub_init_file
    total_result = np.zeros(n)  # 各個体の累積評価値をゼロで初期化

    # 各ジョブ（個体）の専用初期ファイルを各PE用にコピー
    for i in range(n):
        for pe in range(nofpe):
            # 並列数が10未満は "000i"、10以上は "00i" のディレクトリ名を使用
            sub_init_file = f"000{i}/init_00000101-000000.000.pe######.nc"
            if i>=10:
                sub_init_file = f"00{i}/init_00000101-000000.000.pe######.nc"
            sub_init = sub_init_file.replace('######', str(pe).zfill(6))
            init = init_file.replace('######', str(pe).zfill(6))
            subprocess.run(["cp", init, sub_init])
    # シミュレーションを t ステップに渡って実行
    for step in range(t):
        # 各ジョブの history ファイルがあれば削除
        for i in range(n):
            for pe in range(nofpe):
                sub_history_file = f"000{i}/history.pe######.nc"
                if i>=10:
                    sub_init_file = f"00{i}/init_00000101-000000.000.pe######.nc"
                sub_history = sub_history_file.replace('######', str(pe).zfill(6))
                history_path = file_path+'/'+sub_history
                if (os.path.isfile(history_path)):
                    subprocess.run(["rm", sub_history])
            # 各ジョブに対して制御入力を適用
            # inputs[i][step]        : 連続変数（MOMYに加える値）
            # inputs[i][step+T]      : 整数（制御対象のy位置）
            # inputs[i][step+2*T]    : 整数（制御対象のz位置）
            control(i,inputs[i][step],inputs[i][step+T],inputs[i][step+2*T])

        # mpirun を用いて並列シミュレーションを実行
        result = subprocess.run(
            ["mpirun", "-n", "20", "scale-rm", "run.launch.conf"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # 結果の標準出力・標準エラーを表示
        print(result.stdout.decode())
        print(result.stderr.decode())
        # 各ジョブの当該ステップでの評価値（降水量）を計算
        step_result = np.zeros(n) 
        for i in range(n):
            for pe in range(nofpe):
                sub_history_file = f"000{i}/history.pe######.nc"
                if i>=10:
                    sub_history_file = f"00{i}/history.pe######.nc"
                fiy, fix = np.unravel_index(pe, (fny, fnx))
                nc = netCDF4.Dataset(sub_history_file.replace('######', str(pe).zfill(6)))
                nt = nc.dimensions['time'].size
                nx = nc.dimensions['x'].size
                ny = nc.dimensions['y'].size
                nz = nc.dimensions['z'].size
                gx1 = nx * fix
                gx2 = nx * (fix + 1)
                gy1 = ny * fiy
                gy2 = ny * (fiy + 1)
                # PE0の場合、全体のデータ配列を初期化
                if(pe==0):
                    dat = np.zeros((nt, nz, fny*ny, fnx*nx))
                # 対象領域にデータを配置
                dat[:, 0, gy1:gy2, gx1:gx2] = nc[varname][:]
                # 各グリッド（40セル）について降水量を累積
                for j in range(40):
                    step_result[i] += dat[1, 0, j, 0]*600
            total_result[i]+=step_result[i]
    for pe in range(nofpe):
        now = now_file.replace('######', str(pe).zfill(6))
        init = init_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", now, init])
    print(f"predict,loop={loop}")
    
    return total_result

def control(num,input1,input2,input3):
    """
    指定されたジョブ（num番目）の制御ファイルを更新する関数。
      input1: 連続変数（MOMY変数に加える値）
      input2: 整数（制御対象のy位置）
      input3: 整数（制御対象のz位置）
    """
    global org_file
    # 出力ファイルとサブ初期ファイルのパスを生成（個体番号が10以上の場合はディレクトリ名が変わる）
    for pe in range(nofpe):
        output_file = f"000{num}/out-MOMY.pe######.nc"
        sub_init_file = f"000{num}/init_00000101-000000.000.pe######.nc"
        if num>=10:
            output_file = f"00{num}/out-MOMY.pe######.nc"
            sub_init_file = f"00{num}/init_00000101-000000.000.pe######.nc"
        sub_init = sub_init_file.replace('######', str(pe).zfill(6))
        output = output_file.replace('######', str(pe).zfill(6))
        # netCDFファイルの読み込みと新規作成：MOMY変数に対して入力値を加算
        with netCDF4.Dataset(sub_init) as src, netCDF4.Dataset(output, "w") as dst:
            dst.setncatts(src.__dict__)
            # 各次元の作成
            for name, dimension in src.dimensions.items():
                dst.createDimension(
                    name, (len(dimension) if not dimension.isunlimited() else None))
            # 各変数をコピー（MOMY変数の場合は入力値を反映）
            for name, variable in src.variables.items():
                x = dst.createVariable(
                    name, variable.datatype, variable.dimensions)
                dst[name].setncatts(src[name].__dict__)
                if name == 'MOMY':
                    var = src[name][:]
                    if pe == 0:
                        if input2<20:
                            var[int(input2), 0, int(input3)] += input1  # (y,x,z)
                    elif pe==1:
                        if input2>=20:
                            var[int(input2)-20, 0, int(input3)] += input1  # (y,x,z)
                            
                    dst[name][:] = var
                else:
                    dst[name][:] = src[name][:]
        # 更新後の出力ファイルを元のサブ初期ファイルに上書きコピー
        subprocess.run(["cp", output, sub_init ])
    return

def update_control(input1,input2,input3):
    """
    PSO最適化後の最良制御入力を初期状態ファイル（init_file）に反映する関数。
      input1, input2, input3: それぞれ、連続変数、整数（y位置）、整数（z位置）のリスト
    """
    global org_file
    
    for pe in range(nofpe):
        output_file = f"out-MOMY.pe######.nc"
        init = init_file.replace('######', str(pe).zfill(6))
        output = output_file.replace('######', str(pe).zfill(6))
        with netCDF4.Dataset(init) as src, netCDF4.Dataset(output, "w") as dst:
            dst.setncatts(src.__dict__)
            for name, dimension in src.dimensions.items():
                dst.createDimension(
                    name, (len(dimension) if not dimension.isunlimited() else None))
            for name, variable in src.variables.items():
                x = dst.createVariable(
                    name, variable.datatype, variable.dimensions)
                dst[name].setncatts(src[name].__dict__)
                if name == 'MOMY':
                    var = src[name][:]
                    if pe == 0:
                        if input2[0]<20:
                            var[int(input2[0]), 0, int(input3[0])] += input1[0]  # (y,x,z)
                    elif pe==1:
                        if input2[0]>=20:
                            var[int(input2[0])-20, 0, int(input3[0])] += input1[0]  # (y,x,z)
                            
                    dst[name][:] = var
                else:
                    dst[name][:] = src[name][:]
        subprocess.run(["cp", output, init])
    return

def f(inputs):
    """
    評価関数 f
      1. 現在の初期状態（init_file）をバックアップ（temp_fileへコピー）
      2. predict() を呼び出し、各個体の累積評価値（降水量）を算出
      3. バックアップから init_file を復元
      4. 各個体の評価値（コスト）を返す
    """
    
    for pe in range(nofpe):
        temp = temp_file.replace('######', str(pe).zfill(6))
        init = init_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", init, temp])
    
    cost_sum = predict(inputs,T)
    
    
    for pe in range(nofpe):
        temp = temp_file.replace('######', str(pe).zfill(6))
        init = init_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", temp, init])
    return cost_sum #n個の配列　jobごとのコスト関数が入ってる

def state_update():
    """
    シミュレーション結果を更新する関数
      - 既存の history ファイルを削除
      - 新たにシミュレーション（mpirunで実行）して history ファイルを生成
      - 制御前後の降水量データを読み込み、各グリッドでの累積降水量を更新
      - 結果のグラフを作成して保存
      - ループカウンタ(loop)およびタイムステップ数(T)を更新
    """
    global loop,sum_gpy,sum_no,T
    orgfile = f"history-{loop}.pe######.nc"
    gpyoptfile=f"PSO-SHMPC-{loop}-MOMY-opt{opt_num}.pe######.nc"
    for pe in range(nofpe):
        history_file = "history.pe######.nc"
        history = history_file.replace('######', str(pe).zfill(6))
        history_path = file_path+'/'+history
        if (os.path.isfile(history_path)):
            subprocess.run(["rm", history])
    # mpirun によってシミュレーションを実行（設定ファイル run_R20kmDX500m-all-prec.conf を使用）
    subprocess.run(["mpirun", "-n", "2", "scale-rm", "run_R20kmDX500m-all-prec.conf"])
    
    # history ファイルを PSO 結果用ファイルへコピー
    for pe in range(nofpe):
        output = gpyoptfile.replace('######', str(pe).zfill(6))
        history = history_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", history,output])

    # 各PEごとに history ファイルからデータを読み込み、領域ごとに結合
    for pe in range(nofpe):
        fiy, fix = np.unravel_index(pe, (fny, fnx))
        nc = netCDF4.Dataset(history_file.replace('######', str(pe).zfill(6)))
        onc = netCDF4.Dataset(orgfile.replace('######', str(pe).zfill(6)))
        nt = nc.dimensions['time'].size
        nx = nc.dimensions['x'].size
        ny = nc.dimensions['y'].size
        nz = nc.dimensions['z'].size
        gx1 = nx * fix
        gx2 = nx * (fix + 1)
        gy1 = ny * fiy
        gy2 = ny * (fiy + 1)
        if(pe==0):
            dat = np.zeros((nt, nz, fny*ny, fnx*nx))
            odat = np.zeros((nt, nz, fny*ny, fnx*nx))
        dat[:, 0, gy1:gy2, gx1:gx2] = nc[varname][:]
        odat[:, 0, gy1:gy2, gx1:gx2] = onc[varname][:]
     # 各グリッドセルごとに累積降水量を更新
    for i in range(40):
        sum_gpy[i]+=dat[1, 0, i, 0]*600
        sum_no[i]+=odat[1, 0, i, 0]*600
    # 各タイムステップごとに、制御前後の降水量推移をプロットして画像として保存
    for i in range(nt):
            l1, l2 = 'no-control', 'under-control'
            c1, c2 = 'blue', 'green'
            xl = 'y'
            yl = 'PREC'
            plt.plot(dat[i, 0, :, 0], color=c2, label=l2)
            plt.plot(odat[i, 0, :, 0], color=c1, label=l1)
            plt.xlabel(xl)
            plt.ylabel(yl)
            plt.legend()
            dirname = f"PSO-MOMY-={opt_num}-input={input_size}/"
            os.makedirs(dirname, exist_ok=True)
            filename = dirname + \
                f'sabun-MOMY-t={loop}.png'
            plt.ylim(0, 0.025)
            plt.savefig(filename)
            plt.clf()
    # ループカウンタと予測区間を更新
    loop += 1
    T-=1
    return 

# 1次元データをカンマ区切りで出力する関数
def save_single_line_with_trailing_comma(save_file_path, data):
    """
    1次元データをカンマ区切りの1行テキストとして保存する関数
      ※ 末尾にもカンマを追加して出力
    """
    with open(save_file_path, 'w') as f:
        # data が単一の数値の場合はリストに変換
        if isinstance(data, (int, float, np.float64)):
            data = [data]  # リストに変換
        f.write(','.join(map(str, data)) + ',')  # 全ての値をカンマ区切りで1行にし、末尾にもカンマを追加

# ---------------------------
# メイン処理：PSOによる最適化
# ---------------------------

start = time.time()

input_size=30

# ---------------------------
# PSO パラメータの設定
# ---------------------------
maxiter = opt_num  # 最大反復回数
w_max = 0.9        # 初期慣性重み
w_min = 0.4        # 最終慣性重み
c1 = 2.0           # 認知パラメータ（個人最良解への寄与）
c2 = 2.0           # 社会パラメータ（群全体の最良解への寄与）

# 連続変数・整数変数の分割インデックス
split_index1 = T             # 連続変数部分の長さ
split_index2 = T * 2         # 連続変数＋1つ目の整数変数部分の長さ
num_variables = T            # 連続変数の数

# 連続変数と整数変数の下限・上限設定
lb1 = [-30] * T
ub1 = [30] * T
lb2 = [0] * T
ub2 = [39] * T
lb3 = [0] * T
ub3 = [96] * T


# 結果の収束履歴と入力履歴を保存するディレクトリ作成
convergence=f"convergence-PSO-MOMY-opt={opt_num}"
os.makedirs(convergence, exist_ok=True)

learn_process=f"input-PSO-learning-opt={opt_num}"
os.makedirs(learn_process, exist_ok=True)

# 各PEの初期状態を org_file から init_file にコピーして初期化
for pe in range(nofpe):
    org = org_file.replace('######', str(pe).zfill(6))
    init = init_file.replace('######', str(pe).zfill(6))
    subprocess.run(["cp", org, init])

batch_size = n
# 6回のPSOループを実施
for t in range(6):
    start_time=time.time()
    # ループごとに変数の下限・上限や分割インデックスを更新 Shrinking Horizon MPCであるため
    num_variables=T
    split_index1 = T
    split_index2 = T*2
    lb1=[-30]*T
    ub1=[30]*T
    lb2=[0]*T
    ub2=[39]*T
    lb3=[0]*T
    ub3=[96]*T
    best_values = []         # 各反復での最良評価値記録
    function_evals = []      # 関数評価回数記録
    total_function_evals = 0 # 総評価回数
    current_best = float('inf')
    # 各反復での主要制御入力（input1, input2, input3の先頭要素）を記録する配列
    input_learn_process=np.zeros((maxiter*swarmsize,3))
    # 各PEの初期状態をバックアップ（temp2_fileへコピー）
    for pe in range(nofpe):
        temp2 = temp2_file.replace('######', str(pe).zfill(6))
        init = init_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", init, temp2])
    # ---------------------------
    # 粒子群の初期化
    # ---------------------------
    rng_randint1 = np.random.default_rng(random_num+init_cnt)  # randint1用
    rng_randint2 = np.random.default_rng(random_num+10+init_cnt)  # randint2用
    rng_uniform = np.random.default_rng(random_num+20+init_cnt)
    # ランダムサンプルを生成
    random_samples1 = rng_uniform.uniform(-30, 30, (swarmsize, num_variables)) #MOMY
    random_samples2 = rng_randint1.integers(0, 39, (swarmsize, num_variables)) #y座標
    random_samples3 = rng_randint2.integers(0, 96, (swarmsize, num_variables)) #z座標

    

    # 粒子の初期化
    combined_samples = np.concatenate((random_samples1, random_samples2,random_samples3), axis=1)
    positions = combined_samples
    rng_uniform_v = np.random.default_rng(random_num+30+init_cnt)
    

    # 初期速度を設定
    v_ratio = 0.2
    # 速度を生成
    velocities1 = rng_uniform_v.uniform(-v_ratio * (np.array(ub1) - np.array(lb1)), v_ratio * (np.array(ub1) - np.array(lb1)), (swarmsize, T))#MOMY
    velocities2 = rng_uniform_v.uniform(-v_ratio * (np.array(ub2) - np.array(lb2)), v_ratio * (np.array(ub2) - np.array(lb2)), (swarmsize, T))#y座標
    velocities3 = rng_uniform_v.uniform(-v_ratio * (np.array(ub3) - np.array(lb3)), v_ratio * (np.array(ub3) - np.array(lb3)), (swarmsize, T))#z座標

    # 全体を結合
    velocities = np.hstack([velocities1, velocities2, velocities3])

    # 個人最良位置および全体最良位置の初期化
    personal_best_positions = positions.copy()
    personal_best_values = np.array(f(positions))
    global_best_position = personal_best_positions[np.argmin(personal_best_values)]
    global_best_value = min(personal_best_values)
    
    # 探索過程の記録
    history = [positions.copy()]
    global_best_history = [global_best_value]
    r1_uniform = np.random.default_rng(random_num+40+init_cnt)
    init_cnt+=1

    # ---------------------------
    # PSO のメイン反復処理
    # ---------------------------
    for k in range(maxiter):
            for j in range(swarmsize):
                # ランダム係数 r1, r2 を生成
                r1, r2 = r1_uniform.uniform(0.0,1.0), r1_uniform.uniform(0.0,1.0)
                # 慣性重みの線形減衰
                w = w_max - (w_max - w_min) * (k / maxiter)
                # 速度更新：慣性項 + 個人最良解への引力 + 群全体の最良解への引力
                velocities[j] = (w * velocities[j] +
                                c1 * r1 * (personal_best_positions[j] - positions[j]) +
                                c2 * r2 * (global_best_position - positions[j]))
                # 粒子の位置更新
                positions[j] = positions[j] + velocities[j]
                # 制約を満たすように位置をクリップ
                
                positions[j][0:split_index1] = np.clip(positions[j][0:split_index1], lb1, ub1)
                positions[j][split_index1:split_index2] = np.clip(positions[j][split_index1:split_index2], lb2, ub2)
                positions[j][split_index2:] = np.clip(positions[j][split_index2:], lb3, ub3)
                # 整数部分（input2, input3）の丸め処理
                positions[j][split_index1:split_index2] = np.rint(positions[j][split_index1:split_index2]).astype(int)
                positions[j][split_index2:] = np.rint(positions[j][split_index2:]).astype(int)
            # 現在の粒子群の評価（各個体について f 関数を評価）
            value = f(positions)
            for j in range(swarmsize):
                # 主要な制御入力（input1, input2, input3の先頭要素）を記録
                input_learn_process[k*swarmsize+j,0]=positions[j][0]
                input_learn_process[k*swarmsize+j,1]=positions[j][T]
                input_learn_process[k*swarmsize+j,2]=positions[j][2*T]
                # 個人最良位置の更新
                if value[j] < personal_best_values[j]:
                    personal_best_positions[j] = positions[j]
                    personal_best_values[j] = value[j]
                # 全体最良の更新
                if value[j] < global_best_value:
                    global_best_position = positions[j]
                    global_best_value = value[j]
            # 各反復後、現在の粒子位置を記録
            history.append(positions.copy())
            total_function_evals += swarmsize
            function_evals.append(total_function_evals)
            best_values.append(global_best_value)
    global_best_history.append(global_best_value)
    end_time=time.time()
    print(f'Optimal input: {global_best_position}')
    print(f'Objective function value at minimum: {global_best_value}')
    
    
    # ---------------------------
    # 収束グラフの作成・保存
    # ---------------------------
    plt.figure() 
    plt.plot(function_evals, best_values, marker='o', linestyle='-', color='b')
    plt.title('Convergence of PSO')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Best Objective Value Found')
    plt.grid(True)
    plt.savefig(f"{convergence}PSO-convergence-loop={t}.png")
    save_file_path = f"{convergence}/best_history-loop{t}.txt"
    save_single_line_with_trailing_comma(save_file_path, best_values)

    plt.show()
    # ---------------------------
    # 最良個体の遺伝子を3群に分割（input1, input2, input3）
    # ---------------------------
    arr1 = global_best_position[:split_index1]
    arr2 = global_best_position[split_index1:split_index2]
    arr3 = global_best_position[split_index2:]


    x_iters=np.arange(1, 10*opt_num+1)
    # ---------------------------
    # 制御入力履歴のプロットと保存（y: input2, z: input3, x: input1）
    # ---------------------------
    plt.figure(figsize=(10, 5))
    l1='$y_t$'
    plt.plot(x_iters, input_learn_process[:,1], marker='o', linestyle='-', color='b',label=l1)
    plt.title('y')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Control input')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{learn_process}/y-loop={t}.png")
    save_file_path = f"{learn_process}/input_learn_y-loop{t}.txt"
    save_single_line_with_trailing_comma(save_file_path, input_learn_process[:,1])
    plt.show()

    plt.figure(figsize=(10, 5))
    l2='$z_t$'
    plt.plot(x_iters, input_learn_process[:,2], marker='o', linestyle='-', color='r',label=l2)
    plt.title('z')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Control input')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{learn_process}/z-loop={t}.png")
    save_file_path = f"{learn_process}/input_learn_z-loop{t}.txt"
    save_single_line_with_trailing_comma(save_file_path, input_learn_process[:,2])
    plt.show()

    plt.figure(figsize=(10, 5))
    l3='$MOMY$'
    plt.plot(x_iters, input_learn_process[:,0], marker='o', linestyle='-', color='g',label=l3)
    plt.title('x')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Control input')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{learn_process}/MOMY-loop={t}.png")
    save_file_path = f"{learn_process}/input_learn_MOMY-loop{t}.txt"
    save_single_line_with_trailing_comma(save_file_path, input_learn_process[:,0])
    plt.show()


    # バックアップから初期状態を復元
    for pe in range(nofpe):
        temp2 = temp2_file.replace('######', str(pe).zfill(6))
        init = init_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", temp2, init])
    # 最良の制御入力を init_file に反映
    update_control(arr1,arr2,arr3)
    input_history1[t]=arr1[0]
    input_history2[t]=arr2[0]
    input_history3[t]=arr3[0]
    time_history[t]=end_time-start_time
    print(f"input_history1={input_history1},input_history2={input_history2},input_history3={input_history3}")
    # シミュレーション結果の状態更新（history ファイル取得とグラフ作成）
    state_update()
    print(f"loop={loop}")

end = time.time()
time_diff = end - start
print(f'実行時間{time_diff}')
sum_time=0
for i in range(6):
    sum_time+=time_history[i]

print(f"input_history1 {input_history1}")
print(f"input_history2 {input_history2}")
print(f"input_history3 {input_history3}")
print(f"time_history{time_history}")
print(f"sum_time{sum_time}")
print(f"sum_gpy={sum_gpy}")

# 制御前後の累積降水量の総和を計算
no=0
gpy=0
for i in range(40):
    no+=sum_no[i]
    gpy+=sum_gpy[i]

print(f"PSO={gpy}")
print(f"change%={(no-gpy)/no*100}%")
print(f"%={(gpy)/no*100}%")