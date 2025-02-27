import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess

import time
import scipy.optimize as opt
from skopt.space import Real,Integer
import warnings
from skopt import Optimizer
import random

# 乱数シードの設定（再現性のため）
random_num=1 #必要に応じて乱数シードを変えてください

# 警告を抑制
warnings.filterwarnings('ignore', category=DeprecationWarning)

# matplotlib のバックエンドを設定（GUIがなくても描画可能にする）
matplotlib.use('Agg')

# シミュレーションに関するグローバルパラメータの定義
nofpe = 2       # 並列実行するPE（プロセス）の数
fny = 2         # 格子の分割数（y方向）
fnx = 1         # 格子の分割数（x方向）
loop = 0        # ループ回数の初期値

# 変数名（降水量の変数と想定）
varname = 'PREC'

# 各種ファイル名のテンプレート（pe番号などは後で置換）
init_file = "init_00000101-000000.000.pe######.nc"
sub_init_file = "0000/init_00000101-000000.000.pe######.nc"
org_file = "init_00000101-000000.000.pe######.org.nc"
history_file = "history.pe######.nc"
sub_history_file = "0000/history.pe######.nc"
restart_file = "restart_00000101-010000.000.pe000000.nc"
orgfile = f"history-{loop}.pe######.nc"
now_file = f"now.pe######.nc"
temp_file=f"temp.pe######.nc"
temp2_file=f"temp2.pe######.nc"
# シミュレーション実行ディレクトリのパス
file_path = '/home/rk-nagai/scale-5.5.1/scale-rm/test/tutorial/ideal' #自身のディレクトリに書き換えてください
# ベイズ最適化結果の出力ファイル名（ループ番号含む）
gpyoptfile=f"gpyopt-{loop}.pe######.nc" #BOの結果を保存するファイル　名前は適当なので必要に応じて書き換えてください

# 入力履歴を保存するための配列（各ループごとに保存）
input_history1=np.zeros((6,1))
input_history2=np.zeros((6,1))
input_history3=np.zeros((6,1))
time_history=np.zeros((6,1))

# ベイズ最適化で得られる累積降水量を保存する配列（制御ありとなし）
sum_gpy = np.zeros(40) # 制御後の累積降水量
sum_no = np.zeros(40)  # 制御前の累積降水量

#モデル予測制御の予測区間の長さ
T = 6
# ベイズ最適化の反復回数（最適化試行回数）必要に応じて変更してください
opt_num=50
#バルクジョブの並列数　並列数10で実行する場合はcore数20以上のPCで実行してください
n=10

#MOMYの変更量の上限の大きさ
input_size=30
# 最適化対象のパラメータの範囲を設定
# ここでは、連続変数T個＋整数変数2*T個のパラメータを扱う
bounds = [Real(-input_size, input_size) for _ in range(T)]
bounds += [Integer(0, 39) for _ in range(T)]   # y方向の制御位置（0～39）
bounds += [Integer(0, 96) for _ in range(T)]   # z方向の制御位置（0～96）


def predict(inputs,t):
    """
    シミュレーションを実行し、各ジョブ（n個）の評価値（累積降水量）を計算する関数
    inputs: 各ジョブの入力パラメータ（各ジョブ T個の連続変数＋整数変数2*T個）
    t: 予測区間の長さ
    """
    # 各PEごとに現在の初期ファイル(now_file)を保存しておく（バックアップ）
    for pe in range(nofpe):
            now = now_file.replace('######', str(pe).zfill(6))
            init = init_file.replace('######', str(pe).zfill(6))
            subprocess.run(["cp", init, now])#initを保存しておく
    global sub_history_file,sub_init_file
    total_result = [0]*n  # 各ジョブの累積結果を初期化

    # 各ジョブのシミュレーション用初期ファイルを設定（各ジョブのディレクトリ "000i" にコピー）
    for i in range(n):
        for pe in range(nofpe):
            sub_init_file = f"000{i}/init_00000101-000000.000.pe######.nc"
            sub_init = sub_init_file.replace('######', str(pe).zfill(6))
            init = init_file.replace('######', str(pe).zfill(6))
            subprocess.run(["cp", init, sub_init])

    # 時間ステップ t にわたってシミュレーションを実行
    for step in range(t):
        # 各ジョブごとに履歴ファイル(history)があれば削除する
        for i in range(n):
            for pe in range(nofpe):
                sub_history_file = f"000{i}/history.pe######.nc"
                sub_history = sub_history_file.replace('######', str(pe).zfill(6))
                history_path = file_path+'/'+sub_history
                if (os.path.isfile(history_path)):
                    subprocess.run(["rm", sub_history])
            #各ジョブごとに制御入力を適用する       
            control(i,inputs[i][step],inputs[i][step+T],inputs[i][step+2*T])

        # mpirun を用いて並列シミュレーションを実行
        result = subprocess.run(
            ["mpirun", "--oversubscribe" ,"-n", "20", "scale-rm", "run.launch.conf"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
        print(result.stderr.decode())
        
        # 各ジョブのシミュレーション結果から評価値（降水量）を算出
        step_result = [0]*n # 各ジョブのステップ毎の降水量を初期化
        for i in range(n):
            for pe in range(nofpe):
                sub_history_file = f"000{i}/history.pe######.nc"
                fiy, fix = np.unravel_index(pe, (fny, fnx))
                # netCDF ファイルを読み込み
                nc = netCDF4.Dataset(sub_history_file.replace('######', str(pe).zfill(6)))
                nt = nc.dimensions['time'].size
                nx = nc.dimensions['x'].size
                ny = nc.dimensions['y'].size
                nz = nc.dimensions['z'].size
                gx1 = nx * fix
                gx2 = nx * (fix + 1)
                gy1 = ny * fiy
                gy2 = ny * (fiy + 1)
                #データ配列を初期化
                if(pe==0):
                    dat = np.zeros((nt, nz, fny*ny, fnx*nx))
                #降水量を配列に格納する
                dat[:, 0, gy1:gy2, gx1:gx2] = nc[varname][:]
                #全y座標の降水量を加算する(configファイルで10分あたりの平均PRECを出力するように設定しているので値を600倍する)
                for j in range(40):
                    step_result[i] += dat[1, 0, j, 0]*600
            total_result[i]+=step_result[i]
    # シミュレーション終了後、各PEごとに初期状態をバックアップから復元
    for pe in range(nofpe):
        now = now_file.replace('######', str(pe).zfill(6))
        init = init_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", now, init])
    print(f"predict,loop={loop}")
    return total_result # 各ジョブの累積降水量を返す


def control(num,input1,input2,input3):
    """
    各ジョブ (num) の制御ファイルを更新し、MOMY 変数に対して入力値を適用する関数
    input1: MOMYに加える値（連続変数）
    input2: 制御対象のy位置（整数）
    input3: 制御対象のz位置（整数）
    """
    global org_file
    for pe in range(nofpe):
        output_file = f"000{num}/out-MOMY.pe######.nc"
        sub_init_file = f"000{num}/init_00000101-000000.000.pe######.nc"
        sub_init = sub_init_file.replace('######', str(pe).zfill(6))
        output = output_file.replace('######', str(pe).zfill(6))
        # netCDF ファイルの読み込みと新規作成（コピー）を行う
        with netCDF4.Dataset(sub_init) as src, netCDF4.Dataset(output, "w") as dst:
            # グローバル属性のコピー
            dst.setncatts(src.__dict__)
            for name, dimension in src.dimensions.items():
                dst.createDimension(
                    name, (len(dimension) if not dimension.isunlimited() else None))
            # 各変数のコピーと、MOMY変数に対して入力値を加算
            for name, variable in src.variables.items():
                x = dst.createVariable(
                    name, variable.datatype, variable.dimensions)
                dst[name].setncatts(src[name].__dict__)
                if name == 'MOMY':
                    var = src[name][:]
                    # PE番号に応じて制御対象のインデックスを調整
                    if pe == 0:
                        if input2<20:
                            var[int(input2), 0, int(input3)] += input1  # (y,x,z)
                    elif pe==1:
                        if input2>=20:
                            var[int(input2)-20, 0, int(input3)] += input1  # (y,x,z)
                            
                    dst[name][:] = var
                else:
                    dst[name][:] = src[name][:]
        # 出力ファイルを初期ファイルに上書きコピーする
        subprocess.run(["cp", output, sub_init ])
    return

def update_control(input1,input2,input3):
    """
    初期ファイルに対して最適化後の制御入力を反映させるための関数
    ここでは、シングルジョブ（PEごと）のファイルを更新する
    input1, input2, input3 はリスト形式（各PEに対して適用）
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
                    # PEごとに条件分岐し、最適化入力を MOMY に加算
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
    1. 初期状態を temp に保存
    2. predict 関数で各ジョブのシミュレーションを実行し、評価値（降水量）を取得
    3. temp から初期状態を復元
    4. 評価値を返す
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

    return cost_sum  # 各ジョブのコスト（累積降水量）のリストを返す

def state_update():
    """
    状態の更新を行う関数
    ・制御前後の history ファイルの整理
    ・別の設定ファイルを用いてシミュレーションを実行（制御前後の比較）
    ・降水量の累積値の更新とプロットの生成
    ・ループカウンタ、T（タイムステップ数）および最適化パラメータの範囲(bounds)の更新
    """
    global loop,sum_gpy,sum_no,T,bounds
    orgfile = f"history-{loop}.pe######.nc"
    gpyoptfile=f"BO-SHMPC-{loop}-MOMY-opt{opt_num}.pe######.nc"
    # 各PEの history ファイルがあれば削除する
    for pe in range(nofpe):
        history_file = "history.pe######.nc"
        history = history_file.replace('######', str(pe).zfill(6))
        history_path = file_path+'/'+history
        if (os.path.isfile(history_path)):
            subprocess.run(["rm", history])
    # 制御前後のシミュレーションを実行する設定ファイルを用いて実行
    subprocess.run(["mpirun", "-n", "2", "scale-rm", "run_R20kmDX500m-all-prec.conf"])
    
    # history ファイルを BO の結果ファイルへコピー
    for pe in range(nofpe):
        output = gpyoptfile.replace('######', str(pe).zfill(6))
        history = history_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", history,output])

    # 各PEごとに制御前後の降水量データを読み込み，領域ごとに結合する
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
    # 各領域ごとに累積降水量を更新（600をかけるのは10分間の降水量を表している）
    for i in range(40):
        sum_gpy[i]+=dat[1, 0, i, 0]*600
        sum_no[i]+=odat[1, 0, i, 0]*600
    # 各時間ステップごとに制御前後の降水量の推移をプロットし，画像ファイルとして保存
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
            dirname = f"BO-MOMY-={opt_num}-input={input_size}/"
            os.makedirs(dirname, exist_ok=True)
            filename = dirname + \
                f'sabun-MOMY-t={loop}.png'
            plt.ylim(0, 0.025)
            plt.savefig(filename)
            plt.clf()
    # 更新後、ループカウンタをインクリメントし，タイムステップ数 T をデクリメント(Shrinking Horizon MPCであるため)
    loop += 1
    T-=1
    # 新しい予測区間 T に合わせてパラメータの範囲(bounds)を再設定
    bounds = [Real(-input_size, input_size) for _ in range(T)]
    bounds += [Integer(0, 39) for _ in range(T)]   # y方向
    bounds += [Integer(0, 96) for _ in range(T)]   # z方向


    return 
# メイン処理開始時刻の記録
start = time.time()
# 初期状態を org_file から init_file にコピーして初期化
for pe in range(nofpe):
    org = org_file.replace('######', str(pe).zfill(6))
    init = init_file.replace('######', str(pe).zfill(6))
    subprocess.run(["cp", org, init]) #initファイルの初期化



# 初期データの準備（初期サンプル数）
n_initial_points = 10 #batch_sizeに合わせる
random.seed(random_num)
# 学習過程を保存するディレクトリの作成
learn_process=f"input-BO-learning-opt={opt_num}-random-{random_num}"
os.makedirs(learn_process, exist_ok=True)

batch_size = n # バッチサイズはジョブ数と同じ
# ループ回数（ここでは6回の最適化ループを実行）
for i in range(6):
    # 各PEの初期状態を一時保存（temp2ファイルにコピー）
    for pe in range(nofpe):
        temp2 = temp2_file.replace('######', str(pe).zfill(6))
        init = init_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", init, temp2])
    # 各最適化ループでの入力値履歴を保存するためのリストを初期化
    input_learn_process1 = [0] * opt_num
    input_learn_process2 = [0] * opt_num
    input_learn_process3 = [0] * opt_num
    
    start_time = time.time()
    # 各パラメータの初期サンプルをランダムに生成
    random_samples1 = [[random.uniform(-30, 30) for _ in range(T)] for _ in range(n_initial_points)]
    random_samples2 = [[random.randint(0, 39) for _ in range(T)] for _ in range(n_initial_points)]
    random_samples3 = [[random.randint(0, 96) for _ in range(T)] for _ in range(n_initial_points)]

    # リストを結合してサンプルを作成
    combined_samples = [sample1 + sample2 + sample3 for sample1, sample2, sample3 in zip(random_samples1, random_samples2, random_samples3)]
    # ベイズ最適化のための Optimizer を初期化（ガウス過程を使用）
    opt = Optimizer(bounds, base_estimator="GP", acq_func="EI", random_state=random_num) 

    # 初期サンプルで評価関数を実行し，結果を登録
    X = combined_samples
    Y = f(X)
    opt.tell(X, Y)
    
    best_values = []
    current_best = float('inf')

    # 初期サンプルの入力値のうち，最初のパラメータを記録
    for t in range(10):
        
        input_learn_process1[t]=combined_samples[t][0]
        input_learn_process2[t]=combined_samples[t][T]
        input_learn_process3[t]=combined_samples[t][2*T]
    
    # 以降、バッチごとの最適化を実行
    for j in range(int(opt_num/n)-1):
        # 次の探索点をベイズ最適化のacquisition関数を用いて取得（バッチサイズ分）
        next_points = opt.ask(n_points=batch_size)

        # 並列で評価関数を計算
        values = f(next_points)
        for t in range(10):
             input_learn_process1[10+j*10+t]=next_points[t][0]
             input_learn_process2[10+j*10+t]=next_points[t][T]
             input_learn_process3[10+j*10+t]=next_points[t][2*T]
        
        
        # 評価結果をモデルに反映
        opt.tell(next_points, values)
        print(f"Batch {j+1}: Best value so far: {min(opt.yi)}")

    # 最適化結果から最小値とそのときのパラメータを抽出
    best_value = min(opt.yi)
    min_index = opt.yi.index(min(opt.yi))
    best_point = opt.Xi[min_index]
    print(f"Best value: {best_value} at point {best_point}")

    end_time = time.time()
    optimal_inputs=best_point
    Y_array = np.array(Y)
    
    # 最適化結果を3つのパラメータ群に分割
    split_index1 = T
    split_index2 = T*2
    arr1 = optimal_inputs[:split_index1]
    arr2 = optimal_inputs[split_index1:split_index2]
    arr3 = optimal_inputs[split_index2:]

    # 評価関数の呼び出し回数に対応する x 軸の値を生成
    x_iters=np.arange(1, opt_num+1)

    # 各パラメータの履歴をプロットして保存（y, z, MOMY）
    plt.figure(figsize=(10, 5))
    l1='$y_t$'
    plt.plot(x_iters, input_learn_process2, marker='o', linestyle='-', color='b',label=l1)
    plt.title('y')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Control input')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{learn_process}/y-loop={i}.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    l2='$z_t$'
    plt.plot(x_iters, input_learn_process3, marker='o', linestyle='-', color='r',label=l2)
    plt.title('z')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Control input')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{learn_process}/z-loop={i}.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    l3='$MOMY$'
    plt.plot(x_iters, input_learn_process1, marker='o', linestyle='-', color='g',label=l3)
    plt.title('x')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Control input')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{learn_process}/MOMY-loop={i}.png")
    plt.show()


    # 各PEの初期状態を一時保存したもの（temp2）から復元
    for pe in range(nofpe):
        temp2 = temp2_file.replace('######', str(pe).zfill(6))
        init = init_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", temp2, init])
    # 最適化で得た入力値を反映して初期状態を更新
    update_control(arr1,arr2,arr3)
    # 各ループでの最適入力と処理時間を記録
    input_history1[i]=arr1[0]
    input_history2[i]=arr2[0]
    input_history3[i]=arr3[0]
    time_history[i]=end_time-start_time
    # 状態更新（シミュレーション結果のプロット生成、累積降水量の更新、ループカウンタや T の更新）
    state_update()
    print(f"loop={loop}")

# メイン処理終了時刻を取得
end = time.time()
time_diff = end - start
print(f'実行時間{time_diff}')

# 各ループの処理時間の総和を計算
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

print(f"BO={gpy}")
print(f"change%={(no-gpy)/no*100}%")
print(f"%={(gpy)/no*100}%")