import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import subprocess
import time
import warnings
# 乱数シードの設定（再現性のため）
random_num=0 #必要に応じて乱数シードを変えてください


# 警告を抑制
warnings.filterwarnings('ignore', category=DeprecationWarning)
# matplotlib のバックエンドを 'Agg' に設定（GUI を使用せず画像保存可能）
matplotlib.use('Agg')
# ---------------------------
# シミュレーションおよびファイル設定
# ---------------------------

nofpe = 2                 # 使用する PE (Processing Element) の数
fny = 2                   # y 方向の領域分割数
fnx = 1                   # x 方向の領域分割数
loop = 0                  # ループ回数の初期値
varname = 'PREC'          # 対象変数（降水量と想定）


# ファイル名のテンプレート（"######" を実際の PE 番号で置換する）
init_file = "init_00000101-000000.000.pe######.nc"
sub_init_file = "0000/init_00000101-000000.000.pe######.nc"
org_file = "init_00000101-000000.000.pe######.org.nc"
history_file = "history.pe######.nc"
sub_history_file = "0000/history.pe######.nc"
restart_file = "restart_00000101-010000.000.pe000000.nc"
# その他のファイル名設定
orgfile = f"history-{loop}.pe######.nc"
now_file = f"now.pe######.nc"
temp_file=f"temp.pe######.nc" # 一時的に初期状態を保存するためのファイル
temp2_file=f"temp2.pe######.nc" # さらに別のバックアップ用ファイル
file_path = '/home/rk-nagai/scale-5.5.1/scale-rm/test/tutorial/ideal' #自身のディレクトリに書き換えてください

gpyoptfile=f"gpyopt-{loop}.pe######.nc" #RSの結果を保存するファイル　名前は適当なので必要に応じて書き換えてください

# ---------------------------
# 結果保存用の配列
# ---------------------------
input_history1=np.zeros((6,1))
input_history2=np.zeros((6,1))
input_history3=np.zeros((6,1))
time_history=np.zeros((6,1))

sum_gpy = np.zeros(40)  # PSO 後の累積降水量
sum_no = np.zeros(40)   # 制御前の累積降水量

n = 10      # ジョブ数（並列数）10並列で実行する場合はcore数20以上のPCで実行してください
T = 6       # 予測区間の長さ confファイルにて1ステップ10分で設定しているので、60分先の状態まで予測します
opt_num =100  # 最適化（RS：Random Search）の評価回数 必要に応じて変えてください



def predict(inputs,t):
    """
    各個体（ジョブ）の入力パラメータに基づいてシミュレーションを実行し、
    各ジョブの累積降水量（評価値）を算出する。
      inputs: 各個体の入力パラメータ（3*T 次元：連続変数＋2 種類の整数変数）
      t: 予測区間の長さ
    """
    # 各 PE ごとに、現在の初期状態をバックアップ (init_file → now_file)
    for pe in range(nofpe):
            now = now_file.replace('######', str(pe).zfill(6))
            init = init_file.replace('######', str(pe).zfill(6))
            subprocess.run(["cp", init, now])#initを保存しておく
    global sub_history_file,sub_init_file
    total_result = np.zeros(n)  # 各ジョブの累積評価値を 0 で初期化

    # 各ジョブの専用初期ファイルを個別ディレクトリにコピー
    for i in range(n):
        for pe in range(nofpe):
            sub_init_file = f"000{i}/init_00000101-000000.000.pe######.nc"
            if i>=10:
                sub_init_file = f"00{i}/init_00000101-000000.000.pe######.nc"
            sub_init = sub_init_file.replace('######', str(pe).zfill(6))
            init = init_file.replace('######', str(pe).zfill(6))
            subprocess.run(["cp", init, sub_init])
    # t ステップに渡ってシミュレーション実行
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
                    
            # 各ジョブに対して制御入力を反映
            # inputs[i][step]     : 連続変数（MOMY に加える値）
            # inputs[i][step+T]   : 整数（制御対象の y 位置）
            # inputs[i][step+2*T] : 整数（制御対象の z 位置）
            control(i,inputs[i][step],inputs[i][step+T],inputs[i][step+2*T])

        # mpirun を用いて並列シミュレーションを実行
        result = subprocess.run(
            ["mpirun", "-n", "20", "scale-rm", "run.launch.conf"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
        print(result.stderr.decode())
        # 各ジョブの当該ステップでの評価値（降水量）を算出
        step_result = np.zeros(n) # ステップごとの結果を初期化
        for i in range(n):
            for pe in range(nofpe):
                sub_history_file = f"000{i}/history.pe######.nc"
                if i>=10:
                    sub_history_file = f"00{i}/history.pe######.nc"
                # 各 PE の領域割当（fny, fnx による）を計算
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
                # PE0 では全体のデータ配列を初期化
                if(pe==0):
                    dat = np.zeros((nt, nz, fny*ny, fnx*nx))
                # 対象領域にデータを配置
                dat[:, 0, gy1:gy2, gx1:gx2] = nc[varname][:]
                # 40 グリッドセルについて600秒間に降る降水量を累積
                for j in range(40):
                    step_result[i] += dat[1, 0, j, 0]*600
            total_result[i]+=step_result[i]

    # シミュレーション終了後、バックアップから初期状態を復元   
    for pe in range(nofpe):
        now = now_file.replace('######', str(pe).zfill(6))
        init = init_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", now, init])
    print(f"predict,loop={loop}")
    
    return total_result

def control(num,input1,input2,input3):
    """
    指定されたジョブ (num 番目) の制御ファイルを更新する関数
      input1: 連続変数（MOMY に加える値）
      input2: 整数（制御対象の y 位置）
      input3: 整数（制御対象の z 位置）
    """
    global org_file
    print(f"control input1={input1},input2={input2},input3={input3}")
    for pe in range(nofpe):
        # 並列数によりディレクトリ名が変化
        output_file = f"000{num}/out-MOMY.pe######.nc"
        sub_init_file = f"000{num}/init_00000101-000000.000.pe######.nc"
        if num>=10:
            output_file = f"00{num}/out-MOMY.pe######.nc"
            sub_init_file = f"00{num}/init_00000101-000000.000.pe######.nc"
        sub_init = sub_init_file.replace('######', str(pe).zfill(6))
        output = output_file.replace('######', str(pe).zfill(6))
        # netCDF ファイルを読み込み、新規作成し、MOMY 変数に入力値を加算
        with netCDF4.Dataset(sub_init) as src, netCDF4.Dataset(output, "w") as dst:
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
                    # PE 番号により制御適用位置を変更
                    if pe == 0:
                        if input2<20:
                            var[int(input2), 0, int(input3)] += input1  # (y,x,z)
                    elif pe==1:
                        if input2>=20:
                            var[int(input2)-20, 0, int(input3)] += input1  # (y,x,z)
                            
                    dst[name][:] = var
                else:
                    dst[name][:] = src[name][:]
        subprocess.run(["cp", output, sub_init ])
    return

def update_control(input1,input2,input3):
    """
    PSO 最適化後の最良制御入力 (input1, input2, input3) を
    初期状態ファイル (init_file) に反映する関数
      input1: 連続変数のリスト (MOMY)
      input2: 整数のリスト (y 位置)
      input3: 整数のリスト (z 位置)
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
      1. 現在の初期状態をバックアップ (init_file → temp_file)
      2. predict() を呼び出し、各個体の累積評価値（コスト）を計算
      3. バックアップから初期状態を復元
      4. 各個体の評価値 (コスト) を返す
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
    シミュレーション結果の状態を更新する関数
      - 既存の history ファイルを削除し、シミュレーションを実行して最新の history を取得
      - 制御前後の降水量データを各グリッドごとに読み込み、累積降水量を更新
      - 制御前後の結果をプロットし画像ファイルとして保存
      - ループカウンタとタイムステップ数を更新
    """
    global loop,sum_gpy,sum_no,T
    orgfile = f"history-{loop}.pe######.nc"
    gpyoptfile=f"RS-SHMPC-{loop}-MOMY-opt{opt_num}.pe######.nc"
    for pe in range(nofpe):
        history_file = "history.pe######.nc"
        history = history_file.replace('######', str(pe).zfill(6))
        history_path = file_path+'/'+history
        if (os.path.isfile(history_path)):
            subprocess.run(["rm", history])
    subprocess.run(["mpirun", "-n", "2", "scale-rm", "run_R20kmDX500m-all-prec.conf"])
    

    for pe in range(nofpe):
        output = gpyoptfile.replace('######', str(pe).zfill(6))
        history = history_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", history,output])

    # 各 PE ごとに history ファイルからデータを読み込み、領域ごとに結合
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
    # 各yグリッドセルごとに降水量を累積
    for i in range(40):
        sum_gpy[i]+=dat[1, 0, i, 0]*600
        sum_no[i]+=odat[1, 0, i, 0]*600
    # 各タイムステップごとに、制御前後の降水量推移をプロットし画像保存
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
            dirname = f"RS-MOMY-={opt_num}-input={input_size}/"
            os.makedirs(dirname, exist_ok=True)
            filename = dirname + \
                f'sabun-MOMY-t={loop}.png'
            plt.ylim(0, 0.025)
            plt.savefig(filename)
            plt.clf()

    loop += 1
    T-=1
    return 

# 1次元データをカンマ区切りで出力する関数
def save_single_line_with_trailing_comma(save_file_path, data):
    """
    1次元データをカンマ区切りの1行テキストとして保存する関数
      ※ 末尾にもカンマを追加して出力する
    """
    with open(save_file_path, 'w') as f:
        # data が単一の数値の場合はリストに変換
        if isinstance(data, (int, float, np.float64)):
            data = [data]  # リストに変換
        f.write(','.join(map(str, data)) + ',')  # 全ての値をカンマ区切りで1行にし、末尾にもカンマを追加

# ---------------------------
# メイン処理：RS（ランダムサーチ？）による最適化
# ---------------------------
start = time.time()


input_size=30
num_variables = T

# 結果保存用ディレクトリの作成
dirname3=f"cost-RS-MOMY-T={T}-opt={opt_num}"
os.makedirs(dirname3, exist_ok=True)

convergence=f"convergence-RS-MOMY-opt={opt_num}"
os.makedirs(convergence, exist_ok=True)

learn_process=f"input-RS-learning-opt={opt_num}"
os.makedirs(learn_process, exist_ok=True)

# 各 PE の初期状態を org_file から init_file にコピーして初期化
for pe in range(nofpe):
    org = org_file.replace('######', str(pe).zfill(6))
    init = init_file.replace('######', str(pe).zfill(6))
    subprocess.run(["cp", org, init])

batch_size = n
# 6 ループ分の最適化を実施
for t in range(6):
    start_time = time.time()
    num_variables=T
    # 乱数生成器の設定（ループごとにシードを変える）
    rng_randint1 = np.random.default_rng(random_num+t)  # randint1用
    rng_randint2 = np.random.default_rng(random_num+10+t)  # randint2用
    rng_uniform = np.random.default_rng(random_num+20+t)

    
    best_values = []
    current_best = float('inf')
    # 各評価における主要制御入力（input1, input2, input3 の先頭要素）を記録する配列
    input_learn_process=np.zeros((opt_num,3))
    # 各 PE の初期状態をバックアップ (temp2_file にコピー)
    for pe in range(nofpe):
        temp2 = temp2_file.replace('######', str(pe).zfill(6))
        init = init_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", init, temp2])
    # 各評価のサンプル生成（連続変数は一様分布、整数は離散一様分布）
    samples_length = opt_num * num_variables
    samples1=[rng_uniform.uniform(-30, 30) for _ in range(samples_length)] #MOMY
    samples2=[rng_randint1.integers(0, 39) for _ in range(samples_length)] #y座標
    samples3=[rng_randint2.integers(0, 96) for _ in range(samples_length)] #z座標
    random_samples1=np.zeros((opt_num,num_variables))
    random_samples2=np.zeros((opt_num,num_variables))
    random_samples3=np.zeros((opt_num,num_variables))
    for i in range(opt_num):
        for j in range(num_variables):
            # サンプルを 2 次元配列に整形
            random_samples1[i][j] = samples1[num_variables*i+j] #縦opt_num 横num_variablesで埋める
            random_samples2[i][j] = samples2[num_variables*i+j]
            random_samples3[i][j] = samples3[num_variables*i+j]
        
    # 各パラメータ群を連結して各個体の表現とする
    combined_samples = np.concatenate((random_samples1, random_samples2,random_samples3), axis=1)
    # バッチに分割して評価（バッチサイズは n 個のジョブ）
    batches = [combined_samples[i:i + batch_size] for i in range(0, len(combined_samples), batch_size)]
    
    function_values=np.zeros(opt_num)
    # 各バッチごとに評価関数 f を実行し、評価値を取得
    for j in range(int(opt_num/n)):
        f_val=f(batches[j]) #n個jobのコスト関数の
        for k in range(n):
            function_values[n*j+k]=f_val[k]
            input_learn_process[n*j+k,0]=batches[j][k][0] #MOMY
            input_learn_process[n*j+k,1]=batches[j][k][T] #y座標
            input_learn_process[n*j+k,2]=batches[j][k][2*T] #z座標


    # 最良のサンプルとその評価値を取得
    min_index = np.argmin(function_values)
    best_sample = combined_samples[min_index]
    best_value = function_values[min_index]
    end_time = time.time()
    # 結果の表示
    print(f"Best sample: {best_sample}")
    print(f"Best function value: {best_value}")
    optimal_inputs=best_sample

    split_index1 = T
    split_index2 = T*2
    arr1 = optimal_inputs[:split_index1]
    arr2 = optimal_inputs[split_index1:split_index2]
    arr3 = optimal_inputs[split_index2:]

    # 収束プロットの作成
    x_iters=np.arange(1, opt_num+1)
    if(len(x_iters)==len(function_values)):
        plt.plot(x_iters,function_values)
        plt.savefig(f"{dirname3}/t={t}.png")
        plt.close()
        plt.clf()
    
    # 各評価回数での最良評価値を記録
    for j in range(len(function_values)):
        if function_values[j]<current_best:
            current_best=function_values[j]
        best_values.append(current_best)

    plt.figure(figsize=(10, 6))
    
    plt.plot(range(1, len(best_values) + 1), best_values, marker='o', linestyle='-', color='r')
    plt.title('Convergence of gp_minimize')
    plt.xlabel('Number of Evaluations')
    plt.ylabel('Best Objective Value Found')
    plt.grid(True)
    plt.savefig(f"{convergence}/convergence_plot{t}.png")
    save_file_path = f"{convergence}/best_history-loop{t}.txt"
    plt.close()
    plt.clf()

    x_iters=np.arange(1, opt_num+1)
    # 制御入力履歴のプロット（y: input2, z: input3, x: input1）
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
    # 最適な制御入力を初期状態ファイルに反映
    update_control(arr1,arr2,arr3)
    input_history1[t]=arr1[0]
    input_history2[t]=arr2[0]
    input_history3[t]=arr3[0]
    time_history[t]=end_time-start_time
    print(f"input_history1={input_history1},input_history2={input_history2},input_history3={input_history3}")
    # シミュレーション結果の最新状態を取得して更新
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

print(f"no={no}")
print(f"RS={gpy}")
print(f"change%={(no-gpy)/no*100}%")
print(f"%={(gpy)/no*100}%")