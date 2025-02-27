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

# グローバルパラメータの設定
nofpe = 2       # 使用するプロセス数（PE: Processing Element）
fny = 2         # y方向のサブ領域数
fnx = 1         # x方向のサブ領域数
loop = 0        # ループ回数の初期値

varname = 'PREC' # 対象変数（降水量と想定）

# ファイル名のテンプレート（pe番号などの置換用）
init_file = "init_00000101-000000.000.pe######.nc"
sub_init_file = "0000/init_00000101-000000.000.pe######.nc"
org_file = "init_00000101-000000.000.pe######.org.nc"
history_file = "history.pe######.nc"
sub_history_file = "0000/history.pe######.nc"
restart_file = "restart_00000101-010000.000.pe000000.nc"

# その他ファイル名
orgfile = f"history-{loop}.pe######.nc"
now_file = f"now.pe######.nc"
temp_file=f"temp.pe######.nc"
temp2_file=f"temp2.pe######.nc"
file_path = '/home/rk-nagai/scale-5.5.1/scale-rm/test/tutorial/ideal' #自身のディレクトリに書き換えてください

# GA最適化結果の出力ファイル名（ループ番号などを含む）
gpyoptfile=f"gpyopt-{loop}.pe######.nc" #GAの結果を保存するファイル　名前は適当なので必要に応じて書き換えてください

# 経過入力値や処理時間、降水量の履歴を記録する配列
input_history1=np.zeros((6,1))
input_history2=np.zeros((6,1))
input_history3=np.zeros((6,1))
time_history=np.zeros((6,1))
prec=np.zeros(60)

# 制御後（GA適用後）と制御前の累積降水量を記録する配列
sum_gpy=np.zeros(40)
sum_no=np.zeros(40)

# 個体数（ジョブ数）並列数10以上で行う場合はcore数20以上のPCで実行してください
n=10
# 予測区間のステップ数
T = 6

# GAにおける評価関数の呼び出し回数（最適化反復回数）必要に応じて変更してください
opt_num=5

# GA用のカウンタ（各種乱数生成器のシードに利用）
init_cnt=0
blx_cnt=10
mutate_cnt=20
tournament_cnt=30



def predict(inputs,t):
 
    """
    シミュレーションを実行し、各個体（ジョブ）の評価値（累積降水量）を計算する関数
    inputs: 個体ごとの入力パラメータ（3*T 次元：連続変数＋2種類の整数変数）
    t: シミュレーションのタイムステップ数
    """
    # 各PE毎に現在のinitファイルをバックアップ（now_fileへコピー）
    for pe in range(nofpe):
            now = now_file.replace('######', str(pe).zfill(6))
            init = init_file.replace('######', str(pe).zfill(6))
            subprocess.run(["cp", init, now])#initを保存しておく
    global sub_history_file,sub_init_file
    total_result = np.zeros(n)  # 各個体の累積評価値を初期化

    # 各個体ごとにサブディレクトリへ初期状態（init_file）をコピー
    for i in range(n):
        for pe in range(nofpe):
            sub_init_file = f"000{i}/init_00000101-000000.000.pe######.nc"
            # 10並列以上の場合で実行する場合、ディレクトリ名を"00i"形式に変更
            if i>=10:
                sub_init_file = f"00{i}/init_00000101-000000.000.pe######.nc"
            sub_init = sub_init_file.replace('######', str(pe).zfill(6))
            init = init_file.replace('######', str(pe).zfill(6))
            subprocess.run(["cp", init, sub_init])

    for step in range(t):
        # 各個体内で前回のシミュレーション結果（historyファイル）があれば削除
        for i in range(n):
            for pe in range(nofpe):
                sub_history_file = f"000{i}/history.pe######.nc"
                if i>=10:
                    sub_init_file = f"00{i}/init_00000101-000000.000.pe######.nc"
                sub_history = sub_history_file.replace('######', str(pe).zfill(6))
                history_path = file_path+'/'+sub_history
                if (os.path.isfile(history_path)):
                    subprocess.run(["rm", sub_history])
            # 各個体に対して制御パラメータを反映
            # inputs[i][step]：連続変数、inputs[i][step+T]：整数（y位置）、inputs[i][step+2*T]：整数（z位置）
            control(i,inputs[i][step],inputs[i][step+T],inputs[i][step+2*T])

        # mpirun を使って並列シミュレーション実行
        result = subprocess.run(
            ["mpirun", "-n", "20", "scale-rm", "run.launch.conf"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(result.stdout)
        print(result.stderr)

        # 各個体の今回のタイムステップにおける評価値（降水量）を算出
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
                #対応する領域にデータを配置
                dat[:, 0, gy1:gy2, gx1:gx2] = nc[varname][:]
                # 40個のグリッドセルに対して降水量（600は600秒あたりの降水量を意味する）を累積
                for j in range(40):
                    step_result[i] += dat[1, 0, j, 0]*600
            total_result[i]+=step_result[i]
        print(f"Result at step {step}: {step_result}")
    # シミュレーション終了後、バックアップからinitファイルを復元
    for pe in range(nofpe):
        now = now_file.replace('######', str(pe).zfill(6))
        init = init_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", now, init])
    print(f"predict,loop={loop}")
    
    return total_result # 各個体の累積評価値を返す

def control(num,input1,input2,input3):
    """
    各個体（num番目）の制御ファイルを更新する関数
    input1: MOMY変数に加える値（連続変数）
    input2: 制御対象のy位置（整数）
    input3: 制御対象のz位置（整数）
    """
    global org_file
    for pe in range(nofpe):
        # 出力用ファイルとサブ初期ファイルのパスを設定
        output_file = f"000{num}/out-MOMY.pe######.nc"
        sub_init_file = f"000{num}/init_00000101-000000.000.pe######.nc"
        #並列数10以上の場合はディレクトリ名を変更
        if num>=10:
            output_file = f"00{num}/out-MOMY.pe######.nc"
            sub_init_file = f"00{num}/init_00000101-000000.000.pe######.nc"
        sub_init = sub_init_file.replace('######', str(pe).zfill(6))
        output = output_file.replace('######', str(pe).zfill(6))
        # netCDFファイルの読み込みとコピー（MOMY変数に対して制御入力を加える）
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
                    if pe == 0:
                        if input2<20:
                            var[int(input2), 0, int(input3)] += input1  # (y,x,z)
                    elif pe==1:
                        if input2>=20:
                            var[int(input2)-20, 0, int(input3)] += input1  # (y,x,z)
                            
                    dst[name][:] = var
                else:
                    dst[name][:] = src[name][:]
        # 出力ファイルをサブ初期ファイルに上書きコピー
        subprocess.run(["cp", output, sub_init ])
    return

def update_control(input1,input2,input3):
    """
    GA最適化後の最良制御入力を、初期状態ファイルに反映させる関数
    input1, input2, input3: それぞれ連続変数と2種類の整数変数（リスト形式）
    """
    global org_file
    
    for pe in range(nofpe):
        output_file = f"out-MOMY.pe######.nc"
        init = init_file.replace('######', str(pe).zfill(6))
        output = output_file.replace('######', str(pe).zfill(6))
        # initファイルを読み込み、MOMY変数に最適化入力を反映して新たなファイルを作成
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
      1. 現在のinitファイルをtemp_fileにバックアップ
      2. predict()を呼び出し、各個体の累積評価値（降水量）を算出
      3. バックアップからinitファイルを復元
      4. 評価値（各個体のコスト）を返す
    """
    for pe in range(nofpe):
        temp = temp_file.replace('######', str(pe).zfill(6))
        init = init_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", init, temp])
    
    cost_sum = predict(inputs,T)
    print(f"Cost at input {inputs}: Cost_sum {cost_sum}")
    
    for pe in range(nofpe):
        temp = temp_file.replace('######', str(pe).zfill(6))
        init = init_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", temp, init])
    return cost_sum # 各個体ごとの評価値（コスト）の配列を返す

def state_update():
    """
    シミュレーションの状態更新関数
      - 制御前後のhistoryファイルの整理
      - 別設定のシミュレーション実行により、制御結果を評価
      - 制御前後の降水量データを読み込み、累積値の更新とプロット生成
      - ループカウンタおよびタイムステップ T の更新
    """
    global loop,sum_gpy,sum_no,T
    orgfile = f"history-{loop}.pe######.nc"
    gpyoptfile=f"GA-SHMPC-{loop}-MOMY-opt{opt_num}-random{random_num}.pe######.nc"
    for pe in range(nofpe):
        history_file = "history.pe######.nc"
        history = history_file.replace('######', str(pe).zfill(6))
        history_path = file_path+'/'+history
        if (os.path.isfile(history_path)):
            subprocess.run(["rm", history])
    #MPIによるシミュレーションを実行
    subprocess.run(["mpirun", "-n", "2", "scale-rm", "run_R20kmDX500m-all-prec.conf"])
    
    # 実行結果のhistoryファイルをGA結果用のファイルにコピー
    for pe in range(nofpe):
        output = gpyoptfile.replace('######', str(pe).zfill(6))
        history = history_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", history,output])

    # 各PEのhistoryファイルから、領域ごとに降水量データを読み込み合成
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
    # 各タイムステップごとに制御前後の降水量の推移をプロットし、画像として保存
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
            dirname = f"GA-MOMY-={opt_num}-input={input_size}/"
            os.makedirs(dirname, exist_ok=True)
            filename = dirname + \
                f'sabun-MOMY-t={loop}.png'
            plt.ylim(0, 0.025)
            plt.savefig(filename)
            plt.clf()
    # ループカウンタを更新し、タイムステップ T を1減らす
    loop += 1
    T-=1
    return 

start = time.time()


input_size=30
num_variables = T
# GA（遺伝的アルゴリズム）に関するパラメータ設定

gene_length = 3 * T  # 各個体の遺伝子長（連続変数＋2種類の整数変数）
crossover_rate = 0.8  # 交叉率
mutation_rate = 0.05  # 変異率
lower_bound = -30.0   # 連続変数の下限
upper_bound = 30.0    # 連続変数の上限
alpha = 0.5           # BLX-α交叉のパラメータ
tournament_size = 3   # トーナメント選択の参加個体数

pop_size=n # 集団サイズ（個体数）

# 結果の収束履歴を保存するディレクトリの作成
convergence=f"convergence-GA-MOMY-opt={opt_num}"
os.makedirs(convergence, exist_ok=True)

##########################################################################
# GA関連の関数定義
##########################################################################


def initialize_population(pop_size, gene_length, lower_bound, upper_bound):
    """
    初期集団を生成する関数
      - 連続変数は一様分布で、整数変数はそれぞれの範囲からランダムに生成
    """
    global init_cnt
    rng_randint1 = np.random.default_rng(random_num+init_cnt)  
    rng_randint2 = np.random.default_rng(random_num+10+init_cnt)  
    rng_uniform = np.random.default_rng(random_num+20+init_cnt)
    random_samples1 = rng_uniform.uniform(-30, 30, (pop_size, T))
    random_samples2 = rng_randint1.integers(0, 39, (pop_size, T))
    random_samples3 = rng_randint2.integers(0, 96, (pop_size, T))
    combined_samples = np.concatenate((random_samples1, random_samples2,random_samples3), axis=1)
    init_cnt+=1
    return combined_samples

def tournament_selection(population, fitness, tournament_size):
    """
    トーナメント選択により、次世代の親個体を選出する関数
    """
    global tournament_cnt
    selected_parents = []
    rng = np.random.default_rng(tournament_cnt)
    for _ in range(len(population)):
        participants_idx = rng.choice(np.arange(len(population)), tournament_size, replace=False)
        best_idx = participants_idx[np.argmin(fitness[participants_idx])]
        selected_parents.append(population[best_idx])
    tournament_cnt+=1
    return np.array(selected_parents)

def blx_alpha_crossover(parents, offspring_size, alpha):
    """
    BLX-α交叉を用いて、親個体から子個体（オフスプリング）を生成する関数
      - 連続変数は一様乱数で生成、整数部分は下限・上限から乱数で選択
    """
    global blx_cnt
    offspring = np.empty(offspring_size)
    rng_randint1 = np.random.default_rng(random_num+blx_cnt)  # randint1用
    rng_randint2 = np.random.default_rng(random_num+10+blx_cnt)  # randint2用
    rng_uniform = np.random.default_rng(random_num+20+blx_cnt)

    for i in range(0, offspring_size[0], 2):
        parent1_idx = i % parents.shape[0]
        parent2_idx = (i + 1) % parents.shape[0]
        
        parent1 = parents[parent1_idx]
        parent2 = parents[parent2_idx]

        min_gene = np.minimum(parent1, parent2)
        max_gene = np.maximum(parent1, parent2)
        
        diff = max_gene - min_gene
        blx_lower_bound = min_gene - alpha * diff
        blx_upper_bound = max_gene + alpha * diff
        
        # 連続変数部分（input1）の交叉
        offspring[i, :T] = rng_uniform.uniform(blx_lower_bound[:T], blx_upper_bound[:T])
        offspring[i, :T] = np.clip(offspring[i, :T], lower_bound, upper_bound)
        
        # 整数変数部分（input2）の交叉
        lower_bound_int2 = np.clip(blx_lower_bound[T:2*T], 0, 39).astype(int)
        upper_bound_int2 = np.clip(blx_upper_bound[T:2*T], 1, 40).astype(int)

        for j in range(T):
            if lower_bound_int2[j]==upper_bound_int2[j]:
                offspring[i,j+T]=lower_bound_int2[j]
            else:
                offspring[i,j+T]=rng_randint1.integers(lower_bound_int2[j], upper_bound_int2[j])
        

        # 整数変数部分（input3）の交叉
        lower_bound_int3 = np.clip(blx_lower_bound[2*T:3*T], 0, 96).astype(int)
        upper_bound_int3 = np.clip(blx_upper_bound[2*T:3*T], 1, 97).astype(int)

        for j in range(T):
            if lower_bound_int3[j]==upper_bound_int3[j]:
                offspring[i,j+2*T]=lower_bound_int3[j]
            else:
                offspring[i,j+2*T]=rng_randint2.integers(lower_bound_int3[j], upper_bound_int3[j])

        # 次の子個体（i+1番目）の生成（同様の手順）
        if i + 1 < offspring_size[0]:
            offspring[i + 1, :T] = rng_uniform.uniform(blx_lower_bound[:T], blx_upper_bound[:T])
            offspring[i + 1, :T] = np.clip(offspring[i + 1, :T], lower_bound, upper_bound)
            for j in range(T):
                if lower_bound_int2[j]==upper_bound_int2[j]:
                    offspring[i+1,j+T]=lower_bound_int2[j]
                else:
                    offspring[i+1,j+T]=rng_randint1.integers(lower_bound_int2[j], upper_bound_int2[j])
            for j in range(T):
                if lower_bound_int3[j]==upper_bound_int3[j]:
                    offspring[i+1,j+2*T]=lower_bound_int3[j]
                else:
                    offspring[i+1,j+2*T]=rng_randint2.integers(lower_bound_int3[j], upper_bound_int3[j])
    blx_cnt+=1

    return offspring

def mutate(offspring, mutation_rate, lower_bound, upper_bound):
    """
    変異操作を行う関数
      - 各遺伝子について、所定の確率で連続変数は一様分布で、整数変数は範囲内の乱数に置換
    """
    global mutate_cnt
    rng_randint1 = np.random.default_rng(random_num+mutate_cnt)  
    rng_randint2 = np.random.default_rng(random_num+10+mutate_cnt)  
    rng_uniform = np.random.default_rng(random_num+20+mutate_cnt)
    rng = np.random.default_rng(random_num+30 + mutate_cnt)
    for idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if rng.random() < mutation_rate:
                if gene_idx<T:
                    offspring[idx, gene_idx] = rng_uniform.uniform(lower_bound, upper_bound)
                elif gene_idx>=T and gene_idx<2*T:
                    offspring[idx, gene_idx] = rng_randint1.integers(0, 39)
                else:
                    offspring[idx, gene_idx] = rng_randint2.integers(0, 96)
    mutate_cnt+=1
    return offspring

def genetic_algorithm(pop_size, gene_length, num_generations, crossover_rate,
                      mutation_rate, lower_bound, upper_bound, alpha, tournament_size,):
    """
    遺伝的アルゴリズムのメイン処理
      - 初期集団生成、適応度評価、選択、交叉、変異を各世代ごとに実施
      - 各世代での最良適応度、最良個体、評価履歴を返す
    """
    best_fitness = float("inf")
    best_individual = None
    best_values = []
    function_evals = []
    total_function_evals = 0  # 総関数評価回数
    input_learn_process=np.zeros((pop_size*num_generations,3))
    # 初期集団の生成
    population = initialize_population(pop_size, gene_length, lower_bound, upper_bound)

    for generation in range(num_generations):
        
        # 現在の個体群の適応度評価（f関数により評価）
        fitness = f(population)
        current_best_fitness = np.min(fitness)
        current_best_individual = population[np.argmin(fitness)]
        # 各個体の主要パラメータ（input1, input2, input3の先頭値）を記録
        for j in range(pop_size):
            input_learn_process[generation*pop_size+j,0]=population[j,0]
            input_learn_process[generation*pop_size+j,1]=population[j,T]
            input_learn_process[generation*pop_size+j,2]=population[j,2*T]

        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness.copy()
            best_individual = current_best_individual.copy()

        total_function_evals += pop_size  # 今回のイテレーションでの評価数を加算
        function_evals.append(total_function_evals)
        best_values.append(best_fitness)
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Best Individual = {best_individual}")
        # 親個体の選択（トーナメント選択）
        parents = tournament_selection(population, fitness, tournament_size)
        # 交叉によりオフスプリング生成
        offspring_size = (int(pop_size * crossover_rate), gene_length)
        offspring = blx_alpha_crossover(parents, offspring_size, alpha)
        # 変異操作の実施
        offspring = mutate(offspring, mutation_rate, lower_bound, upper_bound)
        # 生成したオフスプリングで集団の一部を更新
        population[0:offspring.shape[0]] = offspring
        

    return best_fitness, best_individual, best_values,function_evals,input_learn_process

# 1次元データをカンマ区切りで出力する関数
def save_single_line_with_trailing_comma(save_file_path, data):
    """
    1次元データをカンマ区切りの1行テキストとして保存する関数
      - 最後にもカンマを追加して出力
    """
    with open(save_file_path, 'w') as f:
        # data が単一の数値の場合はリストに変換
        if isinstance(data, (int, float, np.float64)):
            data = [data]  # リストに変換
        f.write(','.join(map(str, data)) + ',')  # 全ての値をカンマ区切りで1行にし、末尾にもカンマを追加

##########################################################################
# メイン処理
##########################################################################

# 初期状態として、org_fileからinit_fileへコピー（各PEごとに初期化）
for pe in range(nofpe):
    org = org_file.replace('######', str(pe).zfill(6))
    init = init_file.replace('######', str(pe).zfill(6))
    subprocess.run(["cp", org, init])

# GAの学習結果を保存するディレクトリを作成
learn_process=f"input-GA-learning-opt={opt_num}-random-{random_num}"
os.makedirs(learn_process, exist_ok=True)


batch_size = n
# 6回のGAループを実施
for t in range(6):
    start_time = time.time()
    gene_length = 3*T 
    best_values = []
    current_best = float('inf')
    input_learn_process=np.zeros((10*opt_num,3))
    # 各PEの初期状態をバックアップ（temp2_fileへコピー）
    for pe in range(nofpe):
        temp2 = temp2_file.replace('######', str(pe).zfill(6))
        init = init_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", init, temp2])
    # GAの実行：個体数10、評価回数 opt_num、交叉・変異・選択などのパラメータ指定
    best_fitness, best_individual,best_histrory,eval_history,input_learn_process = genetic_algorithm(
        10, gene_length, opt_num,
        crossover_rate, mutation_rate, lower_bound, upper_bound,
        alpha, tournament_size)
    end_time = time.time()

    # GAの収束グラフ（評価回数 vs 最良評価値）をプロットして保存
    plt.figure(figsize=(10, 5))
    plt.plot(eval_history, best_histrory, marker='o', linestyle='-', color='b')
    plt.title('Convergence of GA')
    plt.xlabel('Number of Function Evaluations')
    plt.ylabel('Best Objective Value Found')
    plt.grid(True)
    plt.savefig(f"{convergence}/cost_function_vs_generation{t}.png")
    save_file_path = f"{convergence}/best_history-loop{t}.txt"
    save_single_line_with_trailing_comma(save_file_path, best_histrory)
    plt.show()

    # 各個体の主要入力値（input1, input2, input3）を記録してプロット
    x_iters=np.arange(1, 10*opt_num+1)
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


    # 最良個体の遺伝子を3つのパラメータ群に分割
    split_index1 = T
    split_index2 = T*2
    arr1 = best_individual[:split_index1]
    arr2 = best_individual[split_index1:split_index2]
    arr3 = best_individual[split_index2:]

    # バックアップから初期状態を復元
    for pe in range(nofpe):
        temp2 = temp2_file.replace('######', str(pe).zfill(6))
        init = init_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", temp2, init])
    # 最良の制御入力をinitファイルへ反映
    update_control(arr1,arr2,arr3)
    input_history1[t]=arr1[0]
    input_history2[t]=arr2[0]
    input_history3[t]=arr3[0]
    time_history[t]=end_time-start_time
    print(f"input_history1={input_history1},input_history2={input_history2},input_history3={input_history3}")
    # 状態更新（シミュレーション結果の取得とプロット）
    state_update()
    print(f"loop={loop}")

end = time.time()
time_diff = end - start

# 総実行時間の計算と表示
sum_time=0
for i in range(6):
    sum_time+=time_history[i]

print(f'実行時間{time_diff}')


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

print(f"GA={gpy}")
print(f"change%={(no-gpy)/no*100}%")
print(f"%={(gpy)/no*100}%")