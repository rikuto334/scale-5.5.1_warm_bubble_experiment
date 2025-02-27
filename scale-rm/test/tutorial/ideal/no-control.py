import matplotlib
import subprocess
import time

# matplotlib のバックエンドを 'Agg' に設定
# これにより、ディスプレイ環境がなくても画像ファイルを生成できるようになる
matplotlib.use('Agg')

# 各種パラメータの設定
nofpe = 2        # 使用するプロセス数（処理要素数）
fny = 2          # （用途に応じた）y方向の分割数
fnx = 1          # （用途に応じた）x方向の分割数
run_time = 20    # シミュレーションの実行時間（秒などの単位は用途に依存）
loop = 0         # シミュレーションループの初期カウンタ

varname = 'PREC'  # 変数名（例: 降水量などの気象変数を表す）

# ファイル名のテンプレート設定
# '######' の部分は後でプロセス番号（ゼロ埋め6桁）に置き換えられる
init_file = "init_00000101-000000.000.pe######.nc"    # 初期化用ファイル
org_file = "init_00000101-000000.000.pe######.org.nc"   # オリジナルの初期ファイル（バックアップ用）
history_file = "history.pe######.nc"                    # シミュレーション履歴ファイル

filebase = 'history.pe######.nc'    # 履歴ファイルのベース名
orgfile = 'history.pe######.org.nc'   # オリジナル履歴ファイル（必要に応じて利用）
file_path = '/home/rk-nagai/scale-5.5.1/scale-rm/test/tutorial/ideal'  # ファイルを格納するディレクトリ 自信のディレクトリに書き換えてください
output_file = f"history-{loop}.pe######.nc"  # 出力ファイル名（ループ回数に基づく）

# =====================================================
# 初期化：各プロセス（pe）ごとに、オリジナルの初期化ファイルから作業用初期化ファイルを作成
# =====================================================
for pe in range(nofpe):
    # プロセス番号をゼロ埋め6桁に変換して、テンプレート内の '######' を置換
    org = org_file.replace('######', str(pe).zfill(6))
    init = init_file.replace('######', str(pe).zfill(6))
    # subprocess.run を使ってシェルコマンド "cp" でファイルをコピー
    subprocess.run(["cp", org, init])

# =====================================================
# シミュレーションを実行する関数
# input: シミュレーションに渡すパラメータ（今回は使用していない）
# =====================================================
def sim(input):
    global loop
    global output_file
    # 現在のループ番号に応じた出力ファイル名を作成（プロセス番号部分は後で置換）
    output_file = f"history-{loop}.pe######.nc"

    # mpirun を使って並列実行
    # ここでは、2プロセスで "../../../scale-rm" 実行ファイルを、指定の設定ファイルで実行する
    subprocess.run(["mpirun", "-n", "2", "scale-rm", "run_R20kmDX500m-MPC.conf"])
    
    # シミュレーション実行後、各プロセスの履歴ファイルを出力用ファイル名にコピー
    for pe in range(nofpe):
        output = output_file.replace('######', str(pe).zfill(6))
        history = history_file.replace('######', str(pe).zfill(6))
        subprocess.run(["cp", history, output])
    
    return   # 関数終了

# =====================================================
# メイン処理の開始
# =====================================================
start = time.time()  # シミュレーション開始前の現在時刻を取得

D = 0  # シミュレーションに渡すパラメータ（詳細は用途に応じて調整）


# シミュレーションループ：ここでは6回のシミュレーションを実行
for i in range(6):
    sim(D)                    # シミュレーション関数の実行
    print(f"loop={loop}")      # 現在のループ番号を出力
    loop += 1                 # ループカウンタをインクリメント

end = time.time()  # シミュレーション終了後の現在時刻を取得
time_diff = end - start  # 総実行時間を計算

print(f'実行時間{time_diff}')  # 総実行時間を表示
