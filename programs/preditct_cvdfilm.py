import streamlit as st
import numpy as np
import pandas as pd
import torch
import plotly.graph_objs as go

from model import ThreeFullyConnectedLayers

def main():
    st.title("膜厚予測可視化")
    out_file = st.file_uploader("outファイルを選んでください")
    if out_file is not None:
        input_cols, mins, maxs, _n_train, _n_test, run_type = read_out(out_file)
        uploaded_model = st.file_uploader("modelを選んでください(ex. model_weights.pth)", type=["pth"])

        if uploaded_model is not None:
            if run_type == "Normal":
                time_idx = next((i for i, x in enumerate(input_cols) if x == "S14_length"), None) 
                times = [30 + 5*i for i in range(55)] # 30 ~ 300
            elif run_type == "Reduced":
                time_idx = next((i for i, x in enumerate(input_cols) if x == "S12_length"), None) 
                times = [70 + 5*i for i in range(67)] # 70 ~ 400
            input_dict = {k : 0 for k in input_cols}
            product_cols = [c for c in input_cols if c.startswith("品略")]

            # 炉詰め内容を受け取る
            cvdrun = CVDRun()
            cvdrun.receive_input(product_cols)
            inputs = cvdrun.make_inputs(input_dict)
            inputs_norm = normalize_arrs(inputs, mins, maxs)
            # 予測を行う
            preds = predict(times, len(input_cols), uploaded_model, inputs_norm, time_idx, mins, maxs)
            # プロット
            goplot_preds(times, preds, cvdrun.target_film_thickness)

            st.header("類似操炉検索")
            npz_file = st.file_uploader(".npzファイルを選んでください")
            if npz_file is not None:
                data = np.load(npz_file, allow_pickle=True)
                x_train = data["x_train"]
                y_train = data["y_train"]
                fons = data["others_train"]     
                # ベクトル検索
                top_idx, top_scores, top_arrs = search_vector(x_train, inputs_norm, ignore_dims=[11])
                top_arrs = denormalize_arrs(top_arrs, mins, maxs)
                top_fons = np.array(fons[top_idx])
                top_thicknesses = np.array(y_train[top_idx])

                combined = np.hstack([top_fons, top_scores.reshape(-1, 1), top_thicknesses, top_arrs])
                vector_search_cols = ["RunID", "SCORE"] + ["膜厚_上", "膜厚_中", "膜厚_下"] + input_cols
                df_query = pd.DataFrame([inputs], columns = input_cols)
                df_top = pd.DataFrame(combined, columns=vector_search_cols)
                st.write("今回の操炉内容")
                st.dataframe(df_query)
                st.write("内容の近い操炉")
                st.dataframe(df_top)

    st.markdown("""
        #### 膜厚予測について  
        操炉条件から膜厚を予測して、横軸を成膜時間(S14)、縦軸を予想膜厚としたグラフを作成します.  
        modelファイルには学習済み機械学習モデルファイル(model_weights.pth)を選択してください.    
        outファイルにはmodel_full.pthと同じディレクトリにあるoutファイルを選択してください.   
        炉詰めする製品を選択し（例えばOTを入れるなら、品略_OTを選択する）,その個数を入力してください.  
        複数の製品種類の操炉の場合は複数選択して、それぞれ個数を入力してください.  
        加えて温度条件、狙い膜厚をそれぞれ入力してください。狙い膜厚は予測結果には影響しません.  
        結果として、下に3本の曲線をグラフが得られます。 赤、青、緑がそれぞれ上部、中部、下部の予測膜厚です.  
        横軸は成膜時間です。黒い点線は入力した狙い膜厚です.        

        #### 類似操炉検索について 
        操炉条件より、類似した条件の操炉の上位10個を出力します.  
        SCOREは類似度であり、1に近いほど類似しています.  
        例えば、1番類似している操炉が0.9であれば、似た操炉を過去に行っているということ、0.5程度であれば似た操炉を過去にやっていないことを表します.  
        似た操炉をやっていなければ、この予測は少し信頼性に欠けるということにもなります.  
    """)

def normalize_arrs(arrs, mins, maxs):
    """正規化を行う"""
    return  (arrs - mins) / (maxs - mins)           

def denormalize_arrs(arrs, mins, maxs):
    """正規化をもとに戻す"""
    return arrs * (maxs - mins) + mins
    
def search_vector(arr, q, ignore_dims, func = "CosineSimilarity", top_n = 10):
    """ベクトル検索を行う"""
    use_dims = [i for i in range(arr.shape[1]) if i not in ignore_dims]
    arr_use = arr[:, use_dims]
    q_use = q[use_dims]

    if func == "EuclideanDistance":
        dists = np.linalg.norm(arr_use - q_use, axis=1)
        top_idx = np.argsort(dists)[:top_n]
        top_scores = dists[top_idx]
        top_arrs = arr[top_idx]
    elif func == "CosineSimilarity":
        q_use = q_use / np.linalg.norm(q_use)
        arr_norm = arr_use/ np.linalg.norm(arr_use, axis=1, keepdims=True)
        cos_sim = arr_norm @ q_use
        top_idx = np.argsort(cos_sim)[-top_n:][::-1]
        top_scores = cos_sim[top_idx]
        top_arrs = arr[top_idx]
    return top_idx, top_scores, top_arrs


def read_out(out_file, st_flag=True):
    """
    モデル構築時に出力されるoutファイルを読み込む。
    入力カラム、入力に対する最小値、最大値を返す。
    """
    if st_flag:
        lines = [line.decode("utf-8").strip() for line in out_file.readlines()]
    else:
        with open(out_file, encoding="utf-8") as f:
            lines = f.readlines()
    
    now_status = 0
    get_line_signal, input_cols, mins, maxs = 0, None, None, None
    n_train, n_test = None, None
    run_type = "Normal"
    # 1行ずつ見ていく
    for line in lines:
        spline = line.split()
        # INPUT関連情報を探しに行く
        if now_status == 1:
            match get_line_signal:
                case 3:
                    input_cols = list(line.split())
                case 2:
                    mins = list(map(float, line.split()))
                case 1:
                    maxs = list(map(float, line.split()))
                    now_status = 0
                case _:
                    pass
            get_line_signal -= 1
        elif now_status == 0:
            if (len(spline) >= 3) and spline[1] == "INPUT":
                now_status = 1
                get_line_signal = 3
            elif (len(spline) == 3) and spline[1] == "N_TRAIN":
                n_train = int(spline[2])
            elif (len(spline) == 3) and spline[1] == "N_TRAIN":
                n_test = int(spline[2])
            elif (len(spline) == 3) and spline[1] == "RUNTYPE":
                run_type = spline[2]

    return input_cols, np.array(mins), np.array(maxs), n_train, n_test, run_type

def predict(times, input_dim, model_path, inputs, time_idx, mins, maxs):
    inputs = torch.tensor(inputs).float().unsqueeze(0)
    model = ThreeFullyConnectedLayers(input_dim, 3)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    preds = np.zeros((len(times), 3), dtype=float)
    model.eval()
    with torch.no_grad():
        for i, time in enumerate(times):
            inputs[0][time_idx] = (time - mins[time_idx]) / (maxs[time_idx] - mins[time_idx])
            preds[i] = model(inputs).numpy()
    
    return preds.T

class CVDRun:
    def __init__(self,):
        self.s12_temp, self.s14_temp, self.para = None, None, None
        self.mtcs_rate = None
        self.target_film_thickness = None
        self.selected_vars = None
        self.input_values = []

    @property
    def s12_temp_upper(self):
        return self.s12_temp
    
    @property
    def s12_temp_lower(self):
        return self.s12_temp + self.para
    
    @property
    def s14_temp_upper(self):
        return self.s14_temp
    
    @property
    def s14_temp_lower(self):
        return self.s14_temp + self.para
    
    def receive_input(self, products_list):
        self.selected_vars = st.multiselect("炉詰めする製品を選択してください（複数選択可）", products_list)
        for var in self.selected_vars:
            val = st.number_input(f"{var} の個数を入力", value=1, key=var)
            self.input_values.append(val)
        self.s12_temp = st.number_input("成膜温度S12 (℃)", value=1145)
        self.s14_temp = st.number_input("成膜温度S14 (℃)", value=1185)
        self.para = st.number_input("PARA値", value=15)
        self.mtcs_rate = st.number_input("MTCS比率", value=0.9)
        self.target_film_thickness = st.number_input("狙い膜厚", value=60)

    def make_inputs(self, input_dict):
        for var, val in zip(self.selected_vars, self.input_values):
            input_dict[var] = val
        input_key_cand = ["S12_Ave_TempUpper", "S12_Ave_TempLower", "S14_Ave_TempUpper", "S14_Ave_TempLower", "MTCS比率"]
        input_value_cand = [self.s12_temp_upper, self.s12_temp_lower, self.s14_temp_upper, self.s14_temp_lower, self.mtcs_rate]
        for k, v in zip(input_key_cand, input_value_cand):
            if k in input_dict:
                input_dict[k] = v
        return np.array(list(input_dict.values()))
    
def goplot_preds(times, preds, target_film_thickness):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=preds[0], mode='lines+markers', name='Upper', line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=times, y=preds[1], mode='lines+markers', name='Middle', line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=times, y=preds[2], mode='lines+markers', name='Lower', line=dict(color='green')
    ))
    fig.add_trace(go.Scatter(
        x=times, y=[target_film_thickness] * len(times), mode='lines',
        name='Target', line=dict(color='black', dash='dash')
    ))
    fig.update_layout(
        title="成膜時間-予想膜厚",
        xaxis_title="成膜時間",
        yaxis_title="予想膜厚",
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# for debug
def manual_main():
    out_file_path = r"C:\Users\kai.nakajima\Desktop\AMG\FilmThicknessPrediction\results\20250801_135815_BK\out"
    with open(out_file_path, "r",  encoding="utf-8") as f:
        input_cols, mins, maxs, n_train, n_test, run_type = read_out(f, st_flag=False)
    model = r"C:\Users\kai.nakajima\Desktop\AMG\FilmThicknessPrediction\results\20250801_135815_BK\model_full.pth"
    time_idx = next((i for i, x in enumerate(input_cols) if x == "S14_length"), None)

    inputs = np.array([0, 0, 5, 4, 4, 0, 0, 0, 0, 0, 0, 0, 200, 1155, 1160, 1195, 1200, 0.9])
    inputs_norm = normalize_arrs(inputs, mins, maxs)

    times = [30 + 5*i for i in range(30)]
    preds = predict(times, model, inputs_norm, time_idx, mins, maxs)

    npz_file = r"C:\Users\kai.nakajima\Desktop\AMG\FilmThicknessPrediction\results\20250801_135815_BK\dataset.npz" # st.file_uploader(".npzファイルを選んでください")
    if npz_file is not None:
        data = np.load(npz_file, allow_pickle=True)
        x_train = data["x_train"]
        fons = data["others_train"]     
        top_idx, top_dists, top_arrs = search_vector(x_train, inputs_norm, ignore_dims=[11])
        top_arrs = denormalize_arrs(top_arrs, mins, maxs)
        top_fons = np.array(fons[top_idx])
        combined = np.hstack([top_fons, top_dists.reshape(-1, 1), top_arrs])
        vector_search_cols = ["RunID", "DISTANCE"] + input_cols
        df_query = pd.DataFrame([inputs], columns = input_cols)
        df_top = pd.DataFrame(combined, columns=vector_search_cols)

if __name__ == "__main__":
    main()
    # manual_main()