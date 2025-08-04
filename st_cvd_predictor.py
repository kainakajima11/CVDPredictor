import streamlit as st
import numpy as np
import pandas as pd
import torch
import plotly.graph_objs as go

def main():
    st.title("膜厚予測可視化")
    out_file = st.file_uploader("outファイルを選んでください")
    if out_file is not None:
        input_cols, mins, maxs = read_out(out_file)
        uploaded_model = st.file_uploader("modelを選んでください(ex. model_full.pth)", type=["pth"])

        if uploaded_model is not None:
            time_idx = next((i for i, x in enumerate(input_cols) if x == "S14_length"), None)
            input_dict = {k : 0 for k in input_cols}
            product_cols = [c for c in input_cols if c.startswith("品略")]

            selected_vars, input_values, temp_s12, temp_s14, para, target_film_thickness = st_receive_input(product_cols)
            inputs = make_inputs(input_dict, selected_vars, input_values, temp_s12, temp_s14, para)
            inputs_norm = normalize_arrs(inputs, mins, maxs)

            times = [30 + 5*i for i in range(30)]
            preds = predict(times, uploaded_model, inputs_norm, time_idx, mins, maxs)

            goplot_preds(times, preds, target_film_thickness)

            st.header("類似操炉検索")
            npz_file = st.file_uploader(".npzファイルを選んでください")
            if npz_file is not None:
                data = np.load(npz_file, allow_pickle=True)
                x_train = data["x_train"]
                y_train = data["y_train"]
                fons = data["others_train"]     
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
        modelファイルには学習済み機械学習モデルファイル(model_full.pth)を選択してください.    
        outファイルにはmodel_full.pthと同じディレクトリにあるoutファイルを選択してください.   
        炉詰めする製品を選択し（例えばOTを入れるなら、品略_OTを選択する）,その個数を入力してください.  
        複数の製品種類の操炉の場合は複数選択して、それぞれ個数を入力してください.  
        加えて温度条件、狙い膜厚をそれぞれ入力してください。狙い膜厚は予測結果には影響しません.  
        結果として、下に3本の曲線をグラフが得られます。 赤、青、緑がそれぞれ上部、中部、下部の予測膜厚です.  
        横軸は成膜時間です。黒い点線は入力した狙い膜厚です。<br>           

        #### 類似操炉検索について 
        操炉条件より、類似した条件の操炉の上位10個を出力します.  
        SCOREは類似度であり、1に近いほど類似しています.  
        例えば、1番類似している操炉が0.9であれば、似た操炉を過去に行っているということ、0.5程度であれば似た操炉を過去にやっていないことを表します.  
        似た操炉をやっていなければ、この予測は少し信頼性に欠けるということにもなります.  
    """)

def normalize_arrs(arrs, mins, maxs):
    return  (arrs - mins) / (maxs - mins)           

def denormalize_arrs(arrs, mins, maxs):
    return arrs * (maxs - mins) + mins
    
def search_vector(arr, q, ignore_dims, func = "CosineSimilarity", top_n = 10):
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
    if st_flag:
        lines = [line.decode("utf-8").strip() for line in out_file.readlines()]
    else:
        lines = out_file.readlines()
    get_line_signal, input_cols, mins, maxs = 0, None, None, None
    for line in lines:
        match get_line_signal:
            case 3:
                input_cols = list(line.split())
            case 2:
                mins = list(map(float, line.split()))
            case 1:
                maxs = list(map(float, line.split()))
            case _:
                spline = line.split()
                if (len(spline) >= 3) and spline[1] == "INPUT":
                    get_line_signal = 4
        get_line_signal = max(0, get_line_signal - 1)
    return input_cols, np.array(mins), np.array(maxs)

def predict(times, model_path, inputs, time_idx, mins, maxs):
    inputs = torch.tensor(inputs).float().unsqueeze(0)
    model = torch.load(model_path, weights_only=False)
    preds = np.zeros((len(times), 3), dtype=float)
    model.eval()
    with torch.no_grad():
        for i, time in enumerate(times):
            inputs[0][time_idx] = (time - mins[time_idx]) / (maxs[time_idx] - mins[time_idx])
            preds[i] = model(inputs).numpy()
    
    return preds.T

def make_inputs(input_dict, selected_vars, input_values, temp_s12, temp_s14, para):
    for var, val in zip(selected_vars, input_values):
        input_dict[var] = val
    input_dict["S12_Ave_TempUpper"] = temp_s12
    input_dict["S12_Ave_TempLower"] = temp_s12 + para
    input_dict["S14_Ave_TempUpper"] = temp_s14
    input_dict["S14_Ave_TempLower"] = temp_s14 + para
    input_dict["MTCS比率"] = 0.9
    return np.array(list(input_dict.values()))

def st_receive_input(products_list):
    selected_vars = st.multiselect("製品を選択してください（複数選択可）", products_list)
    input_values = []
    for var in selected_vars:
        val = st.number_input(f"{var} の値を入力", value=1, key=var)
        input_values.append(val)

    temp_s12 = st.number_input("成膜温度S12 (℃)", value=1155)
    temp_s14 = st.number_input("成膜温度S14 (℃)", value=1195)
    para = st.number_input("PARA値", value=0)
    target_film_thickness = st.number_input("狙い膜厚", value=60)

    return selected_vars, input_values, temp_s12, temp_s14, para, target_film_thickness

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
        xaxis_title="成膜時間3",
        yaxis_title="予想膜厚",
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

# for debug
def manual_main():
    out_file_path = r"C:\Users\kai.nakajima\Desktop\AMG\FilmThicknessPrediction\results\20250801_135815_BK\out"
    with open(out_file_path, "r",  encoding="utf-8") as f:
        input_cols, mins, maxs = read_out(f, st_flag=False)
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