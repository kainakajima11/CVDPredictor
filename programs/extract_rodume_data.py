import pandas as pd
import polars as pl
import yaml
import argparse

def main():
    args = parse() # 引数を受け取る
    config = args_to_config(args) # configを定義
    ps = ProductionSchedule(config["paths"])
    ps.deal_per_line(config["products_corrections"]) # 生産計画を1行ずつ処理
    ps.to_dict() # dfのもとになる辞書型に変形
    ps.dict_to_csv(config["run_id_unused"], config["product_unused"], config["out_path"]) # dfにして、csvとして出力

class ProductionSchedule:
    def __init__(self, schedule_paths):
        self.prepare_df(schedule_paths)
        self.rodume_info = {} # 操炉ごとに何が何個入っているか # 生産カレンダーの行ごとに記録 
        self.product_set = set() # 生産カレンダー全体にどんな品略があるか
        self.product_maps_nlist = {} # 操炉ごとに何が何個入っているか # 品略ごとに記録

    def deal_per_line(self, products_corrections): 
        """生産計画を1行ずつ処理"""
        rodume_list = [] # その操炉が製品を何個持っているかを (品略, 個数)で管理, 品略は同じものが含まれる。
        runid = "" # 現在見ている行の操炉番号
        for _, row in self.df.iterrows():
            rodume_list, runid = self.update_status(row["計画ﾅﾝﾊﾞｰ"], row["列1"], rodume_list, runid) # 今見ている行により、操炉番号を更新
            if pd.notna(row["品略"]) and pd.notna(row["個数"]):
                row["品略"] = self.correct_product(row["品略"], products_corrections) # 品略を正す
                if f"品略_{row["品略"]}" not in self.product_set:
                    self.product_set.add(f"品略_{row["品略"]}")
                rodume_list.append((f"品略_{row["品略"]}", row["個数"]))
        self.rodume_info[runid] = rodume_list # 最後に現段階の内容で更新

    def dict_to_csv(self, delete_runid_list, del_products, out_path):
        """dictをdfにして、csvとして出力"""
        df = pl.DataFrame(self.product_maps_nlist)
        df = df.sort("操炉No")
        df = self.delete_runid(df, delete_runid_list)
        df = self.delete_products(df, del_products)
        pandf = df.to_pandas()
        pandf.to_csv(out_path, encoding="shift_jis", index=False)

    def to_dict(self):
        """dictにする"""
        product_list = sorted(list(self.product_set))

        # 製品と数字の対応関係を作成 # ex. {TB0: 0, TB6: 1, ..., OT: 15}
        products_maps_i = {}
        for i, p in enumerate(product_list):
            products_maps_i[p] = i
        
        # 最終的なdfになるdictを作成 
        # rodume_infoだと同じ品略が複数回含まれる。各品略ごとに個数をまとめたものをproduct_maps_nlistに入れる。# 同時に操炉Noも入れておく
        self.product_maps_nlist = {p: [] for p in product_list}
        self.product_maps_nlist["操炉No"] = [] 
        for k, v in self.rodume_info.items(): # k: fon_str, v: rodume_info: [(TB0: 1), (TB0: 1), (BO: 3), ...]
            num_hot = [0 for _ in range(len(product_list))]
            for p, n in v: # product_name, n: number of product
                num_hot[products_maps_i[p]] += int(n)
            for p in product_list:
                self.product_maps_nlist[p].append(int(num_hot[products_maps_i[p]]))
            self.product_maps_nlist["操炉No"].append(k)     

    def update_status(self, no, col1, rodume_list, runid):
        if (
            isinstance(no, str)
            and no.startswith("計画")
            and pd.notna(col1)
        ): # 新しい操炉の行に入った時
            if runid != "":
                self.rodume_info[runid] = rodume_list # 最初以外なら、前の操炉の内容を確定
            runid = f"A{no[2:4]}-{int(col1):03}" # 操炉番号の更新
            rodume_list = [] # 初期化
        return rodume_list, runid

    def prepare_df(self, schedule_paths):
        dfs = []
        for path in schedule_paths: # 生産カレンダーをDataFrameとして読み込む
            tdf = pd.read_excel(
                path,
                skiprows=1,
                sheet_name="生産計画"
                )
            dfs.append(tdf)
        self.df = pd.concat(dfs, axis=0, ignore_index=True) # 複数あれば結合
        self.df["計画ﾅﾝﾊﾞｰ"] = self.df["計画ﾅﾝﾊﾞｰ"].astype(str)

    def correct_product(self, product, products_corrections): 
        """configで従って品略を訂正"""
        return products_corrections.get(product, product) 

    def delete_runid(self, df, delete_runid_list):
        """configに従って操炉Noを除去"""
        return df.filter(~pl.col("操炉No").is_in(
            delete_runid_list
        ))
    
    def delete_products(self, df, del_products):
        """configに従って製品種を除去"""
        del_products = [f"品略_{p}" for p in del_products]
        return df.select(pl.all().exclude(del_products))

def args_to_config(args):
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="config(yaml file)")
    return parser.parse_args()

if __name__ == "__main__":
    main()