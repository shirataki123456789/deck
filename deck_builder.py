import streamlit as st
import pandas as pd
import json
import os
import qrcode
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import re 
import requests 
from io import BytesIO 
# 💡 修正: ThreadPoolExecutorを使用しないため、関連するimportを削除
# from concurrent.futures import ThreadPoolExecutor, as_completed 
# 💡 修正: pyzbarの代わりにOpenCVとNumpyをインポート
import cv2
import numpy as np

# ===============================
# 🛠️ 修正 1: アプリ全体を Wide Mode に設定
# ===============================
st.set_page_config(layout="wide")

# ===============================
# 📱 修正 3 (最終手段: CSS Grid版): モバイルでの列崩れを防止するCSS
# ===============================
st.markdown("""
<style>
@media (max-width: 768px) { /* モバイルとタブレットのブレークポイント */
    
    /* st.columns で作られるコンテナ (親) */
    div[data-testid="stHorizontalBlock"] {
        /* flexbox ではなく grid でレイアウトすることを強制 */
        display: grid !important;
        
        /* 1fr 1fr 1fr は「利用可能なスペースを3等分する」という意味です。
         これにより、iPhoneの画面幅でも強制的に3つの列を作ります。
        */
        grid-template-columns: 1fr 1fr 1fr !important; 
        
        /* 列と行の隙間を指定 */
        gap: 0.75rem !important; 
        
        /* Streamlitが設定する可能性のあるflex関連のプロパティをリセット */
        flex-direction: unset !important;
        flex-wrap: unset !important;
    }
    
    /* st.columns の中の各列 (子) */
    div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
        
        /* Streamlitが設定する width: 100% や flex-basis を上書き */
        /* width: auto または 100% (gridアイテムは親に依存するため) */
        width: 100% !important; 
        
        /* flexアイテムとしての挙動をリセット */
        flex: unset !important;
        min-width: unset !important; /* 最小幅もリセット */
        
        /* 不要なマージンをリセット */
        margin: 0 !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ===============================
# 🧠 キャッシュ付きデータ読み込み
# ===============================
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    if not os.path.exists("cardlist_filtered.csv"):
        st.error("エラー: cardlist_filtered.csv が見つかりません。")
        return pd.DataFrame()
        
    df = pd.read_csv("cardlist_filtered.csv")
    df = df.fillna("-")
    
    # 特徴と属性の処理を統一（全角/半角スラッシュ対応）
    df["特徴リスト"] = df["特徴"].apply(lambda x: [f.strip() for f in str(x).replace("／", "/").split("/") if f.strip() and f.strip() != "-"])
    df["属性リスト"] = df["属性"].apply(lambda x: [f.strip() for f in str(x).replace("／", "/").split("/") if f.strip() and f.strip() != "-"])
    df["コスト数値"] = df["コスト"].replace("-", 0).astype(int)
    
    # 修正: 入手情報から【】内のシリーズ番号のみを抽出
    def extract_series_id(info):
        match = re.search(r'【(.*?)】', str(info))
        if match:
            return match.group(1).strip()
        return "その他" if str(info).strip() not in ["-", ""] else "-"
        
    df["シリーズID"] = df["入手情報"].apply(extract_series_id)
    
    return df

df = load_data()
if df.empty:
    st.stop()

# 無制限カードのリスト
UNLIMITED_CARDS = ["OP01-075", "OP08-072"]

# ===============================
# 🧩 並び順設定
# ===============================
color_order = ["赤", "緑", "青", "紫", "黒", "黄"]
color_priority = {c: i for i, c in enumerate(color_order)}
type_priority = {"LEADER": 0, "CHARACTER": 1, "EVENT": 2, "STAGE": 3}

def color_sort_key(row):
    text = str(row["色"])
    t = str(row["タイプ"])
    if text.strip() == "-" or text.strip() == "":
        return (999, 999, 999, 999)

    found_colors = [c for c in color_order if c in text]
    if not found_colors:
        return (999, 999, 999, 999)

    first_color = found_colors[0]
    base_priority = color_priority[first_color]

    is_multi = "/" in text or "／" in text
    sub_colors = [c for c in color_order if c in text and c != first_color]
    sub_priority = color_order.index(sub_colors[0]) + 1 if is_multi and sub_colors else 0
    multi_flag = 1 if is_multi else 0

    type_rank = type_priority.get(t, 9)
    return (base_priority, type_rank, sub_priority, multi_flag)

df["ソートキー"] = df.apply(color_sort_key, axis=1)

# ===============================
# 💾 セッション初期化
# ===============================
if "leader" not in st.session_state:
    st.session_state["leader"] = None
if "deck" not in st.session_state:
    st.session_state["deck"] = {}
if "mode" not in st.session_state:
    st.session_state["mode"] = "検索"
if "deck_view" not in st.session_state:
    st.session_state["deck_view"] = "leader"
if "deck_name" not in st.session_state:
    st.session_state["deck_name"] = ""
if "search_cols" not in st.session_state: 
    st.session_state["search_cols"] = 3
if "qr_upload_key" not in st.session_state: 
    st.session_state["qr_upload_key"] = 0
    
# デッキ追加画面用のフィルタ状態を初期化
if "deck_filter" not in st.session_state:
    st.session_state["deck_filter"] = {
        "colors": [],
        "types": [], # 💡 修正: 初期選択を空リストに変更
        "costs": [],
        "counters": [],
        "attributes": [],
        "blocks": [],
        "features": [],
        "series_ids": [],
        "free_words": ""
    }

# ===============================
# 🔍 検索関数
# ===============================
def filter_cards(df, colors, types, costs, counters, attributes, blocks, feature_selected, free_words, series_ids=None, leader_colors=None):
    results = df.copy()

    # デッキ作成モードの場合、リーダーの色に基づいてフィルタ
    if leader_colors:
        results = results[results["タイプ"] != "LEADER"]
        results = results[results["色"].apply(lambda c: any(lc in c for lc in leader_colors))]

    if colors:
        results = results[results["色"].apply(lambda c: any(col in c for col in colors))]

    if types:
        results = results[results["タイプ"].isin(types)]

    if costs:
        results = results[results["コスト数値"].isin(costs)]

    if counters:
        results = results[results["カウンター"].isin(counters)]

    if attributes:
        results = results[results["属性リスト"].apply(lambda lst: any(attr in lst for attr in attributes))]

    if blocks:
        results = results[results["ブロックアイコン"].isin(blocks)]
        
    # シリーズIDフィルタ
    if series_ids:
        results = results[results["シリーズID"].isin(series_ids)]

    if feature_selected:
        results = results[results["特徴リスト"].apply(lambda lst: any(f in lst for f in feature_selected))]

    if free_words:
        keywords = free_words.split()
        for k in keywords:
            results = results[
                results["カード名"].str.contains(k, case=False, na=False) |
                results["特徴"].str.contains(k, case=False, na=False) |
                results["テキスト"].str.contains(k, case=False, na=False) |
                results["トリガー"].str.contains(k, case=False, na=False)
            ]

    results = results.sort_values(
        by=["ソートキー", "コスト数値", "カードID"], ascending=[True, True, True]
    )
    return results

# ===============================
# 🖼️ デッキ画像生成関数 
# ===============================
@st.cache_data(ttl=3600, show_spinner=False) 
def create_deck_image(leader, deck_dict, df, deck_name=""):
    """デッキリストの画像を生成（カード画像＋QRコード付き）2150x2048固定サイズ"""
    
    # 最終画像サイズ
    FINAL_WIDTH = 2150
    FINAL_HEIGHT = 2048
    
    # カードグリッドの高さ 
    GRID_HEIGHT = 1500 
    
    # 上セクションの最大高さ
    UPPER_HEIGHT = FINAL_HEIGHT - GRID_HEIGHT
    
    # リーダーの色を取得
    leader_color_text = leader["色"]
    leader_colors = [c.strip() for c in leader_color_text.replace("／", "/").split("/") if c.strip()]
    
    # 色から背景色を取得 
    color_map = {
        "赤": "#AC1122", "緑": "#008866", "青": "#0084BD", 
        "紫": "#93388B", "黒": "#211818", "黄": "#F7E731"
    }
    
    # デッキリストテキスト生成
    deck_lines = []
    if deck_name:
        deck_lines.append(f"# {deck_name}")
    deck_lines.append(f"1x{leader['カードID']}")
    
    deck_cards_sorted = []
    for card_id, count in deck_dict.items():
        card_row = df[df["カードID"] == card_id].iloc[0]
        base_priority, type_rank, sub_priority, multi_flag = card_row["ソートキー"]
        deck_cards_sorted.append({
            "card_id": card_id,
            "count": count,
            "new_sort_key": (type_rank, card_row["コスト数値"], base_priority, card_id), 
            "cost": card_row["コスト数値"]
        })
    deck_cards_sorted.sort(key=lambda x: x["new_sort_key"])
    
    for card_info in deck_cards_sorted:
        deck_lines.append(f"{card_info['count']}x{card_info['card_id']}")
    deck_text = "\n".join(deck_lines)
    
    # QRコード生成
    qr = qrcode.QRCode(version=1, box_size=8, border=2)
    qr.add_data(deck_text)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    QR_SIZE = 400
    qr_img = qr_img.resize((QR_SIZE, QR_SIZE), Image.LANCZOS)
    
    # カード画像のサイズ（下部グリッド用）
    card_width = 215
    card_height = 300
    
    # 💡 修正 2B-1: モバイル対応のため、グリッドの列数を10から3に変更（画像生成時は10列を維持）
    cards_per_row = 10 
    cards_per_col = 5
    
    margin_card = 0
    
    # 画像作成 (RGBAモードで初期化)
    img = Image.new('RGBA', (FINAL_WIDTH, FINAL_HEIGHT), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # 背景色（グラデーション対応）
    if len(leader_colors) == 1:
        bg_color = color_map.get(leader_colors[0], "#FFFFFF")
        draw.rectangle([0, 0, FINAL_WIDTH, FINAL_HEIGHT], fill=bg_color)
    elif len(leader_colors) >= 2:
        color1 = color_map.get(leader_colors[0], "#FFFFFF")
        color2 = color_map.get(leader_colors[1], "#FFFFFF")
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        rgb1, rgb2 = hex_to_rgb(color1), hex_to_rgb(color2)

        for x in range(FINAL_WIDTH):
            ratio = x / FINAL_WIDTH
            r = int(rgb1[0] * (1 - ratio) + rgb2[0] * ratio)
            g = int(rgb1[1] * (1 - ratio) + rgb2[1] * ratio)
            b = int(rgb1[2] * (1 - ratio) + rgb2[2] * ratio)
            draw.line([(x, 0), (x, FINAL_HEIGHT)], fill=(r, g, b))
    
    # 画像ダウンロード関数
    def download_card_image(card_id, target_size, crop_top_half=False):
        try:
            card_url = f"https://www.onepiece-cardgame.com/images/cardlist/card/{card_id}.png"
            response = requests.get(card_url, timeout=5)
            if response.status_code == 200:
                card_img = Image.open(BytesIO(response.content)).convert("RGBA")
                
                if crop_top_half:
                    CROPPED_WIDTH = target_size[0]
                    CROPPED_HEIGHT = target_size[1]
                    
                    full_height_target = CROPPED_HEIGHT * 2 
                    card_img = card_img.resize((CROPPED_WIDTH, full_height_target), Image.LANCZOS)
                    
                    card_img = card_img.crop((0, 0, CROPPED_WIDTH, CROPPED_HEIGHT))
                else:
                    card_img = card_img.resize(target_size, Image.LANCZOS) 
                    
                return card_id, card_img
        except Exception as e:
            return card_id, None

    # --- 上セクションの配置（リーダー → デッキ名 → QR） ---
    
    GAP = 48 
    
    LEADER_CROPPED_HEIGHT = UPPER_HEIGHT 
    LEADER_CROPPED_WIDTH = int(LEADER_CROPPED_HEIGHT * (400 / 280)) 
    LEADER_TARGET_SIZE = (LEADER_CROPPED_WIDTH, LEADER_CROPPED_HEIGHT) 
    
    QR_SIZE = 400
    
    DECK_NAME_AREA_WIDTH = FINAL_WIDTH - (GAP * 3) - LEADER_CROPPED_WIDTH - QR_SIZE 

    leader_x = GAP 
    deck_name_area_start_x = leader_x + LEADER_CROPPED_WIDTH + GAP 
    qr_x = deck_name_area_start_x + DECK_NAME_AREA_WIDTH + GAP 
    
    leader_y = 0 
    qr_y = (UPPER_HEIGHT - QR_SIZE) // 2 

    # 1. リーダー画像を配置 
    try:
        _, leader_img = download_card_image(leader['カードID'], LEADER_TARGET_SIZE, crop_top_half=True) 
        if leader_img:
            img.paste(leader_img, (leader_x, leader_y), leader_img) 
    except:
        pass

    # 3. QRコードを配置 
    img.paste(qr_img.convert("RGBA"), (qr_x, qr_y), qr_img.convert("RGBA"))
    
    # 2. デッキ名（中央）
    if deck_name:
        FONT_SIZE = 70
        font_name = None 
        
        # 💡 最終修正: アプリに同梱したフォントを最優先で試行する
        # ファイル名: meiryo.ttc (事前にアップロードが必要です)
        BUNDLED_FONT_PATH = "meiryo.ttc"

        # Streamlit Cloud環境での文字化け対策として、以下の順で試行
        font_paths_to_try = [
            # 1. アプリに同梱したフォント（最優先）
            (BUNDLED_FONT_PATH, None),
            
            # 2. 前回の修正で試したStreamlit Cloud/Linux 環境の標準パス
            ("/usr/share/fonts/truetype/noto/NotoSansJP-Regular.otf", None),
            ("/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf", None),
            ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0), 
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", None),
            ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", None),

            # 3. ローカル Windows のパス (Streamlit Cloudでは無視される)
            ("C:\\Windows\\Fonts\\meiryo.ttc", 0),
            ("C:\\Windows\\Fonts\\msgothic.ttc", 0),
        ]
        
        for path, index in font_paths_to_try:
            try:
                if os.path.exists(path): # ファイルが存在するかチェックを追加
                    if index is not None:
                        font_name = ImageFont.truetype(path, FONT_SIZE, index=index)
                    else:
                        font_name = ImageFont.truetype(path, FONT_SIZE)
                    break # 成功したらループを抜ける
            except IOError:
                continue # 次のフォントを試す

        # 💡 最終フォールバック
        if font_name is None:
             font_name = ImageFont.load_default() 
        
        try:
            # 描画処理
            bbox = draw.textbbox((0, 0), deck_name, font=font_name)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            BG_HEIGHT = text_height + 40 
            bg_x1 = deck_name_area_start_x + 50 
            bg_x2 = deck_name_area_start_x + DECK_NAME_AREA_WIDTH - 50
            bg_y1 = (UPPER_HEIGHT - BG_HEIGHT) // 2
            bg_y2 = bg_y1 + BG_HEIGHT

            overlay = Image.new('RGBA', (FINAL_WIDTH, FINAL_HEIGHT), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0, 128))
            
            img = Image.alpha_composite(img, overlay)
            draw = ImageDraw.Draw(img) 

            text_x = bg_x1 + (bg_x2 - bg_x1 - text_width) // 2
            text_y = bg_y1 + 20 

            draw.text((text_x, text_y), deck_name, fill="white", font=font_name)
            
        except Exception as e:
            # 描画中にエラーが発生した場合の最終手段（最終フォールバック）
            try:
                # デフォルトフォントで再描画を試みる（このfont_nameは小さいが、表示はされる）
                font_name = ImageFont.load_default()
                bbox = draw.textbbox((0, 0), deck_name, font=font_name)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                BG_HEIGHT = text_height + 40 
                bg_x1 = deck_name_area_start_x + 50 
                bg_x2 = deck_name_area_start_x + DECK_NAME_AREA_WIDTH - 50
                bg_y1 = (UPPER_HEIGHT - BG_HEIGHT) // 2
                bg_y2 = bg_y1 + BG_HEIGHT

                overlay = Image.new('RGBA', (FINAL_WIDTH, FINAL_HEIGHT), (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0, 128))
                
                img = Image.alpha_composite(img, overlay)
                draw = ImageDraw.Draw(img) 

                text_x = bg_x1 + (bg_x2 - bg_x1 - text_width) // 2
                text_y = bg_y1 + 20 

                draw.text((text_x, text_y), deck_name, fill="white", font=font_name)
            except:
                pass # 完全に失敗した場合は何もしない
    
    # 下セクション：デッキカード（10x5グリッド）
    y_start = UPPER_HEIGHT 
    x_start = (FINAL_WIDTH - (card_width * cards_per_row + margin_card * (cards_per_row - 1))) // 2
    
    all_deck_cards = []
    for card_info in deck_cards_sorted:
        all_deck_cards.extend([card_info['card_id']] * card_info['count'])
    
    card_images = {}
    cards_to_download = set(all_deck_cards[:cards_per_row * cards_per_col])
    
    # 💡 修正: ThreadPoolExecutorを削除し、同期的なダウンロードに変更
    # Pyodide/Streamlit Cloudなどの環境でマルチスレッドがエラーになるため
    with st.spinner("カード画像をダウンロード中..."):
        for card_id in cards_to_download:
            card_id, card_img = download_card_image(card_id, (card_width, card_height))
            if card_img:
                card_images[card_id] = card_img
    
    for idx, card_id in enumerate(all_deck_cards):
        if idx >= cards_per_row * cards_per_col:
            break
        
        row = idx // cards_per_row
        col = idx % cards_per_row
        
        x = x_start + col * (card_width + margin_card)
        y = y_start + row * (card_height + margin_card)
        
        if card_id in card_images:
            img.paste(card_images[card_id], (x, y), card_images[card_id])
    
    return img.convert('RGB')

# ===============================
# 🎯 モード切替
# ===============================
st.sidebar.title("🎯 モード選択")

def set_mode_on_change():
    selected_label = st.session_state["mode_radio_key"]
    st.session_state["mode"] = "検索" if "検索" in selected_label else "デッキ"

mode_labels = ["🔍 カード検索", "🧱 デッキ作成"]
current_index = 0 if st.session_state["mode"] == "検索" else 1

st.sidebar.radio(
    "モード", 
    mode_labels, 
    index=current_index, 
    key="mode_radio_key", 
    on_change=set_mode_on_change, 
    label_visibility="collapsed"
)

# ===============================
# 🔍 カード検索モード 
# ===============================
if st.session_state["mode"] == "検索":
    st.title("🔍 カード検索")
    
    # --- 検索フィルタ（サイドバー） ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("検索フィルタ")
    
    colors = st.sidebar.multiselect("色を選択", color_order, key="search_colors")
    types = st.sidebar.multiselect("タイプを選択", list(type_priority.keys()), key="search_types")
    costs = st.sidebar.multiselect("コストを選択", sorted(df["コスト数値"].unique()), key="search_costs")
    counters = st.sidebar.multiselect("カウンターを選択", sorted(df["カウンター"].unique()), key="search_counters")
    
    all_attributes = sorted({attr for lst in df["属性リスト"] for attr in lst if attr})
    attributes = st.sidebar.multiselect("属性を選択", all_attributes, key="search_attributes")
    
    blocks = st.sidebar.multiselect("ブロックアイコン", sorted(df["ブロックアイコン"].unique()), key="search_blocks")
    
    all_features = sorted({f for lst in df["特徴リスト"] for f in lst if f})
    feature_selected = st.sidebar.multiselect("特徴を選択", all_features, key="search_features")
    
    # シリーズIDフィルタ 
    all_series_ids = sorted([s for s in df["シリーズID"].unique() if s != "-"])
    series_ids = st.sidebar.multiselect("入手シリーズを選択", all_series_ids, key="search_series_ids")

    # フリーワード検索
    free_words = st.sidebar.text_input("フリーワード検索（スペース区切り可）", key="search_free")
    
    # --- 検索ロジック (常に実行) ---
    st.session_state["search_results"] = filter_cards(
        df, colors, types, costs, counters, attributes, blocks, feature_selected, free_words, series_ids=series_ids
    )
    
    results = st.session_state["search_results"]
    
    # 該当カード数表示
    st.write(f"該当カード数：{len(results)} 枚")
    
    # --- 検索結果表示 ---
    # 💡 修正 2A: モバイルでの視認性を考慮し、2列を選択肢に追加
    selected_cols = st.sidebar.selectbox( 
        "1列あたりのカード数", 
        [2, 3, 4, 5], 
        # 既存の値がない/無効な場合は3列をデフォルトにする
        index=([2, 3, 4, 5].index(st.session_state.get("search_cols", 3)) 
               if st.session_state.get("search_cols", 3) in [2, 3, 4, 5] else 1), 
        key="search_cols_selectbox"
    )
    st.session_state["search_cols"] = selected_cols
    
    cols_count = st.session_state["search_cols"]
    cols = st.columns(cols_count) 
    for idx, (_, row) in enumerate(results.iterrows()):
        card_id = row['カードID']
        img_url = f"https://www.onepiece-cardgame.com/images/cardlist/card/{card_id}.png"
        
        with cols[idx % cols_count]: 
            # 💡 修正: use_column_width=True を use_container_width=True に置き換え
            st.image(img_url, use_container_width=True) 

# ===============================
# 🧱 デッキ作成モード
# ===============================
else:
    st.title("🧱 デッキ作成モード")
    
    # サイドバー：デッキ情報
    st.sidebar.markdown("---")
    st.sidebar.title("🧾 現在のデッキ")
    
    leader = st.session_state.get("leader")
    if leader is not None:
        st.sidebar.markdown(f"**リーダー:** {leader['カード名']} ({leader['カードID']})")
    
    if leader is not None:
        deck_name_input = st.sidebar.text_input("デッキ名", value=st.session_state.get("deck_name", ""), key="deck_name_input")
        if deck_name_input != st.session_state.get("deck_name", ""):
            st.session_state["deck_name"] = deck_name_input
    
    total_cards = sum(st.session_state["deck"].values())
    st.sidebar.markdown(f"**合計カード:** {total_cards}/50")
    
    if st.session_state["deck"]:
        deck_cards = []
        for card_id, count in st.session_state["deck"].items():
            card_row = df[df["カードID"] == card_id].iloc[0]
            # ソートキーを再計算 (元のコードのソートキー取得ロジックに合わせる)
            base_priority, type_rank, sub_priority, multi_flag = card_row["ソートキー"]
            deck_cards.append({
                "card_id": card_id,
                "count": count,
                "new_sort_key": (type_rank, card_row["コスト数値"], base_priority, card_id), 
                "cost": card_row["コスト数値"],
                "name": card_row["カード名"]
            })
        
        # 💡 ソートキー、コスト、カードIDでソート
        deck_cards.sort(key=lambda x: x["new_sort_key"])
        
        for card_info in deck_cards:
            
            # 💡 コンパクト表示: 名前(4) / +ボタン(1) / -ボタン(1)
            col_name, col_add, col_del = st.sidebar.columns([4, 1, 1])
            
            current = st.session_state["deck"].get(card_info['card_id'], 0)
            is_unlimited = card_info['card_id'] in UNLIMITED_CARDS
            
            with col_name:
                # 名前と枚数、IDをコンパクトに表示
                st.markdown(f"**{card_info['name']}** x {card_info['count']} *<small>({card_info['card_id']})</small>*", unsafe_allow_html=True)
            
            with col_add:
                if st.button("＋", key=f"add_sidebar_{card_info['card_id']}", width='stretch', 
                             disabled=(not is_unlimited and current >= 4)):
                    if is_unlimited or current < 4:
                        st.session_state["deck"][card_info['card_id']] = current + 1
                        st.rerun()
            with col_del:
                if st.button("−", key=f"del_{card_info['card_id']}", width='stretch', 
                             disabled=current == 0):
                    if st.session_state["deck"].get(card_info['card_id'], 0) > 0:
                        if st.session_state["deck"][card_info['card_id']] > 1:
                            st.session_state["deck"][card_info['card_id']] -= 1
                        else:
                            del st.session_state["deck"][card_info['card_id']]
                        st.rerun()
            
            st.sidebar.markdown("---")
    
    if total_cards > 50:
        st.sidebar.error("⚠️ 50枚を超えています！")
    elif total_cards < 50:
        st.sidebar.info(f"残り {50 - total_cards} 枚を追加できます。")
    else:
        st.sidebar.success("✅ デッキが完成しました！")
    
    # デッキ管理
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 デッキ管理")
    
    if leader is not None and st.sidebar.button("👁️ デッキプレビュー", key="preview_btn"):
        st.session_state["deck_view"] = "preview"
        st.rerun()
    
    SAVE_DIR = "saved_decks"
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # エクスポート機能（ロジック修正なし）
    if st.sidebar.button("📤 デッキをエクスポート"):
        if leader is None:
            st.sidebar.warning("リーダーを選択してください。")
        else:
            export_lines = []
            if st.session_state["deck_name"]:
                export_lines.append(f"# {st.session_state['deck_name']}")
            export_lines.append(f"1x{leader['カードID']}")
            
            deck_cards_sorted = []
            for card_id, count in st.session_state["deck"].items():
                card_row = df[df["カードID"] == card_id].iloc[0]
                base_priority, type_rank, _, _ = card_row["ソートキー"]
                deck_cards_sorted.append({
                    "card_id": card_id,
                    "count": count,
                    "new_sort_key": (type_rank, card_row["コスト数値"], base_priority, card_id),
                })
            deck_cards_sorted.sort(key=lambda x: x["new_sort_key"])
            
            for card_info in deck_cards_sorted:
                export_lines.append(f"{card_info['count']}x{card_info['card_id']}")
            
            export_text = "\n".join(export_lines)
            st.sidebar.text_area("エクスポートされたデッキ", export_text, height=200)
            st.sidebar.download_button(
                label="📥 テキストファイルとしてダウンロード",
                data=export_text,
                file_name=f"{st.session_state['deck_name']}_export.txt" if st.session_state["deck_name"] else "deck_export.txt",
                mime="text/plain"
            )
    
    # デッキ画像生成 (キャッシュのおかげで高速化)
    if st.sidebar.button("🖼️ デッキ画像を生成"):
        if leader is None:
            st.sidebar.warning("リーダーを選択してください。")
        else:
            with st.spinner("画像を生成中...（初回はカードダウンロードが同期処理のため時間がかかる場合があります）"):
                deck_name = st.session_state.get("deck_name", "")
                deck_img = create_deck_image(leader, st.session_state["deck"], df, deck_name)
                buf = io.BytesIO()
                deck_img.save(buf, format="PNG")
                buf.seek(0)
                # 💡 修正: use_column_width=True を use_container_width=True に置き換え
                st.sidebar.image(deck_img, caption="デッキ画像（QRコード付き）", use_container_width=True) 
                
                file_name = f"{deck_name}_deck.png" if deck_name else "deck_image.png"
                st.sidebar.download_button(
                    label="📥 画像をダウンロード",
                    data=buf,
                    file_name=file_name,
                    mime="image/png"
                )
    
    # インポート機能（OpenCV対応で修正）
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 デッキをインポート")
    
    st.sidebar.markdown("**QRコード画像からインポート**")
    uploaded_qr = st.sidebar.file_uploader(
        "QRコード画像をアップロード", 
        type=["png", "jpg", "jpeg"], 
        key=f"qr_upload_{st.session_state['qr_upload_key']}"
    )
    
    if uploaded_qr is not None:
        try:
            # 💡 OpenCVで画像を読み込み
            file_bytes = np.asarray(bytearray(uploaded_qr.read()), dtype=np.uint8)
            qr_image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            detector = cv2.QRCodeDetector()
            # 💡 QRコード検出とデータ取得
            qr_data, points, straight_qrcode = detector.detectAndDecode(qr_image_cv)
            
            if qr_data:
                st.sidebar.success("QRコードを読み取りました！")
                
                lines = [line.strip() for line in qr_data.strip().split("\n") if line.strip()]
                
                imported_deck_name = ""
                start_idx = 0
                if lines and lines[0].startswith("#"):
                    imported_deck_name = lines[0][1:].strip()
                    start_idx = 1
                
                if start_idx < len(lines):
                    first_line = lines[start_idx]
                    
                    if "x" not in first_line:
                        raise ValueError("デッキリスト形式が不正です（リーダー行に'x'がない）。")
                        
                    leader_count, leader_id = first_line.split("x")
                        
                    leader_row = df[df["カードID"] == leader_id]
                    if not leader_row.empty:
                        st.session_state["leader"] = leader_row.iloc[0].to_dict()
                        st.session_state["deck"] = {}
                        st.session_state["deck_name"] = imported_deck_name
                        
                        for line in lines[start_idx + 1:]:
                            if "x" in line:
                                count, card_id = line.split("x")
                                count = int(count)
                                if card_id in df["カードID"].values:
                                    st.session_state["deck"][card_id] = count
                        
                        st.session_state["deck_view"] = "preview"
                        st.sidebar.success("デッキをインポートしました！")
                        st.session_state["qr_upload_key"] += 1 
                        st.rerun()
                    else:
                        st.sidebar.error(f"リーダーカード {leader_id} が見つかりません。")
                else:
                    st.sidebar.error("デッキリストが空か、リーダーが特定できませんでした。")
            else:
                st.sidebar.warning("QRコードが検出されませんでした。")
        except Exception as e:
            # 💡 OpenCVのエラーもキャッチできるように修正
            st.sidebar.error(f"QRコード読み取りエラー: {str(e)}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 デッキをインポート")
    
    st.sidebar.markdown("**テキストからインポート**")
    import_text = st.sidebar.text_area("デッキリストを貼り付け", height=150, placeholder="1xOP03-040\n4xOP01-088\n...")
    
    if st.sidebar.button("📥 インポート実行"):
        if not import_text.strip():
            st.sidebar.warning("デッキリストを入力してください。")
        else:
            try:
                lines = [line.strip() for line in import_text.strip().split("\n") if line.strip()]
                if not lines:
                    st.sidebar.error("有効なデッキリストがありません。")
                else:
                    start_idx = 0
                    imported_deck_name = ""
                    if lines[0].startswith("#"):
                        imported_deck_name = lines[0][1:].strip()
                        start_idx = 1
                    
                    if start_idx < len(lines):
                        first_line = lines[start_idx]
                        if "x" not in first_line:
                             raise ValueError("デッキリスト形式が不正です（リーダー行に'x'がない）。")
                             
                        leader_count, leader_id = first_line.split("x")
                             
                        leader_row = df[df["カードID"] == leader_id]
                        if leader_row.empty:
                            st.sidebar.error(f"リーダーカード {leader_id} が見つかりません。")
                        else:
                            st.session_state["leader"] = leader_row.iloc[0].to_dict()
                            st.session_state["deck"] = {}
                            st.session_state["deck_name"] = imported_deck_name
                            
                            for line in lines[start_idx + 1:]:
                                if "x" in line:
                                    count, card_id = line.split("x")
                                    count = int(count)
                                    card_row = df[df["カードID"] == card_id]
                                    if not card_row.empty:
                                        st.session_state["deck"][card_id] = count
                            
                            st.session_state["deck_view"] = "preview"
                            st.sidebar.success("デッキをインポートしました！")
                            st.rerun()
            except Exception as e:
                st.sidebar.error(f"インポートエラー: {str(e)}")
    
    # ローカル保存・読込（削除機能を追加）
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 ローカル保存")
    
    current_deck_name = st.session_state.get("deck_name", "")
    
    if st.sidebar.button("💾 デッキを保存"):
        if not current_deck_name:
            st.sidebar.warning("デッキ名を入力してください。")
        elif leader is None:
            st.sidebar.warning("リーダーを選択してください。")
        else:
            save_lines = []
            if current_deck_name:
                save_lines.append(f"# {current_deck_name}")
            save_lines.append(f"1x{leader['カードID']}")
            
            deck_cards_sorted = []
            for card_id, count in st.session_state["deck"].items():
                card_row = df[df["カードID"] == card_id].iloc[0]
                base_priority, type_rank, _, _ = card_row["ソートキー"]
                deck_cards_sorted.append({
                    "card_id": card_id,
                    "count": count,
                    "new_sort_key": (type_rank, card_row["コスト数値"], base_priority, card_id),
                })
            deck_cards_sorted.sort(key=lambda x: x["new_sort_key"])
            
            for card_info in deck_cards_sorted:
                save_lines.append(f"{card_info['count']}x{card_info['card_id']}")
                
            save_text = "\n".join(save_lines)
            
            path = os.path.join(SAVE_DIR, f"{current_deck_name}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(save_text)
            st.sidebar.success(f"デッキ「{current_deck_name}」を保存しました。")
            st.rerun() # 保存後に選択肢を更新するためにリロード
    
    saved_files = [f[:-4] for f in os.listdir(SAVE_DIR) if f.endswith(".txt")]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 デッキの読み込みと削除")
    
    # デッキ読み込み
    col_load, col_del = st.sidebar.columns([3, 1])
    
    with col_load:
        selected_load = st.selectbox("読み込みまたは削除するデッキ", ["選択なし"] + saved_files, key="select_deck_to_manage")
    
    if selected_load != "選択なし":
        # 読み込みボタン
        if st.sidebar.button("📥 読み込む", key="load_saved_deck"):
            path = os.path.join(SAVE_DIR, f"{selected_load}.txt")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded_text = f.read()
            except FileNotFoundError:
                st.sidebar.error(f"ファイル {selected_load}.txt が見つかりません。")
                st.rerun()
                
            lines = [line.strip() for line in loaded_text.strip().split("\n") if line.strip()]
            
            imported_deck_name = ""
            start_idx = 0
            if lines and lines[0].startswith("#"):
                imported_deck_name = lines[0][1:].strip()
                start_idx = 1
            
            if start_idx < len(lines):
                first_line = lines[start_idx]
                if "x" not in first_line:
                    st.sidebar.error("読み込みファイルの内容が不正です。")
                    st.rerun()

                leader_count, leader_id = first_line.split("x")
                leader_row = df[df["カードID"] == leader_id]
                if not leader_row.empty:
                    st.session_state["leader"] = leader_row.iloc[0].to_dict()
                    st.session_state["deck"] = {}
                    st.session_state["deck_name"] = imported_deck_name
                    
                    for line in lines[start_idx + 1:]:
                        if "x" in line:
                            count, card_id = line.split("x")
                            count = int(count)
                            if card_id in df["カードID"].values:
                                st.session_state["deck"][card_id] = count
                    
                    st.session_state["deck_view"] = "preview"
                    st.sidebar.success(f"デッキ「{selected_load}」を読み込みました。")
                    st.rerun()
                else:
                    st.sidebar.error(f"リーダーカード {leader_id} が見つかりません。")
                    st.rerun()
            else:
                st.sidebar.error("読み込みファイルの内容が不正です。")
                st.rerun()

        # 💡 追加: 削除ボタン
        with col_del:
            if st.button("❌ 削除", key="delete_saved_deck"):
                path = os.path.join(SAVE_DIR, f"{selected_load}.txt")
                try:
                    os.remove(path)
                    st.sidebar.success(f"デッキ「{selected_load}」を削除しました。")
                    st.session_state["deck_view"] = "leader" # 削除後は初期画面に戻す
                    st.rerun() # ファイルリストを更新するためにリロード
                except FileNotFoundError:
                    st.sidebar.error(f"ファイル {selected_load}.txt が見つかりません。")
                except Exception as e:
                    st.sidebar.error(f"削除エラー: {str(e)}")
            
    
    # メインエリア：リーダー選択 / デッキプレビュー / カード追加
    if st.session_state["deck_view"] == "leader" or st.session_state["leader"] is None:
        st.subheader("① リーダーを選択")
        leaders = df[df["タイプ"] == "LEADER"]
        
        leaders = leaders.sort_values(by=["ソートキー", "コスト数値", "カードID"], ascending=[True, True, True])
        
        # 💡 モバイルでも見やすいように3列に固定
        cols = st.columns(3)
        for idx, (_, row) in enumerate(leaders.iterrows()):
            card_id = row['カードID'] # 💡 追加: card_idを取得
            img_url = f"https://www.onepiece-cardgame.com/images/cardlist/card/{card_id}.png"
            with cols[idx % 3]:
                # 💡 修正: use_column_width=True を use_container_width=True に置き換え
                st.image(img_url, caption=row["カード名"], use_container_width=True) 
                if st.button(f"選択", key=f"leader_{card_id}"):
                    st.session_state["leader"] = row.to_dict()
                    st.session_state["deck"].clear()
                    st.session_state["deck_name"] = ""
                    st.session_state["deck_view"] = "preview"
                    st.rerun()
    
    elif st.session_state["deck_view"] == "preview":
        leader = st.session_state["leader"]
        
        st.subheader("🃏 デッキプレビュー")
        
        # リーダー表示
        col1, col2 = st.columns([1, 3])
        with col1:
            leader_img_url = f"https://www.onepiece-cardgame.com/images/cardlist/card/{leader['カードID']}.png"
            # 💡 修正: use_column_width=True を use_container_width=True に置き換え
            st.image(leader_img_url, use_container_width=True) 
        with col2:
            st.markdown(f"**{leader['カード名']}**")
            st.markdown(f"色: {leader['色']}")
            st.markdown(f"カードID: {leader['カードID']}")
            if st.button("🔄 リーダーを変更"):
                st.session_state["leader"] = None
                st.session_state["deck"].clear()
                st.session_state["deck_view"] = "leader"
                if "deck_results" in st.session_state:
                    del st.session_state["deck_results"]
                st.rerun()
        
        st.markdown("---")
        
        # デッキカード表示
        st.markdown("### デッキ内のカード")
        if st.session_state["deck"]:
            deck_cards_sorted = []
            for card_id, count in st.session_state["deck"].items():
                card_row = df[df["カードID"] == card_id].iloc[0]
                base_priority, type_rank, sub_priority, multi_flag = card_row["ソートキー"]
                deck_cards_sorted.append({
                    "card_id": card_id,
                    "count": count,
                    "new_sort_key": (type_rank, card_row["コスト数値"], base_priority, card_id),
                    "cost": card_row["コスト数値"],
                    "name": card_row["カード名"]
                })
            
            deck_cards_sorted.sort(key=lambda x: x["new_sort_key"])
            
            # 💡 修正 2B-2: デッキプレビューの表示を3列に変更
            deck_cols = st.columns(3)
            col_idx = 0
            for card_info in deck_cards_sorted:
                card_img_url = f"https://www.onepiece-cardgame.com/images/cardlist/card/{card_info['card_id']}.png"
                
                with deck_cols[col_idx % 3]:
                    # 💡 修正: use_column_width=True を use_container_width=True に置き換え
                    st.image(card_img_url, use_container_width=True) 
                col_idx += 1
                
                # 3枚ごとに改行
                if col_idx % 3 == 0:
                     if col_idx < len(deck_cards_sorted) :
                         deck_cols = st.columns(3)
                         
        else:
            st.info("デッキにカードが追加されていません")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ カードを追加", key="add_card_btn", type="primary"):
                st.session_state["deck_view"] = "add_cards"
                st.rerun()
        with col2:
            if st.button("🔙 リーダー選択に戻る", key="back_to_leader_btn"):
                st.session_state["leader"] = None
                st.session_state["deck"].clear()
                st.session_state["deck_view"] = "leader"
                if "deck_results" in st.session_state:
                    del st.session_state["deck_results"]
                st.rerun()
    
    else:
        # ③ カード追加画面（検索フィルタを拡張）
        leader = st.session_state["leader"]
        leader_color_text = leader["色"]
        leader_colors = [c.strip() for c in leader_color_text.replace("／", "/").split("/") if c.strip()]
        
        st.subheader("➕ カードを追加")
        st.info(f"リーダー: {leader['カード名']}（{leader_color_text}） - **リーダーの色と同じカードのみが表示されます。**")
        
        if st.button("🔙 プレビューに戻る", key="back_to_preview_btn"):
            st.session_state["deck_view"] = "preview"
            st.rerun()
            
        st.markdown("---")
        
        st.subheader("🔍 カード検索フィルタ")
        
        # 既存のフィルタ状態を取得
        current_filter = st.session_state["deck_filter"]

        # UIの再構築：カード検索モードと同等のフィルタ
        # 💡 フィルタUIは3列を維持（コンテンツが多いため）
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            # 💡 修正: default=[] により初期選択をなしにする
            deck_types = st.multiselect("タイプ", ["CHARACTER", "EVENT", "STAGE"], default=current_filter["types"], key="deck_types")
            deck_costs = st.multiselect("コスト", sorted(df["コスト数値"].unique()), default=current_filter["costs"], key="deck_costs")
        with col_b:
            deck_counters = st.multiselect("カウンター", sorted(df["カウンター"].unique()), default=current_filter["counters"], key="deck_counters")
            all_deck_attributes = sorted({attr for lst in df["属性リスト"] for attr in lst if attr})
            deck_attributes = st.multiselect("属性", all_deck_attributes, default=current_filter["attributes"], key="deck_attributes")
        with col_c:
            all_deck_features = sorted({f for lst in df["特徴リスト"] for f in lst if f})
            deck_features = st.multiselect("特徴", all_deck_features, default=current_filter["features"], key="deck_features")
            all_series_ids = sorted([s for s in df["シリーズID"].unique() if s != "-"])
            deck_series_ids = st.multiselect("入手シリーズ", all_series_ids, default=current_filter["series_ids"], key="deck_series_ids")
            
        # 1行で配置
        col_d, col_e = st.columns([3, 1])
        with col_d:
            deck_free = st.text_input("フリーワード（カード名/特徴/テキスト/トリガー）", value=current_filter["free_words"], key="deck_free")
        with col_e:
            deck_blocks = st.multiselect("ブロックアイコン", sorted(df["ブロックアイコン"].unique()), default=current_filter["blocks"], key="deck_blocks")

        # フィルタ状態の更新
        st.session_state["deck_filter"] = {
            "colors": [], # リーダー色で絞られるため空
            "types": deck_types,
            "costs": deck_costs,
            "counters": deck_counters,
            "attributes": deck_attributes,
            "blocks": deck_blocks,
            "features": deck_features,
            "series_ids": deck_series_ids,
            "free_words": deck_free
        }
        
        # フィルタの自動適用
        st.session_state["deck_results"] = filter_cards(
            df, 
            colors=[], # リーダーの色で自動的にフィルタされる
            types=deck_types, 
            costs=deck_costs, 
            counters=deck_counters, 
            attributes=deck_attributes, 
            blocks=deck_blocks, 
            feature_selected=deck_features, 
            free_words=deck_free, 
            series_ids=deck_series_ids,
            leader_colors=leader_colors # リーダーの色を渡してフィルタリング
        )
        
        color_cards = st.session_state["deck_results"]
        
        # --- 💡 修正: 1列あたりのカード数選択 ---
        selected_cols = st.selectbox( 
            "1列あたりのカード数", 
            [2, 3, 4, 5], 
            # 検索モードと設定を共有するため、同じセッションステートを参照
            index=([2, 3, 4, 5].index(st.session_state.get("search_cols", 3)) 
                   if st.session_state.get("search_cols", 3) in [2, 3, 4, 5] else 1), 
            key="add_card_cols_selectbox" # 検索モードとキーを分ける
        )
        # st.session_state["search_cols"] を更新して、検索モードとの設定を共有する
        st.session_state["search_cols"] = selected_cols
        cols_count = st.session_state["search_cols"]
        # ----------------------------------------
        
        st.write(f"表示中のカード：{len(color_cards)} 枚")
        st.markdown("---")
        
        # 💡 修正 2B-3: 固定の3列ではなく、選択された列数を使用
        card_cols = st.columns(cols_count)
        for idx, (_, card) in enumerate(color_cards.iterrows()):
            img_url = f"https://www.onepiece-cardgame.com/images/cardlist/card/{card['カードID']}.png"
            card_id = card["カードID"]
            
            with card_cols[idx % cols_count]: # 💡 修正: 選択された列数を使用
                current_count = st.session_state["deck"].get(card_id, 0)
                # 💡 修正: use_column_width=True を use_container_width=True に置き換え
                st.image(img_url, caption=f"({current_count}/4枚)", use_container_width=True) 
                
                is_unlimited = card_id in UNLIMITED_CARDS
                
                # 📌 変更後: st.columns(2)を削除し、縦に配置
                
                # ＋ボタンを配置（画面幅いっぱいになる）
                if st.button("＋", key=f"add_deck_{card_id}_{idx}", type="primary", width='stretch', disabled=(not is_unlimited and current_count >= 4)):
                    count = st.session_state["deck"].get(card_id, 0)
                    if is_unlimited or count < 4:
                        st.session_state["deck"][card_id] = count + 1
                        st.rerun()
                
                # −ボタンを配置（＋ボタンの下に縦に並ぶ）
                if st.button("−", key=f"sub_deck_{card_id}_{idx}", width='stretch', disabled=current_count == 0):
                    if card_id in st.session_state["deck"] and st.session_state["deck"][card_id] > 0:
                        st.session_state["deck"][card_id] -= 1
                        if st.session_state["deck"][card_id] == 0:
                            del st.session_state["deck"][card_id]
                        st.rerun()
