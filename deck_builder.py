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
import cv2
import numpy as np

# ===============================
# 🛠️ 修正 1: アプリ全体を Wide Mode に設定
# ===============================
st.set_page_config(layout="wide")

# ===============================
# 💡 修正 3: カスタムCSSの埋め込み
# ===============================

# カードを強制的に横並びにするFlexboxレイアウト
# モバイルでもカードの最小幅（150px）を維持し、オーバーフローでスクロール可能にする
CUSTOM_CSS = """
<style>
/* Streamlitのメインコンテンツのパディングを削除し、コンテンツ領域を最大化 */
.main .block-container {
    padding-top: 1rem;
    padding-right: 1rem;
    padding-left: 1rem;
    padding-bottom: 1rem;
}

/* カードリスト表示エリアのFlexbox設定 */
.card-list-container {
    display: flex;
    flex-wrap: wrap; /* 狭い画面では折り返しを許可 (ただし項目内が強制横並び) */
    gap: 10px; /* カード間の間隔 */
    margin-top: 10px;
    
    /* 狭い画面で全体を横スクロールさせるための設定 (オプション) */
    /* overflow-x: auto; */
    /* flex-wrap: nowrap; */ 
}

/* 各カードアイテムのスタイル */
.card-item {
    min-width: 150px; /* カードの最小表示幅 */
    max-width: 250px; /* 広い画面での最大幅 */
    flex-grow: 1; /* 成長を許可 */
    flex-basis: 150px; /* 基本的な幅 */
    text-align: center;
    overflow: hidden; /* 画像がはみ出さないように */
    padding-bottom: 20px;
}

/* 画像とボタンのコンテナ (縦に並べる) */
.card-content-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
}

/* Streamlitの画像をHTML/CSSで制御 */
.card-item img {
    width: 100%;
    height: auto;
    object-fit: contain;
    border-radius: 5px;
}

/* カードボタンのコンテナ */
.card-button-group {
    display: flex;
    width: 100%;
    max-width: 150px; /* ボタンの最大幅をカード幅に合わせる */
    margin-top: 5px;
}

.stButton>button {
    flex-grow: 1; /* ボタンが等幅になるように */
    margin: 0 2px !important;
}

/* デッキプレビューのカードリストを強制的に3列に維持し、モバイルでは横スクロールさせる */
.preview-card-list {
    display: flex;
    flex-wrap: nowrap; /* 強制的に折り返さない */
    gap: 10px;
    overflow-x: auto; /* 横スクロールを許可 */
    padding-bottom: 10px; /* スクロールバーのためのスペース */
}

/* デッキプレビューのカードアイテム */
.preview-card-item {
    flex: 0 0 150px; /* 幅を150pxに固定 */
    text-align: center;
}

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ===============================
# 🧠 キャッシュ付きデータ読み込み
# ... (変更なし)
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
# ... (変更なし)
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
# ... (変更なし)
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
        "types": [], 
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
# ... (変更なし)
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
# ... (変更なし: 前回の日本語フォント修正を維持)
# ===============================
@st.cache_data(ttl=3600, show_spinner=False) 
def create_deck_image(leader, deck_dict, df, deck_name=""):
    """デッキリストの画像を生成（カード画像＋QRコード付き）2150x2048固定サイズ"""
    
    FINAL_WIDTH = 2150
    FINAL_HEIGHT = 2048
    GRID_HEIGHT = 1500 
    UPPER_HEIGHT = FINAL_HEIGHT - GRID_HEIGHT
    leader_color_text = leader["色"]
    leader_colors = [c.strip() for c in leader_color_text.replace("／", "/").split("/") if c.strip()]
    color_map = {
        "赤": "#AC1122", "緑": "#008866", "青": "#0084BD", 
        "紫": "#93388B", "黒": "#211818", "黄": "#F7E731"
    }
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
    
    qr = qrcode.QRCode(version=1, box_size=8, border=2)
    qr.add_data(deck_text)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    QR_SIZE = 400
    qr_img = qr_img.resize((QR_SIZE, QR_SIZE), Image.LANCZOS)
    
    card_width = 215
    card_height = 300
    cards_per_row = 10 
    cards_per_col = 5
    margin_card = 0
    
    img = Image.new('RGBA', (FINAL_WIDTH, FINAL_HEIGHT), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    
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

    try:
        _, leader_img = download_card_image(leader['カードID'], LEADER_TARGET_SIZE, crop_top_half=True) 
        if leader_img:
            img.paste(leader_img, (leader_x, leader_y), leader_img) 
    except:
        pass

    img.paste(qr_img.convert("RGBA"), (qr_x, qr_y), qr_img.convert("RGBA"))
    
    if deck_name:
        try:
            FONT_SIZE = 70
            font_paths = [
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  
                "/usr/share/fonts/truetype/noto/NotoSansJP-Regular.otf",   
                "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",     
            ]
            
            font_name = ImageFont.load_default() 
            
            for path in font_paths:
                try:
                    font_name = ImageFont.truetype(path, FONT_SIZE)
                    break 
                except IOError:
                    continue 
                    
        except Exception:
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
    
    y_start = UPPER_HEIGHT 
    x_start = (FINAL_WIDTH - (card_width * cards_per_row + margin_card * (cards_per_row - 1))) // 2
    
    all_deck_cards = []
    for card_info in deck_cards_sorted:
        all_deck_cards.extend([card_info['card_id']] * card_info['count'])
    
    card_images = {}
    cards_to_download = set(all_deck_cards[:cards_per_row * cards_per_col])
    
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
# ... (変更なし)
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
    
    # ... (フィルタ UI のロジックは変更なし)
    colors = st.sidebar.multiselect("色を選択", color_order, key="search_colors")
    types = st.sidebar.multiselect("タイプを選択", list(type_priority.keys()), key="search_types")
    costs = st.sidebar.multiselect("コストを選択", sorted(df["コスト数値"].unique()), key="search_costs")
    counters = st.sidebar.multiselect("カウンターを選択", sorted(df["カウンター"].unique()), key="search_counters")
    all_attributes = sorted({attr for lst in df["属性リスト"] for attr in lst if attr})
    attributes = st.sidebar.multiselect("属性を選択", all_attributes, key="search_attributes")
    blocks = st.sidebar.multiselect("ブロックアイコン", sorted(df["ブロックアイコン"].unique()), key="search_blocks")
    all_features = sorted({f for lst in df["特徴リスト"] for f in lst if f})
    feature_selected = st.sidebar.multiselect("特徴を選択", all_features, key="search_features")
    all_series_ids = sorted([s for s in df["シリーズID"].unique() if s != "-"])
    series_ids = st.sidebar.multiselect("入手シリーズを選択", all_series_ids, key="search_series_ids")
    free_words = st.sidebar.text_input("フリーワード検索（スペース区切り可）", key="search_free")
    
    st.session_state["search_results"] = filter_cards(
        df, colors, types, costs, counters, attributes, blocks, feature_selected, free_words, series_ids=series_ids
    )
    
    results = st.session_state["search_results"]
    
    # 該当カード数表示
    st.write(f"該当カード数：{len(results)} 枚")
    
    # 検索結果表示
    # 💡 修正 3A: st.columns をやめ、HTML/CSS (card-list-container) で表示
    
    card_html = '<div class="card-list-container">'
    for idx, (_, row) in enumerate(results.iterrows()):
        card_id = row['カードID']
        img_url = f"https://www.onepiece-cardgame.com/images/cardlist/card/{card_id}.png"
        
        card_html += f"""
        <div class="card-item">
            <div class="card-content-container">
                <img src="{img_url}">
            </div>
        </div>
        """
        
    card_html += '</div>'
    st.markdown(card_html, unsafe_allow_html=True)
    
    # 注意: st.sidebar.selectbox("1列あたりのカード数", ...) は、カスタムCSSを使うことで意味がなくなるため削除しました。
    # 代わりに、カードは画面幅に応じて動的に横に並びます。

# ===============================
# 🧱 デッキ作成モード
# ... (サイドバーのロジックは変更なし)
# ===============================
else:
    st.title("🧱 デッキ作成モード")
    
    # サイドバー：デッキ情報
    # ... (省略: サイドバーの表示ロジックは変更なし)
    st.sidebar.markdown("---")
    st.sidebar.title("🧾 現在のデッキ")
    leader = st.session_state.get("leader")
    if leader is not None:
        st.sidebar.markdown(f"**リーダー:** {leader['カード名']} ({leader['カードID']})")
        deck_name_input = st.sidebar.text_input("デッキ名", value=st.session_state.get("deck_name", ""), key="deck_name_input")
        if deck_name_input != st.session_state.get("deck_name", ""):
            st.session_state["deck_name"] = deck_name_input
    total_cards = sum(st.session_state["deck"].values())
    st.sidebar.markdown(f"**合計カード:** {total_cards}/50")
    
    if st.session_state["deck"]:
        deck_cards = []
        for card_id, count in st.session_state["deck"].items():
            card_row = df[df["カードID"] == card_id].iloc[0]
            base_priority, type_rank, sub_priority, multi_flag = card_row["ソートキー"]
            deck_cards.append({
                "card_id": card_id,
                "count": count,
                "new_sort_key": (type_rank, card_row["コスト数値"], base_priority, card_id), 
                "cost": card_row["コスト数値"],
                "name": card_row["カード名"]
            })
        deck_cards.sort(key=lambda x: x["new_sort_key"])
        
        for card_info in deck_cards:
            col_name, col_add, col_del = st.sidebar.columns([4, 1, 1])
            current = st.session_state["deck"].get(card_info['card_id'], 0)
            is_unlimited = card_info['card_id'] in UNLIMITED_CARDS
            with col_name:
                st.markdown(f"**{card_info['name']}** x {card_info['count']} *<small>({card_info['card_id']})</small>*", unsafe_allow_html=True)
            with col_add:
                if st.button("＋", key=f"add_sidebar_{card_info['card_id']}", width='stretch', disabled=(not is_unlimited and current >= 4)):
                    if is_unlimited or current < 4:
                        st.session_state["deck"][card_info['card_id']] = current + 1
                        st.rerun()
            with col_del:
                if st.button("−", key=f"del_{card_info['card_id']}", width='stretch', disabled=current == 0):
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
                deck_cards_sorted.append({"card_id": card_id, "count": count, "new_sort_key": (type_rank, card_row["コスト数値"], base_priority, card_id)})
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
                st.sidebar.image(deck_img, caption="デッキ画像（QRコード付き）", use_container_width=True) 
                file_name = f"{deck_name}_deck.png" if deck_name else "deck_image.png"
                st.sidebar.download_button(label="📥 画像をダウンロード", data=buf, file_name=file_name, mime="image/png")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📥 デッキをインポート")
    st.sidebar.markdown("**QRコード画像からインポート**")
    uploaded_qr = st.sidebar.file_uploader("QRコード画像をアップロード", type=["png", "jpg", "jpeg"], key=f"qr_upload_{st.session_state['qr_upload_key']}")
    
    if uploaded_qr is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_qr.read()), dtype=np.uint8)
            qr_image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            detector = cv2.QRCodeDetector()
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
                deck_cards_sorted.append({"card_id": card_id, "count": count, "new_sort_key": (type_rank, card_row["コスト数値"], base_priority, card_id)})
            deck_cards_sorted.sort(key=lambda x: x["new_sort_key"])
            for card_info in deck_cards_sorted:
                save_lines.append(f"{card_info['count']}x{card_info['card_id']}")
            save_text = "\n".join(save_lines)
            path = os.path.join(SAVE_DIR, f"{current_deck_name}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(save_text)
            st.sidebar.success(f"デッキ「{current_deck_name}」を保存しました。")
    
    saved_files = [f[:-4] for f in os.listdir(SAVE_DIR) if f.endswith(".txt")]
    selected_load = st.sidebar.selectbox("📂 保存済みデッキを読み込み", ["選択なし"] + saved_files)
    
    if selected_load != "選択なし":
        if st.sidebar.button("📥 読み込む"):
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
    
    # メインエリア：リーダー選択 / デッキプレビュー / カード追加
    if st.session_state["deck_view"] == "leader" or st.session_state["leader"] is None:
        st.subheader("① リーダーを選択")
        leaders = df[df["タイプ"] == "LEADER"]
        leaders = leaders.sort_values(by=["ソートキー", "コスト数値", "カードID"], ascending=[True, True, True])
        
        # 💡 修正 3B: st.columns をやめ、HTML/CSS (card-list-container) で表示
        leader_html = '<div class="card-list-container">'
        for idx, (_, row) in enumerate(leaders.iterrows()):
            card_id = row['カードID']
            img_url = f"https://www.onepiece-cardgame.com/images/cardlist/card/{card_id}.png"
            
            # st.buttonをカスタムキーで表示し、HTMLのdiv内に埋め込む
            st_button_key = f"leader_{card_id}"
            st_button_placeholder = st.empty() # ボタンのプレースホルダー
            
            leader_html += f"""
            <div class="card-item">
                <div class="card-content-container">
                    <img src="{img_url}">
                    <p style="margin: 5px 0 0; font-size: 0.8rem;">{row["カード名"]}</p>
                    <div id="leader-btn-{card_id}"></div>
                </div>
            </div>
            """
        
        leader_html += '</div>'
        st.markdown(leader_html, unsafe_allow_html=True)

        # HTMLを先に描画した後、Pythonのボタンを所定のIDに挿入
        for idx, (_, row) in enumerate(leaders.iterrows()):
             card_id = row['カードID']
             with st.container():
                st.markdown(f'<div id="insert-leader-btn-{card_id}"></div>', unsafe_allow_html=True)
                if st.button(f"選択", key=f"leader_{card_id}"):
                    st.session_state["leader"] = row.to_dict()
                    st.session_state["deck"].clear()
                    st.session_state["deck_name"] = ""
                    st.session_state["deck_view"] = "preview"
                    st.rerun()
             # 挿入されたボタンを所定のIDのdivに移動させるためのJavaScriptを挿入（簡略化のため省略、このままだとボタンの位置がズレるが、Streamlitの制約上これが簡便）
             # 正確な配置には複雑なJavaScriptが必要になるため、ここでは見栄えよりも機能維持を優先

    
    elif st.session_state["deck_view"] == "preview":
        leader = st.session_state["leader"]
        
        st.subheader("🃏 デッキプレビュー")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            leader_img_url = f"https://www.onepiece-cardgame.com/images/cardlist/card/{leader['カードID']}.png"
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
            
            # 💡 修正 3C: st.columns をやめ、HTML/CSS (preview-card-list) で表示
            preview_html = '<div class="preview-card-list">'
            for card_info in deck_cards_sorted:
                card_img_url = f"https://www.onepiece-cardgame.com/images/cardlist/card/{card_info['card_id']}.png"
                
                preview_html += f"""
                <div class="preview-card-item">
                    <img src="{card_img_url}" style="width: 100%; height: auto; object-fit: contain;">
                    <p style="font-size: 0.7rem; margin: 0; white-space: nowrap;">{card_info['name']}</p>
                    <p style="font-size: 0.9rem; font-weight: bold; margin: 0;">× {card_info['count']}</p>
                </div>
                """
                         
            preview_html += '</div>'
            st.markdown(preview_html, unsafe_allow_html=True)
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
        current_filter = st.session_state["deck_filter"]

        col_a, col_b, col_c = st.columns(3)
        with col_a:
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
            
        col_d, col_e = st.columns([3, 1])
        with col_d:
            deck_free = st.text_input("フリーワード（カード名/特徴/テキスト/トリガー）", value=current_filter["free_words"], key="deck_free")
        with col_e:
            deck_blocks = st.multiselect("ブロックアイコン", sorted(df["ブロックアイコン"].unique()), default=current_filter["blocks"], key="deck_blocks")

        st.session_state["deck_filter"] = {
            "colors": [], "types": deck_types, "costs": deck_costs, "counters": deck_counters, 
            "attributes": deck_attributes, "blocks": deck_blocks, "features": deck_features, 
            "series_ids": deck_series_ids, "free_words": deck_free
        }
        
        st.session_state["deck_results"] = filter_cards(
            df, colors=[], types=deck_types, costs=deck_costs, counters=deck_counters, attributes=deck_attributes, 
            blocks=deck_blocks, feature_selected=deck_features, free_words=deck_free, 
            series_ids=deck_series_ids, leader_colors=leader_colors
        )
        
        color_cards = st.session_state["deck_results"]
        
        st.write(f"表示中のカード：{len(color_cards)} 枚")
        st.markdown("---")
        
        # 💡 修正 3D: st.columns をやめ、HTML/CSS (card-list-container) で表示
        card_add_html = '<div class="card-list-container">'
        
        # ボタンを配置するためのプレースホルダーリスト
        button_containers = []
        
        for idx, (_, card) in enumerate(color_cards.iterrows()):
            img_url = f"https://www.onepiece-cardgame.com/images/cardlist/card/{card['カードID']}.png"
            card_id = card["カードID"]
            current_count = st.session_state["deck"].get(card_id, 0)
            
            # ボタンを埋め込むためのカスタムIDを設定
            btn_container_id = f"btn_group_{card_id}_{idx}"
            
            card_add_html += f"""
            <div class="card-item">
                <div class="card-content-container">
                    <img src="{img_url}">
                    <p style="margin: 5px 0; font-size: 0.8rem;">({current_count}/4枚)</p>
                    <div id="{btn_container_id}" class="card-button-group"></div>
                </div>
            </div>
            """
            button_containers.append((card_id, idx, btn_container_id, current_count))
            
        card_add_html += '</div>'
        
        # HTMLを先に描画
        st.markdown(card_add_html, unsafe_allow_html=True)

        # PythonのボタンをHTMLの描画後に配置 (ボタンの位置がずれる場合がありますが、機能維持のため)
        for card_id, idx, btn_container_id, current_count in button_containers:
            is_unlimited = card_id in UNLIMITED_CARDS
            
            # Streamlitの列を使い、ボタンを横並びにする
            btn_col_a, btn_col_b = st.columns(2)
            
            with btn_col_a:
                # ユーザーがクリックしたときに st.rerun() が発生し、全てのボタンが再描画されます
                if st.button("＋", key=f"add_deck_{card_id}_{idx}", type="primary", width='stretch', disabled=(not is_unlimited and current_count >= 4)):
                    count = st.session_state["deck"].get(card_id, 0)
                    if is_unlimited or count < 4:
                        st.session_state["deck"][card_id] = count + 1
                        st.rerun()
            with btn_col_b:
                if st.button("−", key=f"sub_deck_{card_id}_{idx}", width='stretch', disabled=current_count == 0):
                    if card_id in st.session_state["deck"] and st.session_state["deck"][card_id] > 0:
                        if st.session_state["deck"][card_id] > 1:
                            st.session_state["deck"][card_id] -= 1
                        else:
                            del st.session_state["deck"][card_id]
                        st.rerun()
