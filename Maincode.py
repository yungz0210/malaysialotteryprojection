import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import itertools
from datetime import date

# ---------- CONFIG ----------
st.set_page_config(layout="wide", page_title="4D Analytics & Prediction")

# ---------- DATA LOADING ----------
columns = [
    "Year", "Month", "Day",
    "First Prize", "Second Prize", "Third Prize",
    "Starter Prize 1", "Starter Prize 2", "Starter Prize 3", "Starter Prize 4", "Starter Prize 5",
    "Starter Prize 6", "Starter Prize 7", "Starter Prize 8", "Starter Prize 9", "Starter Prize 10",
    "Consolation Prize 1", "Consolation Prize 2", "Consolation Prize 3", "Consolation Prize 4", "Consolation Prize 5",
    "Consolation Prize 6", "Consolation Prize 7", "Consolation Prize 8", "Consolation Prize 9", "Consolation Prize 10",
    "DrawDay"
]

urls = {
    "Magnum": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQSqTy2fRIQ3BFXqZ7l4n7jVJjNlrN2fHGPAjgyHPEBeCWH6HfENiLZCyTRepmZvqugM83OaUwr4vLy/pub?gid=408011920&single=true&output=csv",
    "Damacai": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQSqTy2fRIQ3BFXqZ7l4n7jVJjNlrN2fHGPAjgyHPEBeCWH6HfENiLZCyTRepmZvqugM83OaUwr4vLy/pub?gid=148251219&single=true&output=csv",
    "Toto": "https://docs.google.com/spreadsheets/d/e/2PACX-1vQSqTy2fRIQ3BFXqZ7l4n7jVJjNlrN2fHGPAjgyHPEBeCWH6HfENiLZCyTRepmZvqugM83OaUwr4vLy/pub?gid=420129454&single=true&output=csv"
}

PRIZE_COLS = columns[3:-1]

@st.cache_data(show_spinner=False)
def load_data():
    data = {}
    for name, url in urls.items():
        game_columns = [
            "Year", "Month", "Day", "First Prize", "Second Prize", "Third Prize",
            "Starter Prize 1", "Starter Prize 2", "Starter Prize 3", "Starter Prize 4", "Starter Prize 5",
            "Starter Prize 6", "Starter Prize 7", "Starter Prize 8", "Starter Prize 9", "Starter Prize 10",
            "Consolation Prize 1", "Consolation Prize 2", "Consolation Prize 3", "Consolation Prize 4", "Consolation Prize 5",
            "Consolation Prize 6", "Consolation Prize 7", "Consolation Prize 8", "Consolation Prize 9", "Consolation Prize 10",
            "DrawDay"
        ]
        try:
            df = pd.read_csv(url, names=game_columns, dtype=str, keep_default_na=False, on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(url, names=game_columns, dtype=str, keep_default_na=False, encoding="latin1", on_bad_lines="skip")

        # strip strings
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # normalize date parts
        for col in ["Year", "Month", "Day"]:
            if col in df.columns:
                df[col] = df[col].str.strip().replace("", np.nan)

        def safe_date(row):
            try:
                return pd.Timestamp(year=int(row["Year"]), month=int(row["Month"]), day=int(row["Day"]))
            except Exception:
                return pd.NaT

        df["draw_date"] = df.apply(safe_date, axis=1)

        # Normalize prize columns (keep leading zeros)
        PRIZE_COLS_GAME = [c for c in game_columns if c not in ["Year", "Month", "Day", "DrawDay"]]
        for c in PRIZE_COLS_GAME:
            if c in df.columns:
                df[c] = df[c].astype(str).replace("nan", "").apply(lambda x: x.zfill(4) if x.isdigit() else x)

        data[name] = df

    return data

data = load_data()

# ---------- UTIL ----------
def df_to_csv_bytes(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def get_game_latest_draw(df):
    if "draw_date" in df.columns and df["draw_date"].notna().any():
        mx = df["draw_date"].max()
        return mx.date().isoformat() if not pd.isna(mx) else "No valid date"
    return "No draw date"

# ---------- MODELING HELPERS ----------
def make_seq_features(seq_list, n_lag):
    X, y = [], []
    for i in range(n_lag, len(seq_list)):
        X.append(seq_list[i-n_lag:i])
        y.append(seq_list[i])
    return np.array(X), np.array(y)

def build_position_markov(sequences):
    # sequences: list[str] each 4 digits
    pos_trans = {pos: np.zeros((10, 10), dtype=float) for pos in range(4)}
    last_digits = {pos: None for pos in range(4)}
    for s in sequences:
        if len(s) != 4 or not s.isdigit():
            continue
        for pos in range(4):
            d = int(s[pos])
            if last_digits[pos] is not None:
                pos_trans[pos][last_digits[pos], d] += 1
            last_digits[pos] = d
    pos_probs = {}
    for pos in range(4):
        mat = pos_trans[pos].copy()
        mat += 1e-6  # smoothing
        row_sums = mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        pos_probs[pos] = mat / row_sums
    return pos_probs, last_digits, pos_trans

def build_full_chain_markov(sequences):
    init_counts = np.zeros(10, dtype=float)
    trans_counts = np.zeros((10, 10), dtype=float)
    for s in sequences:
        if len(s) != 4 or not s.isdigit():
            continue
        init_counts[int(s[0])] += 1
        for i in range(3):
            a = int(s[i]); b = int(s[i+1]); trans_counts[a, b] += 1
    init_counts += 1e-6
    trans_counts += 1e-6
    init_probs = init_counts / init_counts.sum()
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    trans_probs = trans_counts / row_sums
    return init_probs, trans_probs

def safe_log(x):
    x = float(x)
    return np.log(x) if x > 0 else np.log(1e-12)

# Session cache for trained models / probabilities
if "models_cache" not in st.session_state:
    st.session_state["models_cache"] = {}

# ---------- PAGE NAV ----------
page = st.sidebar.radio("Select Page", ["Analytics", "Number Lookup", "Prediction"])

# ---------- SIDEBAR FILTERS ----------
st.sidebar.title("Global Filters")

game_choice = st.sidebar.selectbox("Select Game", list(data.keys()), index=0)
df_game = data[game_choice]

# Year slider
years = []
for v in df_game["Year"].unique():
    try:
        years.append(int(v))
    except Exception:
        pass
min_y = min(years) if years else 1980
max_y = max(years) if years else date.today().year
year_range = st.sidebar.slider("Year range", int(min_y), int(max_y), (int(min_y), int(max_y)))

# Date range
min_date = df_game["draw_date"].min()
max_date = df_game["draw_date"].max()
if pd.isna(min_date):
    min_date = date(1980, 1, 1)
else:
    min_date = min_date.date()
if pd.isna(max_date):
    max_date = date.today()
else:
    max_date = max_date.date()
date_start, date_end = st.sidebar.date_input("Date range", value=(min_date, max_date))

# Draw day and prize types (MULTISELECT — used everywhere below)
raw_drawdays = [d for d in df_game["DrawDay"].unique() if d and str(d).strip() != ""]
drawday_choice = st.sidebar.multiselect("Draw Day", sorted(raw_drawdays), default=sorted(raw_drawdays))

prize_options = list(PRIZE_COLS)
prize_types = st.sidebar.multiselect(
    "Prize Types",
    options=prize_options,
    default=prize_options
)

# ---------- FILTER FUNCTION ----------
def filter_df(df, year_range=None, date_range=None, drawdays=None, prize_types=None):
    out = df.copy()

    # Year range filter with robust handling of NaNs and non-numeric years
    if year_range is not None:
        y0, y1 = year_range
        def is_valid_year(x):
            try:
                return int(x)
            except Exception:
                return None
        out["Year_int"] = out["Year"].apply(is_valid_year)
        out = out[out["Year_int"].notna() & out["Year_int"].between(int(y0), int(y1))]
        out = out.drop(columns=["Year_int"])

    # Date range filter
    if date_range is not None:
        s, e = date_range
        s_ts = pd.to_datetime(s)
        e_ts = pd.to_datetime(e)
        out = out[(out["draw_date"].notna()) & (out["draw_date"] >= s_ts) & (out["draw_date"] <= e_ts)]

    # DrawDay filter
    if drawdays and len(drawdays) > 0:
        out = out[out["DrawDay"].isin(drawdays)]

    # Prize type filter: keep rows where at least one selected prize is nonempty
    if prize_types and len(prize_types) > 0:
        mask = pd.Series(False, index=out.index)
        for pt in prize_types:
            if pt in out.columns:
                mask = mask | (out[pt].notna() & (out[pt].str.strip() != ""))
        out = out[mask]

    return out

# Helper: collect numbers as strings using selected prize types
def collect_numbers(df_src, selected_prize_types):
    if not selected_prize_types:
        return []
    # Only keep columns that exist
    cols = [c for c in selected_prize_types if c in df_src.columns]
    if len(cols) == 0:
        return []
    df_numbers = df_src[cols].fillna("").astype(str)
    nums = df_numbers.values.flatten().tolist()
    nums = [x for x in nums if x and x.strip() != ""]
    # Keep leading zeros for numeric entries
    nums = [x.zfill(4) if x.isdigit() else x for x in nums]
    return nums

# ---------- ANALYTICS ----------
if page == "Analytics":
    st.header("Analytical Dashboard")
    st.subheader("Latest available draw per game")
    for gname, gdf in data.items():
        st.write(f"{gname}: {get_game_latest_draw(gdf)}")

    filtered = filter_df(
        df_game,
        year_range=year_range,
        date_range=(date_start, date_end),
        drawdays=drawday_choice,
        prize_types=prize_types
    )

    st.markdown("### Filter summary")
    st.write(
        f"Game **{game_choice}** | Years **{year_range[0]}–{year_range[1]}** | "
        f"Dates **{date_start} → {date_end}** | Draw Days **{', '.join(map(str, drawday_choice)) if drawday_choice else 'All'}** | "
        f"Prizes **{', '.join(prize_types) if prize_types else '(none)'}**"
    )

    # Collect numbers from selected prize types
    nums = collect_numbers(filtered, prize_types)

    if not nums:
        st.warning("No numbers found for the chosen filters.")
    else:
        s = pd.Series(nums)
        freq = s.value_counts().reset_index()
        freq.columns = ["Number", "Frequency"]
        top10 = freq.head(10)
        bottom10 = freq.tail(10)

        st.subheader("Frequency Distribution (Top 50)")
        fig, ax = plt.subplots(figsize=(10, 4))
        freq.head(50).plot(kind="bar", x="Number", y="Frequency", ax=ax)
        ax.set_xlabel("Number")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        st.subheader("Hot Numbers (Top 10)")
        st.write(top10)

        st.subheader("Cold Numbers (Bottom 10)")
        st.write(bottom10)

    st.markdown("---")
    st.subheader("Export filtered data")
    csv_bytes = df_to_csv_bytes(filtered)
    st.download_button(
        "Download filtered rows as CSV",
        data=csv_bytes,
        file_name=f"{game_choice}_filtered.csv",
        mime="text/csv"
    )

# ---------- NUMBER LOOKUP ----------
elif page == "Number Lookup":
    st.header("Number Lookup")

    lookup_num = st.text_input("Enter 4-digit number", value="")
    if lookup_num and lookup_num.isdigit():
        lookup_num = lookup_num.zfill(4)
    elif lookup_num != "":
        st.warning("Please enter digits only")

    lookup_game = st.selectbox(
        "Lookup in game",
        list(data.keys()),
        index=list(data.keys()).index(game_choice)
    )
    df_lookup = data[lookup_game]

    # For lookup we search across ALL prize columns (but still apply year/date/drawday filters)
    filtered_lookup = filter_df(
        df_lookup,
        year_range=year_range,
        date_range=(date_start, date_end),
        drawdays=drawday_choice,
        prize_types=PRIZE_COLS  # keep rows that had at least one prize (any)
    )

    if not lookup_num:
        st.info("Enter a number to see stats")
    else:
        hits = []
        for idx, row in filtered_lookup.iterrows():
            for c in PRIZE_COLS:
                v = row.get(c, "")
                if isinstance(v, str) and v.zfill(4) == lookup_num:
                    hits.append({
                        "draw_date": row.get("draw_date"),
                        "prize": c,
                        "DrawDay": row.get("DrawDay"),
                        "row_index": idx
                    })

        if not hits:
            st.write(f"{lookup_num} not found in filtered dataset for {lookup_game}")
        else:
            hits_df = pd.DataFrame(hits).sort_values("draw_date", ascending=False)
            st.subheader(f"Found {len(hits)} occurrences for {lookup_num} in {lookup_game}")
            st.write(hits_df[["draw_date", "prize", "DrawDay"]])

            last_seen = hits_df["draw_date"].max()
            st.write(f"Last seen on {last_seen.date().isoformat() if not pd.isna(last_seen) else 'unknown'}")

            freq_by_prize = hits_df["prize"].value_counts().reset_index()
            freq_by_prize.columns = ["Prize", "Count"]
            st.subheader("Frequency by prize type")
            st.write(freq_by_prize)

            drawday_counts = hits_df["DrawDay"].value_counts().reset_index()
            drawday_counts.columns = ["DrawDay", "Count"]
            st.subheader("Draw day distribution for this number")
            st.write(drawday_counts)

            dates = pd.to_datetime(hits_df["draw_date"].dropna().sort_values())
            if len(dates) >= 2:
                diffs = dates.diff().dt.days.dropna()
                st.subheader("Gaps between occurrences in days")
                st.write(f"mean {diffs.mean():.1f} | median {diffs.median():.1f} | max {diffs.max()} | min {diffs.min()}")
            else:
                st.info("Not enough dated occurrences to compute gaps")

            csv_bytes_hits = df_to_csv_bytes(hits_df)
            st.download_button(
                "Download occurrences CSV",
                data=csv_bytes_hits,
                file_name=f"{lookup_game}_{lookup_num}_hits.csv",
                mime="text/csv"
            )

# ---------- PREDICTION ----------
elif page == "Prediction":
    st.header("Prediction Models")

    # Apply global filters including prize_types
    filtered_pred = filter_df(
        df_game,
        year_range=year_range,
        date_range=(date_start, date_end),
        drawdays=drawday_choice,
        prize_types=prize_types
    )
    st.markdown(f"Using **{len(filtered_pred)}** rows after filters for predictions")

    if len(filtered_pred) == 0:
        st.warning("No rows available after applying filters")
        st.stop()
    if not prize_types:
        st.warning("No prize types selected")
        st.stop()

    model_choice = st.selectbox(
        "Choose Prediction Method",
        ["Frequency Analysis", "Digit Distribution", "Markov Chains", "Last Digit Markov", "Machine Learning", "Hybrid"]
    )

    # Collect numbers from selected prize types only
    all_nums = collect_numbers(filtered_pred, prize_types)
    if not all_nums:
        st.warning("No numbers available for prediction after applying filters")
        st.stop()

    # ---------- Prediction Methods ----------
    if model_choice == "Frequency Analysis":
        st.write("Predicting based on most frequent numbers (selected prize types)")
        freq_all = pd.Series(all_nums).value_counts().reset_index()
        freq_all.columns = ["Number", "Frequency"]
        freq_all["Probability"] = freq_all["Frequency"] / freq_all["Frequency"].sum()
        top_n = st.slider("Select how many numbers to predict", 5, 50, 10)
        st.subheader("Predicted Numbers by Frequency")
        st.write(freq_all.head(top_n))

    elif model_choice == "Digit Distribution":
        st.write("Predicting using digit distribution (selected prize types)")

        valid_numbers = [n for n in all_nums if n.isdigit() and len(n) == 4]
        if not valid_numbers:
            st.warning("No valid 4-digit numbers for digit distribution")
        else:
            digits = np.array([[int(d) for d in num] for num in valid_numbers])

            # Per-position frequencies
            digit_freq = {}
            for pos in range(4):
                counts = pd.Series(digits[:, pos]).value_counts().sort_index()
                digit_freq[pos] = counts.reindex(range(10), fill_value=0)

            st.subheader("Digit Frequencies per Position")
            for pos, counts in digit_freq.items():
                st.write(f"Position {pos+1} (left → right)")
                st.bar_chart(counts)

            # Extended controls
            st.markdown("**Generation controls**")
            top_per_pos = st.slider("How many top digits per position to combine", 2, 10, 3)
            top_n = st.slider("How many candidate numbers to output", 5, 100, 20)
            dedup_existing = st.checkbox("Remove numbers already seen in filtered data", value=False)

            # Build candidate grid from top-K per position, score by product of frequencies
            top_choices = [digit_freq[pos].nlargest(top_per_pos) for pos in range(4)]
            top_indices = [s.index.tolist() for s in top_choices]
            top_weights = [s.values for s in top_choices]

            candidates = []
            scores = []
            for a_idx, a_w in zip(top_indices[0], top_weights[0]):
                for b_idx, b_w in zip(top_indices[1], top_weights[1]):
                    for c_idx, c_w in zip(top_indices[2], top_weights[2]):
                        for d_idx, d_w in zip(top_indices[3], top_weights[3]):
                            num = f"{a_idx}{b_idx}{c_idx}{d_idx}"
                            score = (a_w + 1e-9) * (b_w + 1e-9) * (c_w + 1e-9) * (d_w + 1e-9)
                            candidates.append(num)
                            scores.append(score)

            cand_df = pd.DataFrame({"Number": candidates, "Score": scores}).sort_values("Score", ascending=False)

            if dedup_existing:
                seen = set(valid_numbers)
                cand_df = cand_df[~cand_df["Number"].isin(seen)]

            st.subheader("Predicted Numbers by Digit Distribution")
            st.write(cand_df.head(top_n).reset_index(drop=True))

    elif model_choice == "Markov Chains":
        st.write("Full-number Markov chain across digits inside number (selected prize types)")

        valid_numbers = [n for n in all_nums if n.isdigit() and len(n) == 4]
        if not valid_numbers:
            st.warning("No valid 4-digit numbers for Markov Chain")
        else:
            init_probs, trans_probs = build_full_chain_markov(valid_numbers)

            st.subheader("Transition heatmap (within-number)")
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(trans_probs, aspect="auto")
            ax.set_xlabel("Next digit")
            ax.set_ylabel("Current digit")
            ax.set_xticks(range(10))
            ax.set_yticks(range(10))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)

            # Extended controls
            top_n = st.slider("How many numbers to return", 5, 100, 20)
            num_samples = st.slider("Internal sample size", 100, 10000, 2000, step=100)
            seed = st.number_input("Random seed", value=42, step=1)
            ensure_unique = st.checkbox("Ensure unique output numbers", value=True)

            rng = np.random.default_rng(seed=int(seed))
            scores = {}

            # Monte-Carlo sample from Markov chain; keep max log-prob per sequence
            for _ in range(int(num_samples)):
                d0 = rng.choice(10, p=init_probs)
                d1 = rng.choice(10, p=trans_probs[d0])
                d2 = rng.choice(10, p=trans_probs[d1])
                d3 = rng.choice(10, p=trans_probs[d2])
                s = f"{d0}{d1}{d2}{d3}"
                lp = (
                    safe_log(init_probs[d0]) +
                    safe_log(trans_probs[d0, d1]) +
                    safe_log(trans_probs[d1, d2]) +
                    safe_log(trans_probs[d2, d3])
                )
                if s not in scores or lp > scores[s]:
                    scores[s] = lp

            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            out_numbers = [n for (n, _) in ranked]
            out_numbers = out_numbers[:top_n]
            out = pd.DataFrame({
                "Number": out_numbers,
                "ChainScore": [float(np.exp(scores[n])) for n in out_numbers]
            })
            st.subheader("Predicted Numbers by Markov Chain")
            st.write(out)

    elif model_choice == "Last Digit Markov":
        st.write("Last-digit Markov across draws (selected prize types)")

        valid_numbers = [n for n in all_nums if n.isdigit() and len(n) == 4]
        if not valid_numbers:
            st.warning("No valid 4-digit numbers for last-digit Markov")
        else:
            last_digits = [int(n[-1]) for n in valid_numbers]
            trans = np.zeros((10, 10), dtype=float)
            for i in range(len(last_digits) - 1):
                a = last_digits[i]
                b = last_digits[i + 1]
                trans[a, b] += 1
            trans += 1e-6
            row_sums = trans.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            trans_probs = trans / row_sums

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(trans_probs, aspect="auto")
            ax.set_xlabel("Next digit")
            ax.set_ylabel("Current digit")
            ax.set_xticks(range(10))
            ax.set_yticks(range(10))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)

            current_last = last_digits[-1]
            st.write(f"Most recent last digit observed: **{current_last}**")

            top_k = st.slider("How many last-digit candidates to show", 1, 10, 5, key="last_k")
            probs = trans_probs[current_last]
            top_idxs = probs.argsort()[-top_k:][::-1]
            st.subheader("Next last-digit candidates")
            st.write(pd.DataFrame({"Digit": top_idxs, "Probability": probs[top_idxs]}))

    # ---------- MACHINE LEARNING ----------
    elif model_choice == "Machine Learning":
        st.write("Machine learning per-position models using filtered data (selected prize types)")

        # Prepare training sequences from filtered_pred
        df_train = filtered_pred[[c for c in prize_types if c in filtered_pred.columns]].copy()
        # Keep only valid 4-digit numbers
        df_train = df_train.applymap(lambda x: x.zfill(4) if str(x).isdigit() else None)
        seqs = [x for x in df_train.values.flatten() if isinstance(x, str) and x.isdigit() and len(x) == 4]

        if len(seqs) < 10:
            st.warning("Very small dataset; models may not train well. Prefer ≥ 30 draws for reliability.")

        ml_method = st.selectbox("ML method", ["RandomForest", "XGBoost", "LSTM"])
        n_lag = st.slider("Lag (previous draws) for per-position models", 1, 12, 5)
        lstm_epochs = st.slider("LSTM epochs", 1, 50, 8)
        lstm_batch = st.slider("LSTM batch size", 8, 128, 32)

        cache_key = (game_choice, str(date_start), str(date_end), tuple(sorted(drawday_choice)), tuple(sorted(prize_types)), ml_method, n_lag, lstm_epochs, lstm_batch)

        if cache_key in st.session_state["models_cache"]:
            st.info("Using cached models for these filters & hyperparams")
            models_info = st.session_state["models_cache"][cache_key]
        else:
            models_info = {"pos_models": {}, "pos_probs": {}, "metrics": {}}
            for pos in range(4):
                pos_seq = [int(s[pos]) for s in seqs]
                X, y = make_seq_features(pos_seq, n_lag)
                if len(X) < 5:
                    models_info["pos_models"][pos] = None
                    models_info["pos_probs"][pos] = np.ones(10) / 10
                    models_info["metrics"][pos] = {"note": "insufficient data"}
                    continue

                if ml_method in ("RandomForest", "XGBoost"):
                    try:
                        from sklearn.model_selection import train_test_split
                        from sklearn.metrics import accuracy_score
                        Xf = X.reshape(X.shape[0], -1)
                        if ml_method == "RandomForest":
                            from sklearn.ensemble import RandomForestClassifier
                            clf = RandomForestClassifier(n_estimators=120, random_state=42)
                        else:
                            import xgboost as xgb
                            clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
                        X_train, X_test, y_train, y_test = train_test_split(Xf, y, test_size=0.2, random_state=42)
                        clf.fit(X_train, y_train)
                        preds = clf.predict(X_test)
                        test_acc = float(accuracy_score(y_test, preds))
                        last_window = np.array(pos_seq[-n_lag:]).reshape(1, -1)
                        proba = clf.predict_proba(last_window)[0]
                        arr = np.zeros(10)
                        for cls_idx, p in zip(clf.classes_, proba):
                            arr[int(cls_idx)] = p
                        models_info["pos_models"][pos] = clf
                        models_info["pos_probs"][pos] = arr
                        models_info["metrics"][pos] = {"test_acc": test_acc}
                    except Exception as e:
                        models_info["pos_models"][pos] = None
                        models_info["pos_probs"][pos] = np.ones(10) / 10
                        models_info["metrics"][pos] = {"error": str(e)}
                else:
                    # LSTM
                    try:
                        import tensorflow as tf
                        from tensorflow import keras
                        Xs = X.astype(float) / 9.0
                        Xs = Xs.reshape((Xs.shape[0], Xs.shape[1], 1))
                        split = int(0.8 * len(Xs))
                        X_tr, X_val = Xs[:split], Xs[split:]
                        y_tr, y_val = y[:split], y[split:]
                        model = keras.Sequential([
                            keras.layers.Input(shape=(n_lag, 1)),
                            keras.layers.LSTM(64, return_sequences=False),
                            keras.layers.Dropout(0.2),
                            keras.layers.Dense(32, activation="relu"),
                            keras.layers.Dense(10, activation="softmax")
                        ])
                        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
                        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
                        history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=lstm_epochs, batch_size=lstm_batch, verbose=0, callbacks=[es])
                        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
                        last_window = np.array(pos_seq[-n_lag:]).astype(float) / 9.0
                        last_window = last_window.reshape(1, n_lag, 1)
                        probs = model.predict(last_window, verbose=0)[0]
                        models_info["pos_models"][pos] = model
                        models_info["pos_probs"][pos] = probs
                        models_info["metrics"][pos] = {"val_acc": float(val_acc), "epochs_trained": len(history.history["loss"])}
                    except Exception as e:
                        models_info["pos_models"][pos] = None
                        models_info["pos_probs"][pos] = np.ones(10) / 10
                        models_info["metrics"][pos] = {"error": str(e)}
            st.session_state["models_cache"][cache_key] = models_info
            st.success("Trained per-position models (cached in session)")

        # Display top digits per position
        models_info = st.session_state["models_cache"][cache_key]
        top_digits_per_pos = []
        for pos in range(4):
            probs = models_info["pos_probs"].get(pos, np.ones(10) / 10)
            top_idxs = probs.argsort()[-4:][::-1]
            top_digits_per_pos.append(top_idxs)
            st.write(f"Position {pos+1} top 4 digits")
            st.write(pd.DataFrame({"Digit": top_idxs, "Probability": probs[top_idxs]}))

        candidates = ["".join(map(str, comb)) for comb in itertools.product(*top_digits_per_pos)]
        st.subheader("Candidate numbers from top digits per position")
        st.write(candidates[:50])

    # ---------- HYBRID ----------
    elif model_choice == "Hybrid":
        st.write("Hybrid scoring: Frequency + Full-chain Markov + Position Markov + ML per-position (selected prize types)")

        nums = collect_numbers(filtered_pred, prize_types)
        valid_numbers = [n for n in nums if n and str(n).isdigit() and len(str(n)) == 4]

        if not valid_numbers:
            st.warning("No valid 4-digit numbers after applying filters")
            st.stop()

        # Frequency
        freq = pd.Series(valid_numbers).value_counts()
        total = freq.sum()

        # Full-chain Markov
        init_probs, trans_probs = build_full_chain_markov(valid_numbers)

        # Position Markov
        pos_probs, last_digits_pos, pos_trans = build_position_markov(valid_numbers)

        # ML per-position (quick training)
        ml_method = st.selectbox("ML method for hybrid", ["RandomForest", "XGBoost", "LSTM"])
        n_lag = st.slider("Lag for ML per-position", 1, 12, 5, key="hybrid_nlag")
        lstm_epochs_h = st.slider("LSTM epochs (hybrid)", 1, 20, 3, key="hybrid_lstm_epochs")
        lstm_batch_h = st.slider("LSTM batch size (hybrid)", 8, 64, 16, key="hybrid_lstm_batch")

        cache_key = ("hybrid", game_choice, str(date_start), str(date_end), tuple(sorted(drawday_choice)), tuple(sorted(prize_types)), ml_method, n_lag, lstm_epochs_h, lstm_batch_h)
        if cache_key in st.session_state["models_cache"]:
            models_info = st.session_state["models_cache"][cache_key]
        else:
            models_info = {"pos_probs": {}}
            seqs_train = [s for s in valid_numbers if s.isdigit() and len(s) == 4]
            for pos in range(4):
                pos_seq = [int(s[pos]) for s in seqs_train]
                X, y = make_seq_features(pos_seq, n_lag)
                if len(X) < 5:
                    models_info["pos_probs"][pos] = np.ones(10) / 10
                    continue
                if ml_method in ("RandomForest", "XGBoost"):
                    try:
                        Xf = X.reshape(X.shape[0], -1)
                        if ml_method == "RandomForest":
                            from sklearn.ensemble import RandomForestClassifier
                            clf = RandomForestClassifier(n_estimators=80, random_state=42)
                            clf.fit(Xf, y)
                        else:
                            import xgboost as xgb
                            clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
                            clf.fit(Xf, y)
                        last_window = np.array(pos_seq[-n_lag:]).reshape(1, -1)
                        proba = clf.predict_proba(last_window)[0]
                        arr = np.zeros(10)
                        for cls_idx, p in zip(clf.classes_, proba):
                            arr[int(cls_idx)] = p
                        models_info["pos_probs"][pos] = arr
                    except Exception:
                        models_info["pos_probs"][pos] = np.ones(10) / 10
                else:
                    try:
                        import tensorflow as tf
                        from tensorflow import keras
                        Xs = X.astype(float) / 9.0
                        Xs = Xs.reshape((Xs.shape[0], Xs.shape[1], 1))
                        split = int(0.8 * len(Xs))
                        model = keras.Sequential([keras.layers.Input(shape=(n_lag,1)), keras.layers.LSTM(32), keras.layers.Dense(10, activation='softmax')])
                        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
                        es = keras.callbacks.EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
                        model.fit(Xs[:split], y[:split], epochs=lstm_epochs_h, batch_size=lstm_batch_h, verbose=0, callbacks=[es])
                        last_window = np.array(pos_seq[-n_lag:]).astype(float) / 9.0
                        last_window = last_window.reshape(1, n_lag, 1)
                        probs = model.predict(last_window, verbose=0)[0]
                        models_info["pos_probs"][pos] = probs
                    except Exception:
                        models_info["pos_probs"][pos] = np.ones(10) / 10
            st.session_state["models_cache"][cache_key] = models_info

        # Candidate pool & weights
        candidate_pool = list(freq.head(500).index) if len(freq) > 0 else valid_numbers
        w_freq = st.slider("Weight Frequency", 0.0, 1.0, 0.25)
        w_full = st.slider("Weight Full-chain Markov", 0.0, 1.0, 0.25)
        w_pos = st.slider("Weight Position Markov", 0.0, 1.0, 0.25)
        w_ml = st.slider("Weight ML per-position", 0.0, 1.0, 0.25)
        wsum = w_freq + w_full + w_pos + w_ml

        if wsum == 0:
            st.warning("At least one weight must be > 0")
        else:
            rows = []
            pos_probs_cache = st.session_state["models_cache"][cache_key]["pos_probs"]

            for num in candidate_pool:
                digits = [int(d) for d in num]
                # Frequency component
                p_freq = (freq.get(num, 0) / total) if total > 0 else 1e-12
                # Full-chain Markov component
                p_full = (
                    init_probs[digits[0]] *
                    trans_probs[digits[0], digits[1]] *
                    trans_probs[digits[1], digits[2]] *
                    trans_probs[digits[2], digits[3]]
                )
                # Position Markov component
                p_pos_comp = 1.0
                for pos in range(4):
                    last = last_digits_pos.get(pos, None) if 'last_digits_pos' in locals() else None
                    if last is None:
                        # fallback uniform
                        p_pos_comp *= 1.0 / 10.0
                    else:
                        row = pos_trans[pos]
                        # if last not observed (possible), use uniform
                        try:
                            denom = row[last].sum()
                            p_pos_comp *= (row[last, digits[pos]] / denom) if denom > 0 else 1.0 / 10.0
                        except Exception:
                            p_pos_comp *= 1.0 / 10.0

                # ML per-position component
                p_ml = 1.0
                for pos in range(4):
                    posp = pos_probs_cache.get(pos, np.ones(10) / 10)
                    p_ml *= float(posp[digits[pos]]) if posp is not None else 1.0 / 10.0

                log_score = (
                    (w_freq / wsum) * safe_log(max(p_freq, 1e-12)) +
                    (w_full / wsum) * safe_log(max(p_full, 1e-12)) +
                    (w_pos / wsum) * safe_log(max(p_pos_comp, 1e-12)) +
                    (w_ml / wsum) * safe_log(max(p_ml, 1e-12))
                )
                rows.append({
                    "Number": num,
                    "P_freq": p_freq,
                    "P_full": p_full,
                    "P_pos": p_pos_comp,
                    "P_ml": p_ml,
                    "HybridScore": float(np.exp(log_score))
                })

            out = pd.DataFrame(rows).sort_values("HybridScore", ascending=False).head(50)
            st.subheader("Hybrid ranking (top candidates)")
            st.write(out)

    # Export dataset used for predictions (whatever filters applied)
    st.markdown("---")
    st.subheader("Export data used for prediction")
    csv_bytes = df_to_csv_bytes(filtered_pred)
    st.download_button(
        "Download prediction dataset as CSV",
        data=csv_bytes,
        file_name=f"{game_choice}_prediction_dataset.csv",
        mime="text/csv"
    )
