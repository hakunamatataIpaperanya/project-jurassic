# Task-1 specimen features adopted after notebook 15 (used by notebooks 11 and 12):
#   known-site genus model : TF-IDF char n-grams of the specimen text -> 32-d SVD          (tfidf_0..31)
#   new-site genus model   : WangchanBERTa (Thai BERT) mean-pooled embedding -> 32-d PCA   (wangchanberta_0..31)
#                            (rule agreed with the user: BERT first, multilingual MiniLM if BERT is not better —
#                             notebook 15 section 5: BERT better on accuracy and top-3, so BERT is used)
#                            + the site's formation (notebook 07 table, "Impute only") + ThaiLex age / lithology (fm_*)
# Text = "com_name_generic | f_type | f_part" (com_name_generic, not com_name, which leaks the species name).
TXT_DIM = 32
_TL_PERIOD = {"Neogene": "Tertiary", "Paleogene": "Tertiary", "Neoproterozoic": "Precambrian"}


_GENUS_RE = {}


def genus_name_pattern():
    # ชื่อสกุลทุกชื่อที่พบในทะเบียน (+ ชื่อชนิด / sp. / cf. ที่ตามมา): f_part บางแถวเขียนชื่อวิทยาศาสตร์ไว้ (เช่น "Siamosaurus teeth",
    # "Spirifer sp.") ถ้าปล่อยไว้ TF-IDF / embedding จะเห็นคำตอบ — ลบออกก่อนสร้าง feature ข้อความ (ใช้รายชื่อเพื่อ "ลบ" เท่านั้น ไม่ได้เพิ่มข้อมูล)
    if "re" not in _GENUS_RE:
        s = pd.read_csv(DATA / "processed" / "fossil_list_final.csv", usecols=["sci_name"], encoding="utf-8-sig", low_memory=False)
        gen = sorted({t[0] for t in s.sci_name.map(parse_taxon) if t}, key=len, reverse=True)
        keep = r"(?!(?:teeth|tooth|bone|bones|shell|skull|jaw|vertebra|vertebrae|neural|spine|fragment|fragments|and|or|with)\b)"
        _GENUS_RE["re"] = re.compile(r"\b(?:" + "|".join(map(re.escape, gen)) + r")\b(?:\s+(?:cf\.|aff\.)?\s*(?:sp\.|spp\.|" + keep + r"[a-z][a-z\-]{2,}))?")
    return _GENUS_RE["re"]


def specimen_text(df):
    # pandas 3: astype(str) คงค่าว่างเป็น NaN -> เติม "nan" ก่อน (ต้องเหมือนกันทุกที่); ลบชื่อสกุลที่เขียนไว้ใน f_part
    part = df.f_part.fillna("nan").astype(str).str.replace(genus_name_pattern(), " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    return df.com_name_generic.fillna("nan").astype(str) + " | " + df.f_type.fillna("nan").astype(str) + " | " + part


class TextTfidf:
    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True).fit(texts)
        self.svd_ = TruncatedSVD(TXT_DIM, random_state=0).fit(self.vec_.transform(texts))
        return self

    def transform(self, texts):
        Z = self.svd_.transform(self.vec_.transform(list(texts)))
        return pd.DataFrame(Z, columns=[f"tfidf_{i}" for i in range(TXT_DIM)])


_EMB = {}
EMB_MODELS = {"wangchanberta": "airesearch/wangchanberta-base-att-spm-uncased",
              "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"}


def encode_texts(texts, kind="multilingual", batch=64):
    # embedding ของข้อความ (ไม่ผ่าน cache): "wangchanberta" = mean pooling ของ WangchanBERTa,
    # "multilingual" = sentence-transformers MiniLM (เหมือน notebook 15)
    import torch
    st = _EMB.setdefault(kind, {})
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    texts = list(texts)
    if kind == "wangchanberta":
        if "model" not in st:
            from transformers import AutoModel, AutoTokenizer
            st["tok"] = AutoTokenizer.from_pretrained(EMB_MODELS[kind])
            st["model"] = AutoModel.from_pretrained(EMB_MODELS[kind]).to(dev).eval()
        tok, mdl = st["tok"], st["model"]
        out = []
        for i in range(0, len(texts), batch):
            b = tok(texts[i:i + batch], padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
            with torch.no_grad():
                h = mdl(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
        return np.vstack(out).astype("float32")
    if "model" not in st:
        from sentence_transformers import SentenceTransformer
        st["model"] = SentenceTransformer(EMB_MODELS[kind], device=dev)
    return st["model"].encode(texts, batch_size=batch, show_progress_bar=False, convert_to_numpy=True).astype("float32")


# WangchanBERTa แบบอื่น — ใช้ทดลองเท่านั้น (notebook 15 หัวข้อ 10); ตัวที่ใช้จริง = "wangchanberta" (mean pooling, ช่องว่างปกติ)
#   kind -> (วิธีรวมเวกเตอร์, แทนช่องว่างด้วยโทเค็น "<_>" ตามที่ WangchanBERTa ใช้ตอน pretrain หรือไม่)
POOLINGS = {"wangchanberta_cls": ("cls", False), "wangchanberta_max": ("max", False),
            "wangchanberta_sp": ("mean", True), "wangchanberta_sp_cls": ("cls", True), "wangchanberta_sp_max": ("max", True)}


def encode_bert_pooled(texts, pooling, space_token=False, batch=64):
    # "mean" = เฉลี่ยโทเค็นจริง (เหมือน encode_texts), "cls" = เวกเตอร์ของโทเค็นแรก <s>, "max" = ค่าสูงสุดรายมิติของโทเค็นจริง
    # space_token=True: แทน " " ด้วย "<_>" ก่อนตัดโทเค็น — โมเดล / tokenizer / ความยาวสูงสุดเท่ากับ encode_texts
    import torch
    st = _EMB.setdefault("wangchanberta", {})
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if "model" not in st:
        from transformers import AutoModel, AutoTokenizer
        st["tok"] = AutoTokenizer.from_pretrained(EMB_MODELS["wangchanberta"])
        st["model"] = AutoModel.from_pretrained(EMB_MODELS["wangchanberta"]).to(dev).eval()
    tok, mdl, out = st["tok"], st["model"], []
    texts = [t.replace(" ", "<_>") for t in texts] if space_token else list(texts)
    for i in range(0, len(texts), batch):
        b = tok(texts[i:i + batch], padding=True, truncation=True, max_length=64, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            h = mdl(**b).last_hidden_state
        keep = b["attention_mask"].unsqueeze(-1)
        if pooling == "cls":
            out.append(h[:, 0].cpu().numpy())
        elif pooling == "max":
            out.append(h.masked_fill(keep == 0, float("-inf")).max(1).values.cpu().numpy())
        else:
            out.append(((h * keep).sum(1) / keep.sum(1)).cpu().numpy())
    return np.vstack(out).astype("float32")


def token_spread(texts, space_token=False, batch=64):
    # ความต่างของเวกเตอร์โทเค็นภายในข้อความ (SD รายมิติเฉลี่ย ของชั้นสุดท้าย): ใกล้ 0 = ทุกโทเค็นได้เวกเตอร์เกือบเท่ากัน ("ยุบ")
    import torch
    encode_bert_pooled(texts[:1], "mean")                       # โหลดโมเดล
    tok, mdl, out = _EMB["wangchanberta"]["tok"], _EMB["wangchanberta"]["model"], []
    texts = [t.replace(" ", "<_>") for t in texts] if space_token else list(texts)
    for i in range(0, len(texts), batch):
        b = tok(texts[i:i + batch], padding=True, truncation=True, max_length=64, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            h = mdl(**b).last_hidden_state
        m = b["attention_mask"].bool()
        out += [float(h[j][m[j]].std(0).mean()) for j in range(h.shape[0])]
    return np.array(out)


def text_embed(texts, kind="multilingual"):
    # encode_texts + cache ต่อข้อความใน data/features/text_emb/<kind>.parquet — ดึงด้วยตัวข้อความ ไม่ใช่ตำแหน่งแถว
    cache_f = FEAT / "text_emb" / f"{kind}.parquet"
    st = _EMB.setdefault(kind, {})
    if "cache" not in st:
        st["cache"] = pd.read_parquet(cache_f).set_index("text") if cache_f.exists() else pd.DataFrame()
    cache = st["cache"]
    texts = list(texts)
    todo = sorted(set(texts) - set(cache.index))
    if todo:
        emb = encode_bert_pooled(todo, *POOLINGS[kind]) if kind in POOLINGS else encode_texts(todo, kind)
        new = pd.DataFrame(emb, index=pd.Index(todo, name="text"))
        new.columns = [str(c) for c in new.columns]
        cache = pd.concat([cache, new]) if len(cache) else new
        cache_f.parent.mkdir(parents=True, exist_ok=True)
        cache.reset_index().to_parquet(cache_f, index=False)
        st["cache"] = cache
    return cache.loc[texts].values


class TextEmbed:
    # embedding -> PCA 32 มิติ (fit เฉพาะข้อความ train) -> คอลัมน์ <kind>_0..31
    kind = "multilingual"

    def fit(self, texts):
        from sklearn.decomposition import PCA
        self.pca_ = PCA(TXT_DIM, random_state=0).fit(text_embed(texts, self.kind))
        return self

    def transform(self, texts):
        return pd.DataFrame(self.pca_.transform(text_embed(texts, self.kind)), columns=[f"{self.kind}_{i}" for i in range(TXT_DIM)])


class TextMultilingual(TextEmbed):
    kind = "multilingual"


class TextBert(TextEmbed):
    kind = "wangchanberta"


def formation_features(names):
    # ชื่อหมวดหิน (เช่น "เสาขัว", "Unknown") -> fm_matched, fm_rank_old/young (ยุคจาก ThaiLex), fm_lith_* / fm_env_*
    P_RANK = {p: i for i, p in enumerate(PERIOD_ORDER)}
    rows = []
    for n in names:
        m = thailex_match(n) if isinstance(n, str) and n not in ("Unknown", "nan", "") else None
        if m is None:
            rows.append({"fm_matched": 0})
            continue
        per = [_TL_PERIOD.get(p.strip(), p.strip()) for p in str(m.period).split(",")]
        ranks = [P_RANK[p] for p in per if p in P_RANK]
        rows.append({"fm_matched": 1, "fm_rank_old": min(ranks) if ranks else np.nan, "fm_rank_young": max(ranks) if ranks else np.nan,
                     **{f"fm_{k}": v for k, v in lithology_flags(f"{m.lithology} {m.lithology_pattern}").items()}})
    cols = ["fm_matched", "fm_rank_old", "fm_rank_young"] + [f"fm_lith_{k}" for k in LITH_KW] + [f"fm_env_{k}" for k in ENV_KW]
    return pd.DataFrame(rows).reindex(columns=cols)


def site_formation_table():
    # OBJECTID -> FORMATION / GEO_GROUP จากตาราง notebook 07 (แบบ Impute only: ค่าว่าง = "Unknown")
    t = pd.read_csv(DATA / "processed" / "model" / "sites_impute_only.csv", encoding="utf-8-sig")[["OBJECTID", "FORMATION", "GEO_GROUP"]]
    return t.set_index("OBJECTID")


def map_formation_name(rock_sym):
    # หมวดหินจากแผนที่ธรณี ทธ. ที่ตำแหน่ง (สำหรับพิกัดใหม่): legend "หมวดหินเสาขัว" -> "เสาขัว"; ไม่มีชื่อ -> "Unknown"
    if "__legend" not in _RASTERS:
        _RASTERS["__legend"] = pd.read_csv(DATA / "dmr_geology" / "rock_unit_250k_legend.csv", encoding="utf-8-sig").set_index("TEXT_SYM").label_th
    lab = _RASTERS["__legend"].get(str(rock_sym))
    return re.sub(r"^(หมวดหิน|กลุ่มหิน)", "", lab) if isinstance(lab, str) and re.search("หมวดหิน|กลุ่มหิน", lab) else "Unknown"
