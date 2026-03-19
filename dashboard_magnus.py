# ==============================================================================
#  DASHBOARD MAGNUS ENTREPRISE — Banque de Projets
#  Streamlit · CamemBERT · IDEC · Analyse de portefeuille
# ==============================================================================

import streamlit as st
import re, warnings, io, os, time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pdfplumber
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
#  CONFIG STREAMLIT
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Magnus · Banque de Projets",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
#  STYLE CSS GLOBAL
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Fond principal ── */
.stApp { background: #f6f8fa; color: #24292f; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f6f8fa 100%);
    border-right: 1px solid #d0d7de;
}

/* ── Titre principal ── */
.magnus-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 50%, #f78166 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}

.magnus-subtitle {
    font-family: 'Inter', sans-serif;
    font-weight: 300;
    font-size: 0.95rem;
    color: #57606a;
    margin-bottom: 2rem;
}

/* ── Cartes KPI ── */
.kpi-card {
    background: linear-gradient(135deg, #ffffff 0%, #f6f8fa 100%);
    border: 1px solid #d0d7de;
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(31, 111, 235, 0.12);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-blue::before   { background: linear-gradient(90deg, #58a6ff, #1f6feb); }
.kpi-green::before  { background: linear-gradient(90deg, #3fb950, #2ea043); }
.kpi-purple::before { background: linear-gradient(90deg, #bc8cff, #8957e5); }
.kpi-orange::before { background: linear-gradient(90deg, #f78166, #da3633); }
.kpi-teal::before   { background: linear-gradient(90deg, #39d353, #1a7f37); }

.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #24292f;
    line-height: 1.1;
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #57606a;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 0.4rem;
}
.kpi-icon {
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
    display: block;
}

/* ── Onglets ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #d0d7de;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #57606a;
    font-weight: 500;
    font-size: 0.88rem;
    padding: 8px 16px;
    transition: all 0.2s ease;
    font-family: 'Inter', sans-serif;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    font-weight: 600;
}

/* ── Section titres ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #24292f;
    margin: 1.5rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #d0d7de, transparent);
    margin-left: 0.8rem;
}

/* ── Badges cluster ── */
.cluster-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}

/* ── Insight box ── */
.insight-box {
    background: #ffffff;
    border: 1px solid #d0d7de;
    border-left: 4px solid #58a6ff;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #24292f;
    line-height: 1.6;
}
.insight-box.warning {
    border-left-color: #f78166;
}
.insight-box.success {
    border-left-color: #3fb950;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #d0d7de;
}

/* ── Bouton upload ── */
.stFileUploader {
    border: 2px dashed #d0d7de !important;
    border-radius: 12px !important;
    background: #ffffff !important;
}

/* ── Metric Delta ── */
[data-testid="stMetricDelta"] { font-size: 0.8rem; }

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #58a6ff, #bc8cff) !important;
    border-radius: 4px;
}

/* ── Séparateur ── */
hr { border-color: #d0d7de; margin: 1.5rem 0; }

/* ── Alert boxes ── */
.stAlert { border-radius: 10px; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #484f58; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  CONSTANTES GLOBALES
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

PALETTE = [
    "#58a6ff", "#3fb950", "#bc8cff", "#f78166", "#ffa657",
    "#39d353", "#ff7b72", "#d2a8ff", "#79c0ff", "#56d364"
]

DEFAULT_PDF = "Banque+Projet+2050+09- CP.pdf"

PATTERN_PROJET    = re.compile(r"^(\d{2}[A-Z]\d{3}[A-Z]\d{4,5})\s*[-–]\s*(.+)$", re.DOTALL)
PATTERN_OPERATION = re.compile(r"^(\d{4}[A-Z]\d{5})\s*[-–]\s*(.+)$", re.DOTALL)

HP = {
    "dim_latente"        : 32,
    "dropout"            : 0.15,
    "lr_pretrain"        : 1e-3,
    "lr_raffinement"     : 1e-4,
    "weight_decay"       : 1e-4,
    "batch_size"         : 4,
    "epochs_pretrain"    : 300,
    "epochs_raffinement" : 200,
    "gamma"              : 0.1,
    "T_update"           : 10,
    "alpha_student"      : 1.0,
}

# ──────────────────────────────────────────────────────────────────────────────
#  FONCTIONS UTILITAIRES
# ──────────────────────────────────────────────────────────────────────────────

def nettoyer_cellule(cell) -> str:
    if cell is None:
        return ""
    valeur = str(cell).replace("\n", " ").strip()
    return re.sub(r"\s{2,}", " ", valeur)


def extraire_df_projets(chemin_pdf) -> pd.DataFrame:
    projets_vus = {}
    with pdfplumber.open(chemin_pdf) as pdf:
        for page in pdf.pages:
            tableaux = page.extract_tables()
            if not tableaux:
                continue
            for tableau in tableaux:
                for ligne in tableau:
                    if not ligne:
                        continue
                    cellule_0 = nettoyer_cellule(ligne[0])
                    if not cellule_0:
                        continue
                    match = PATTERN_PROJET.match(cellule_0)
                    if match:
                        id_projet       = match.group(1).strip()
                        intitule_projet = match.group(2).strip()
                        if id_projet not in projets_vus:
                            projets_vus[id_projet] = intitule_projet
    df = pd.DataFrame([{"ID_PROJET": k, "INTITULE_PROJET": v} for k, v in projets_vus.items()])
    return df


def extraire_df_operations(chemin_pdf) -> pd.DataFrame:
    lignes = []
    with pdfplumber.open(chemin_pdf) as pdf:
        for page in pdf.pages:
            tableaux = page.extract_tables()
            if not tableaux:
                continue
            for tableau in tableaux:
                for ligne in tableau:
                    cellules = [nettoyer_cellule(c) for c in ligne]
                    while len(cellules) < 11:
                        cellules.append("")
                    cellules = cellules[:11]
                    cellule_1 = cellules[1]
                    if not cellule_1:
                        continue
                    match = PATTERN_OPERATION.match(cellule_1)
                    if not match:
                        continue
                    lignes.append({
                        "ID_OPERATION"      : match.group(1).strip(),
                        "INTITULE_OPERATION": match.group(2).strip(),
                        "NATURE_OPERATION"  : cellules[2],
                        "COUT"              : cellules[3],
                        "ETAT_MATURITE"     : cellules[4],
                        "ETAT_EXECUTION"    : cellules[5],
                        "LOCALISATION"      : cellules[6],
                        "ETAT_FINANCEMENT"  : cellules[7],
                        "SOURCE_FINANCEMENT": cellules[8],
                        "BUDGETISE"         : cellules[9],
                        "PROGRAMME"         : cellules[10],
                    })
    df = pd.DataFrame(lignes).reset_index(drop=True)

    def convertir_cout(v):
        if not str(v).strip():
            return np.nan
        n = re.sub(r"[\s\u00a0\u202f]", "", str(v)).replace(",", ".")
        try:
            return float(n)
        except:
            return np.nan

    df["COUT_NUM"] = df["COUT"].apply(convertir_cout)
    for col in ["ETAT_MATURITE","ETAT_EXECUTION","ETAT_FINANCEMENT","BUDGETISE","PROGRAMME"]:
        df[col] = df[col].str.strip()
    return df


def construire_mapping(chemin_pdf) -> dict:
    mapping = {}
    projet_actuel = None
    with pdfplumber.open(chemin_pdf) as pdf:
        for page in pdf.pages:
            tableaux = page.extract_tables()
            if not tableaux:
                continue
            for tableau in tableaux:
                for ligne in tableau:
                    cellules = [nettoyer_cellule(c) for c in ligne]
                    while len(cellules) < 11:
                        cellules.append("")
                    cellules = cellules[:11]
                    m_proj = PATTERN_PROJET.match(cellules[0])
                    if m_proj:
                        projet_actuel = m_proj.group(1).strip()
                    m_op = PATTERN_OPERATION.match(cellules[1])
                    if m_op and projet_actuel:
                        mapping[m_op.group(1).strip()] = projet_actuel
    return mapping


def preprocess_texte(texte: str) -> str:
    texte = re.sub(r"\(BIP\d+\s*:?[^)]*\)", "", texte)
    texte = re.sub(r"\b\d[\d\s.,]*\b", " ", texte)
    texte = re.sub(r"[^\w\séàâäèêëîïôùûüçœÉÀÂÄÈÊËÎÏÔÙÛÜÇŒ'''\-]", " ", texte, flags=re.IGNORECASE)
    texte = texte.lower()
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte


# ──────────────────────────────────────────────────────────────────────────────
#  MODÈLES IA
# ──────────────────────────────────────────────────────────────────────────────

class AutoEncodeur(nn.Module):
    def __init__(self, dim_entree=768, dim_latente=32, dropout=0.15):
        super().__init__()
        self.encodeur = nn.Sequential(
            nn.Linear(dim_entree, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, dim_latente),
        )
        self.decodeur = nn.Sequential(
            nn.Linear(dim_latente, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, dim_entree),
        )

    def encode(self, x):
        return self.encodeur(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decodeur(z), z


def soft_assignment(z, mu, alpha=1.0):
    dist_sq = torch.sum((z.unsqueeze(1) - mu.unsqueeze(0)) ** 2, dim=2)
    num = (1.0 + dist_sq / alpha) ** (-(alpha + 1.0) / 2.0)
    return num / num.sum(dim=1, keepdim=True)


def distribution_cible(Q):
    freq = Q.sum(dim=0, keepdim=True)
    P = Q ** 2 / freq
    return P / P.sum(dim=1, keepdim=True)


def trouver_coude(k_vals, inerties):
    """Détection automatique du coude par distance perpendiculaire maximale."""
    n = len(inerties)
    if n < 3:
        return k_vals[0]
    # Normalisation entre 0 et 1
    y = np.array(inerties, dtype=float)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)
    x_norm = np.linspace(0, 1, n)
    # Droite reliant le premier et le dernier point
    dx = x_norm[-1] - x_norm[0]
    dy = y_norm[-1] - y_norm[0]
    # Distance perpendiculaire de chaque point à cette droite
    distances = abs(dy * x_norm - dx * y_norm + x_norm[-1] * y_norm[0] - y_norm[-1] * x_norm[0])
    distances /= np.sqrt(dy**2 + dx**2 + 1e-12)
    knee_idx = int(np.argmax(distances))
    return k_vals[knee_idx]


def nommer_clusters(df_projets):
    """TF-IDF naming for clusters."""
    ids_clusters = sorted(df_projets["CLUSTER"].unique().astype(int))
    corpus = {
        c: " ".join(df_projets[df_projets["CLUSTER"] == c]["INTITULE_CLEAN"].tolist())
        for c in ids_clusters
    }
    vectorizer    = TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 2))
    matrice_tfidf = vectorizer.fit_transform(list(corpus.values()))
    vocabulaire   = vectorizer.get_feature_names_out()
    noms = {}
    for i, c_id in enumerate(ids_clusters):
        scores  = matrice_tfidf[i].toarray().flatten()
        top_idx = scores.argsort()[::-1][:3]
        mots    = [vocabulaire[j] for j in top_idx if scores[j] > 0]
        nom     = " · ".join([m.capitalize() for m in mots]) if mots else f"Groupe {c_id}"
        noms[c_id] = nom
    return noms


# ──────────────────────────────────────────────────────────────────────────────
#  PIPELINE PRINCIPAL (mis en cache)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def charger_camembert():
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("camembert-base")
    modele    = AutoModel.from_pretrained("camembert-base").to(device)
    modele.eval()
    return tokenizer, modele, device


def generer_embeddings(textes, tokenizer, modele, device, progress_cb=None):
    tous = []
    taille_batch = 8
    n_batches = -(-len(textes) // taille_batch)
    for i in range(0, len(textes), taille_batch):
        batch = textes[i:i + taille_batch]
        enc   = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            out = modele(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        masque = enc["attention_mask"].unsqueeze(-1).float().to(device)
        emb    = (out.last_hidden_state * masque).sum(1) / masque.sum(1)
        tous.append(emb.cpu().numpy())
        if progress_cb:
            progress_cb((i // taille_batch + 1) / n_batches)
    return np.vstack(tous)


def lancer_pipeline(chemin_pdf, progress_placeholder):
    """Pipeline complet. Retourne un dict de résultats."""
    res = {}

    # ── 1. Extraction PDF ──────────────────────────────────────────────────────
    with progress_placeholder.container():
        st.info("📄 Étape 1/6 — Extraction des données du PDF…")
        bar = st.progress(0)
    df_projets    = extraire_df_projets(chemin_pdf)
    df_operations = extraire_df_operations(chemin_pdf)
    bar.progress(100)

    if len(df_projets) == 0:
        return None, "Aucun projet trouvé dans ce PDF. Vérifiez le format du fichier."

    # ── 2. Préprocessing ───────────────────────────────────────────────────────
    with progress_placeholder.container():
        st.info("🔤 Étape 2/6 — Nettoyage des intitulés de projets…")
        bar = st.progress(0)
    df_projets["INTITULE_CLEAN"] = df_projets["INTITULE_PROJET"].apply(preprocess_texte)
    bar.progress(100)

    # ── 3. Embeddings CamemBERT ────────────────────────────────────────────────
    with progress_placeholder.container():
        st.info("🧠 Étape 3/6 — Génération des représentations sémantiques (CamemBERT)…")
        bar = st.progress(0)
    tokenizer, modele_cam, device = charger_camembert()
    textes     = df_projets["INTITULE_CLEAN"].tolist()
    embeddings = generer_embeddings(textes, tokenizer, modele_cam, device, lambda p: bar.progress(int(p*100)))
    normes         = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normes         = np.where(normes == 0, 1.0, normes)
    embeddings_norm = embeddings / normes

    # ── 4. Auto-encodeur + Méthode du coude ───────────────────────────────────
    with progress_placeholder.container():
        st.info("⚙️ Étape 4/6 — Pré-entraînement de l'auto-encodeur…")
        bar = st.progress(0)

    torch.manual_seed(SEED)
    ae = AutoEncodeur(768, HP["dim_latente"], HP["dropout"]).to(device)
    X  = torch.tensor(embeddings_norm, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=HP["batch_size"], shuffle=True, drop_last=False)
    optim_pre  = optim.AdamW(ae.parameters(), lr=HP["lr_pretrain"], weight_decay=HP["weight_decay"])
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(optim_pre, T_max=HP["epochs_pretrain"], eta_min=1e-6)
    mse_loss   = nn.MSELoss()
    hist_pre   = []
    ae.train()
    for epoch in range(1, HP["epochs_pretrain"] + 1):
        perte_ep = 0.0
        for (bx,) in loader:
            bx = bx.to(device)
            x_rec, _ = ae(bx)
            loss = mse_loss(x_rec, bx)
            optim_pre.zero_grad(); loss.backward(); optim_pre.step()
            perte_ep += loss.item() * len(bx)
        scheduler.step()
        hist_pre.append(perte_ep / len(X))
        bar.progress(epoch / HP["epochs_pretrain"])

    # Méthode du coude (INERTIE UNIQUEMENT)
    with progress_placeholder.container():
        st.info("📐 Étape 4b/6 — Sélection automatique du nombre de groupes (méthode du coude)…")
        bar = st.progress(0)

    ae.eval()
    with torch.no_grad():
        _, Z = ae(X.to(device))
        Z_np = Z.cpu().numpy()

    K_MAX  = min(12, len(df_projets) - 1)
    K_MIN  = 2
    k_vals = list(range(K_MIN, K_MAX + 1))
    inerties = []
    for i, k in enumerate(k_vals):
        km  = KMeans(n_clusters=k, random_state=SEED, n_init=20, max_iter=500)
        km.fit(Z_np)
        inerties.append(km.inertia_)
        bar.progress((i + 1) / len(k_vals))

    K_OPT = trouver_coude(k_vals, inerties)

    # ── 5. Raffinement IDEC ───────────────────────────────────────────────────
    with progress_placeholder.container():
        st.info(f"🔬 Étape 5/6 — Raffinement IDEC (K = {K_OPT} groupes)…")
        bar = st.progress(0)

    ae.eval()
    with torch.no_grad():
        _, Z_init = ae(X.to(device))
    km_init = KMeans(n_clusters=K_OPT, random_state=SEED, n_init=20, max_iter=500)
    km_init.fit(Z_init.cpu().numpy())
    centroides = nn.Parameter(torch.tensor(km_init.cluster_centers_, dtype=torch.float32).to(device))

    params_raf = list(ae.parameters()) + [centroides]
    optim_raf  = optim.AdamW(params_raf, lr=HP["lr_raffinement"], weight_decay=HP["weight_decay"])
    hist_total, hist_mse_r, hist_kl = [], [], []
    ae.train()
    for epoch in range(1, HP["epochs_raffinement"] + 1):
        ep_t = ep_m = ep_k = 0.0
        for (bx,) in loader:
            bx = bx.to(device)
            x_rec, z = ae(bx)
            L_mse  = mse_loss(x_rec, bx)
            Q_b    = soft_assignment(z, centroides, HP["alpha_student"])
            P_b    = distribution_cible(Q_b).detach()
            eps    = 1e-10
            L_kl   = (P_b * torch.log((P_b + eps) / (Q_b + eps))).sum(1).mean()
            L      = L_mse + HP["gamma"] * L_kl
            optim_raf.zero_grad(); L.backward(); optim_raf.step()
            n      = len(bx)
            ep_t  += L.item() * n; ep_m += L_mse.item() * n; ep_k += L_kl.item() * n
        N = len(X)
        hist_total.append(ep_t/N); hist_mse_r.append(ep_m/N); hist_kl.append(ep_k/N)
        bar.progress(epoch / HP["epochs_raffinement"])

    # ── 6. Assignation finale et export ───────────────────────────────────────
    with progress_placeholder.container():
        st.info("✅ Étape 6/6 — Assignation des projets aux groupes…")
        bar = st.progress(0)

    ae.eval()
    with torch.no_grad():
        _, Z_fin = ae(X.to(device))
        Z_fin_np = Z_fin.cpu().numpy()
        Q_fin    = soft_assignment(Z_fin, centroides, HP["alpha_student"])
        Q_fin_np = Q_fin.cpu().numpy()

    labels = Q_fin_np.argmax(axis=1)
    df_projets["CLUSTER"]   = labels
    df_projets["CONFIANCE"] = Q_fin_np.max(axis=1)

    # Projection 2D
    try:
        import umap
        reducer  = umap.UMAP(n_components=2, random_state=SEED,
                             n_neighbors=min(5, len(df_projets)-1), min_dist=0.1)
        Z_2d     = reducer.fit_transform(Z_fin_np)
        methode  = "UMAP"
    except:
        pca  = PCA(n_components=2, random_state=SEED)
        Z_2d = pca.fit_transform(Z_fin_np)
        methode = "PCA"

    # Métriques
    sil = silhouette_score(Z_fin_np, labels) if len(np.unique(labels)) > 1 else 0
    db  = davies_bouldin_score(Z_fin_np, labels) if len(np.unique(labels)) > 1 else 0

    # Nommage TF-IDF
    noms_clusters = nommer_clusters(df_projets)
    df_projets["NOM_CLUSTER"] = df_projets["CLUSTER"].apply(lambda c: noms_clusters.get(int(c), f"Groupe {c}"))

    # Mapping et jointure
    mapping = construire_mapping(chemin_pdf)
    df_operations["ID_PROJET"] = df_operations["ID_OPERATION"].map(mapping)
    df_ops_final = df_operations.merge(
        df_projets[["ID_PROJET","CLUSTER","CONFIANCE","NOM_CLUSTER"]],
        on="ID_PROJET", how="left"
    )

    bar.progress(100)

    res = {
        "df_projets"    : df_projets,
        "df_operations" : df_operations,
        "df_ops_final"  : df_ops_final,
        "Z_fin_np"      : Z_fin_np,
        "Z_2d"          : Z_2d,
        "methode_proj"  : methode,
        "Q_fin_np"      : Q_fin_np,
        "k_vals"        : k_vals,
        "inerties"      : inerties,
        "hist_pre"      : hist_pre,
        "hist_total"    : hist_total,
        "hist_mse_r"    : hist_mse_r,
        "hist_kl"       : hist_kl,
        "K_OPT"         : K_OPT,
        "sil"           : sil,
        "db"            : db,
        "noms_clusters" : noms_clusters,
    }
    return res, None


# ──────────────────────────────────────────────────────────────────────────────
#  FONCTIONS DE RENDU PLOTLY (dark theme)
# ──────────────────────────────────────────────────────────────────────────────

PLOTLY_THEME = dict(
    plot_bgcolor  = "#ffffff",
    paper_bgcolor = "#ffffff",
    font          = dict(color="#24292f", family="Inter"),
    xaxis         = dict(gridcolor="#e5e7eb", linecolor="#d0d7de", zerolinecolor="#d0d7de"),
    yaxis         = dict(gridcolor="#e5e7eb", linecolor="#d0d7de", zerolinecolor="#d0d7de"),
)

def fig_update(fig):
    fig.update_layout(**PLOTLY_THEME)
    return fig


def render_kpi(label, value, icon, color_class, delta=None):
    delta_html = f'<div style="font-size:0.75rem;color:#3fb950;margin-top:0.2rem">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card {color_class}">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
        {delta_html}
    </div>"""


def plot_elbow(k_vals, inerties, K_OPT):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_vals, y=inerties, mode="lines+markers",
        line=dict(color="#58a6ff", width=3),
        marker=dict(size=9, color="#58a6ff", line=dict(color="white", width=2)),
        name="Inertie"
    ))
    fig.add_vline(x=K_OPT, line_dash="dash", line_color="#f78166", line_width=2)
    fig.add_annotation(
        x=K_OPT, y=inerties[k_vals.index(K_OPT)],
        text=f" ← K optimal = {K_OPT}", showarrow=False,
        font=dict(color="#f78166", size=13, family="Syne"), xanchor="left"
    )
    fig.update_layout(
        title=dict(text="📐 Méthode du coude — Choix du nombre de groupes", font=dict(size=15, family="Syne")),
        xaxis_title="Nombre de groupes (K)",
        yaxis_title="Inertie (dispersion interne)",
        **PLOTLY_THEME
    )
    return fig


def plot_training_curves(hist_pre, hist_total, hist_mse_r, hist_kl):
    fig = make_subplots(1, 2, subplot_titles=("Pré-entraînement", "Raffinement IDEC"))
    # Pré-entraînement
    fig.add_trace(go.Scatter(
        y=hist_pre, mode="lines", name="Perte MSE",
        line=dict(color=PALETTE[0], width=2)
    ), row=1, col=1)
    # Raffinement
    fig.add_trace(go.Scatter(
        y=hist_total, mode="lines", name="Perte totale",
        line=dict(color=PALETTE[2], width=2.5)
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        y=hist_mse_r, mode="lines", name="MSE recon.",
        line=dict(color=PALETTE[0], width=1.5, dash="dash")
    ), row=1, col=2)
    fig.update_layout(
        title=dict(text="📉 Courbes d'apprentissage", font=dict(size=15, family="Syne")),
        **PLOTLY_THEME, height=380
    )
    fig.update_xaxes(gridcolor="#e5e7eb", linecolor="#d0d7de", tickfont=dict(color="#24292f"))
    fig.update_yaxes(gridcolor="#e5e7eb", linecolor="#d0d7de", tickfont=dict(color="#24292f"))
    return fig


def plot_clusters_scatter(df_projets, Z_2d, methode, K_OPT, noms_clusters):
    couleurs = [PALETTE[i % len(PALETTE)] for i in range(K_OPT)]
    fig = go.Figure()
    for c in range(K_OPT):
        msk   = df_projets["CLUSTER"].values == c
        hover = df_projets[msk]["INTITULE_PROJET"].str[:60].tolist()
        ids   = df_projets[msk]["ID_PROJET"].tolist()
        nom   = noms_clusters.get(c, f"Groupe {c}")
        fig.add_trace(go.Scatter(
            x=Z_2d[msk, 0], y=Z_2d[msk, 1],
            mode="markers+text",
            name=f"Groupe {c} — {nom}",
            marker=dict(size=16, color=couleurs[c],
                        line=dict(color="white", width=1.5), opacity=0.9),
            text=[f"<b>{id_}</b><br>{h}" for id_, h in zip(ids, hover)],
            textposition="top center",
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))
    fig.update_layout(
        title=dict(text=f"🗺️ Carte des projets par groupe ({methode})", font=dict(size=15, family="Syne")),
        xaxis_title=f"Dimension 1 ({methode})",
        yaxis_title=f"Dimension 2 ({methode})",
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#d0d7de", borderwidth=1),
        **PLOTLY_THEME, height=520
    )
    return fig


def plot_heatmap_Q(df_projets, Q_fin_np, K_OPT):
    labels_lignes = (df_projets["ID_PROJET"] + " — " + df_projets["INTITULE_PROJET"].str[:35]).tolist()
    cols = [f"Groupe {c}" for c in range(K_OPT)]
    fig = go.Figure(go.Heatmap(
        z=Q_fin_np, x=cols, y=labels_lignes,
        colorscale="Blues", text=np.round(Q_fin_np, 2),
        texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title="Probabilité", tickfont=dict(color="#24292f")),
        hovertemplate="Projet : %{y}<br>%{x} : %{z:.2f}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="🎯 Probabilité d'appartenance à chaque groupe", font=dict(size=15, family="Syne")),
        **PLOTLY_THEME, height=max(400, len(df_projets) * 40)
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_top5(df_operations):
    df_c = df_operations.dropna(subset=["COUT_NUM"])
    df_c = df_c[df_c["COUT_NUM"] > 0]
    top5_plus  = df_c.nlargest(5, "COUT_NUM").copy()
    top5_moins = df_c.nsmallest(5, "COUT_NUM").copy()

    fig = make_subplots(1, 2, subplot_titles=("🔴 Les 5 plus coûteuses", "🟢 Les 5 moins coûteuses"))
    top5_plus["label"]  = top5_plus["INTITULE_OPERATION"].str[:45] + "…"
    top5_moins["label"] = top5_moins["INTITULE_OPERATION"].str[:45] + "…"

    fig.add_trace(go.Bar(
        y=top5_plus["label"], x=top5_plus["COUT_NUM"],
        orientation="h", marker_color=PALETTE[3],
        text=[f"{v:,.0f}" for v in top5_plus["COUT_NUM"]],
        textposition="inside", textfont=dict(color="white", size=10),
        hovertemplate="%{y}<br>Coût : %{x:,.0f} KFCFA<extra></extra>",
        name="Plus chers"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=top5_moins["label"], x=top5_moins["COUT_NUM"],
        orientation="h", marker_color=PALETTE[1],
        text=[f"{v:,.0f}" for v in top5_moins["COUT_NUM"]],
        textposition="inside", textfont=dict(color="white", size=10),
        hovertemplate="%{y}<br>Coût : %{x:,.0f} KFCFA<extra></extra>",
        name="Moins chers"
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text="💰 Comparatif des coûts des opérations", font=dict(size=15, family="Syne")),
        showlegend=False, **PLOTLY_THEME, height=420
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(gridcolor="#e5e7eb", linecolor="#d0d7de", tickfont=dict(color="#24292f"))
    fig.update_yaxes(gridcolor="#e5e7eb", linecolor="#d0d7de", tickfont=dict(color="#24292f"))
    return fig, top5_plus, top5_moins


def plot_donut_execution(df_operations):
    serie = df_operations["ETAT_EXECUTION"].str.strip().copy()
    serie = serie.fillna("Non renseigné").replace("", "Non renseigné")
    comptage = serie.value_counts()
    fig = go.Figure(go.Pie(
        labels=comptage.index.tolist(), values=comptage.values.tolist(),
        hole=0.55, marker_colors=PALETTE[:len(comptage)],
        textinfo="percent+value", textfont=dict(size=12, color="#24292f"),
        hovertemplate="%{label}<br>Nb : %{value}<br>Part : %{percent}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="🔄 Répartition par état d'exécution", font=dict(size=15, family="Syne")),
        annotations=[dict(text=f"Total<br><b>{len(df_operations)}</b>", showarrow=False,
                          font=dict(size=14, color="#24292f"))],
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#d0d7de"),
        **PLOTLY_THEME, height=420
    )
    return fig


def plot_histogrammes_etats(df_operations):
    df_m = df_operations["ETAT_MATURITE"].str.strip().str.upper().copy()
    df_m = df_m.fillna("Non renseigné").replace("", "Non renseigné")
    df_f = df_operations["ETAT_FINANCEMENT"].str.strip().copy()
    df_f = df_f.fillna("Non renseigné").replace("", "Non renseigné")

    c_m = df_m.value_counts(); c_f = df_f.value_counts()

    fig = make_subplots(1, 2, subplot_titles=("État de maturité", "État de financement"))
    fig.add_trace(go.Bar(
        x=c_m.index, y=c_m.values, marker_color=[PALETTE[1] if "VISA" in str(x) and "NON" not in str(x) else PALETTE[3] for x in c_m.index],
        text=c_m.values, textposition="outside", textfont=dict(color="#24292f"),
        hovertemplate="%{x}<br>Nb : %{y}<extra></extra>"
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=c_f.index, y=c_f.values, marker_color=PALETTE[:len(c_f)],
        text=c_f.values, textposition="outside", textfont=dict(color="#24292f"),
        hovertemplate="%{x}<br>Nb : %{y}<extra></extra>"
    ), row=1, col=2)
    fig.update_layout(
        title=dict(text="📊 États des opérations", font=dict(size=15, family="Syne")),
        showlegend=False, **PLOTLY_THEME, height=400
    )
    fig.update_xaxes(gridcolor="#e5e7eb", linecolor="#d0d7de", tickangle=-15, tickfont=dict(color="#24292f"))
    fig.update_yaxes(gridcolor="#e5e7eb", linecolor="#d0d7de", tickfont=dict(color="#24292f"))
    return fig


def plot_gap_financement(df_operations):
    df_f = df_operations.dropna(subset=["COUT_NUM"]).copy()
    df_f["ETAT_FINANCEMENT"] = df_f["ETAT_FINANCEMENT"].str.strip()

    def risque(x):
        m = str(x).lower()
        if "acquis" in m and "partiel" not in m: return "✅ Financé"
        elif "partiel" in m or "intérieur" in m: return "⚠ Partiel"
        else: return "🔴 Non financé"

    df_f["RISQUE"] = df_f["ETAT_FINANCEMENT"].apply(risque)
    agg = df_f.groupby("ETAT_FINANCEMENT")["COUT_NUM"].agg(total="sum", nb="count").reset_index()
    agg["pct"] = agg["total"] / agg["total"].sum() * 100
    agg["RISQUE"] = agg["ETAT_FINANCEMENT"].apply(risque)

    color_map = {"✅ Financé": PALETTE[1], "⚠ Partiel": PALETTE[4], "🔴 Non financé": PALETTE[3]}
    coul = [color_map.get(r, PALETTE[7]) for r in agg["RISQUE"]]

    fig = make_subplots(1, 2, subplot_titles=("Montant par état (KFCFA)", "Répartition (%)"),
                        specs=[[{"type": "bar"}, {"type": "pie"}]])
    fig.add_trace(go.Bar(
        y=agg["ETAT_FINANCEMENT"], x=agg["total"],
        orientation="h", marker_color=coul,
        text=[f"{v:,.0f}" for v in agg["total"]],
        textposition="inside", textfont=dict(color="#24292f", size=9),
        hovertemplate="%{y}<br>%{x:,.0f} KFCFA<extra></extra>"
    ), row=1, col=1)
    fig.add_trace(go.Pie(
        labels=agg["ETAT_FINANCEMENT"], values=agg["total"],
        hole=0.5, marker_colors=coul,
        textinfo="percent", textfont=dict(size=11, color="#24292f"),
        hovertemplate="%{label}<br>%{value:,.0f} KFCFA (%{percent})<extra></extra>"
    ), row=1, col=2)
    fig.update_layout(
        title=dict(text="💼 Analyse du gap de financement", font=dict(size=15, family="Syne")),
        showlegend=False, **PLOTLY_THEME, height=420
    )
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    return fig, agg


def plot_boxplot_couts(df_operations):
    df_b = df_operations.dropna(subset=["COUT_NUM"]).copy()
    df_b["ETAT_EXECUTION"] = df_b["ETAT_EXECUTION"].str.strip()
    etats = df_b["ETAT_EXECUTION"].unique().tolist()

    fig = go.Figure()
    for i, e in enumerate(etats):
        data = df_b[df_b["ETAT_EXECUTION"] == e]["COUT_NUM"].tolist()
        fig.add_trace(go.Box(
            y=data, name=e, marker_color=PALETTE[i % len(PALETTE)],
            boxpoints="outliers", jitter=0.3, pointpos=-1.8,
            marker=dict(size=6), line=dict(width=2),
            hovertemplate=f"<b>{e}</b><br>Coût : %{{y:,.0f}} KFCFA<extra></extra>"
        ))
    fig.update_layout(
        title=dict(text="📦 Distribution des coûts par état d'exécution", font=dict(size=15, family="Syne")),
        yaxis_title="Coût (KFCFA)",
        showlegend=False, **PLOTLY_THEME, height=420
    )
    return fig


def plot_couts_par_cluster(df_ops_final, noms_clusters, K_OPT):
    df_c = df_ops_final.dropna(subset=["COUT_NUM","CLUSTER"]).copy()
    df_c["CLUSTER"] = df_c["CLUSTER"].astype(int)
    agg = df_c.groupby("CLUSTER")["COUT_NUM"].agg(total="sum", moyenne="mean", nb="count").reset_index()
    agg["label"] = agg["CLUSTER"].apply(lambda c: f"Groupe {c}<br><sub>{noms_clusters.get(c,'')}</sub>")
    coul = [PALETTE[int(c) % len(PALETTE)] for c in agg["CLUSTER"]]

    fig = make_subplots(1, 3, subplot_titles=("Coût total (KFCFA)", "Coût moyen (KFCFA)", "Nb d'opérations"))
    for col_idx, (col_data, title) in enumerate([("total","Total"), ("moyenne","Moyenne"), ("nb","Opérations")], 1):
        fig.add_trace(go.Bar(
            x=agg["label"], y=agg[col_data],
            marker_color=coul, marker_line_color="white", marker_line_width=1.5,
            text=[f"{v:,.0f}" for v in agg[col_data]],
            textposition="outside", textfont=dict(color="#24292f", size=9),
            hovertemplate=f"<b>%{{x}}</b><br>{title} : %{{y:,.0f}}<extra></extra>",
            showlegend=False
        ), row=1, col=col_idx)
    fig.update_layout(
        title=dict(text="🏷️ Profil financier par groupe de projets", font=dict(size=15, family="Syne")),
        **PLOTLY_THEME, height=420
    )
    fig.update_xaxes(gridcolor="#e5e7eb", linecolor="#d0d7de", tickfont=dict(color="#24292f"))
    fig.update_yaxes(gridcolor="#e5e7eb", linecolor="#d0d7de", tickfont=dict(color="#24292f"))
    return fig


def plot_radar(df_ops_final, noms_clusters, K_OPT):
    df_r = df_ops_final.dropna(subset=["CLUSTER"]).copy()
    df_r["CLUSTER"] = df_r["CLUSTER"].astype(int)
    df_r["M_SCORE"] = df_r["ETAT_MATURITE"].str.strip().str.upper().map({"VISA":1,"NON VISA":0}).fillna(0)
    df_r["E_SCORE"] = df_r["ETAT_EXECUTION"].str.strip().map({"Achevé":1.0,"En cours d'exécution":0.5,"Non entamé":0.0}).fillna(0)
    df_r["F_SCORE"] = df_r["ETAT_FINANCEMENT"].str.strip().apply(
        lambda x: 1.0 if "acquis" in str(x).lower() and "partiel" not in str(x).lower()
        else 0.5 if "partiel" in str(x).lower() else 0.0)
    df_r["B_SCORE"] = df_r["BUDGETISE"].str.strip().str.upper().map({"O":1,"N":0}).fillna(0)
    df_r["P_SCORE"] = df_r["PROGRAMME"].str.strip().str.upper().map({"O":1,"N":0}).fillna(0)

    dims   = ["M_SCORE","E_SCORE","F_SCORE","B_SCORE","P_SCORE"]
    labels = ["Maturité","Exécution","Financement","Budgétisé","Programmé"]
    profils = df_r.groupby("CLUSTER")[dims].mean().reset_index()

    fig = go.Figure()
    for _, row in profils.iterrows():
        c    = int(row["CLUSTER"])
        vals = row[dims].tolist() + [row[dims[0]]]
        lbs  = labels + [labels[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=lbs, fill="toself",
            name=f"Groupe {c} — {noms_clusters.get(c,'')}",
            line_color=PALETTE[c % len(PALETTE)], opacity=0.8,
            hovertemplate="%{theta} : %{r:.2f}<extra>Groupe " + str(c) + "</extra>"
        ))
    fig.update_layout(
        polar=dict(
            bgcolor="#ffffff",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#d0d7de", color="#24292f"),
            angularaxis=dict(gridcolor="#d0d7de", color="#24292f")
        ),
        title=dict(text="🕸️ Profil opérationnel par groupe", font=dict(size=15, family="Syne")),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#d0d7de"),
        **PLOTLY_THEME, height=480
    )
    return fig


def plot_croisement_maturite(df_operations):
    df_c = df_operations.copy()
    df_c["ETAT_MATURITE"]  = df_c["ETAT_MATURITE"].str.strip().str.upper().fillna("NR")
    df_c["ETAT_EXECUTION"] = df_c["ETAT_EXECUTION"].str.strip().fillna("NR")
    contingence = pd.crosstab(df_c["ETAT_MATURITE"], df_c["ETAT_EXECUTION"])

    fig = go.Figure(go.Heatmap(
        z=contingence.values, x=contingence.columns.tolist(), y=contingence.index.tolist(),
        colorscale="Blues", text=contingence.values, texttemplate="%{text}",
        textfont=dict(size=13, color="#24292f"),
        hovertemplate="Maturité: %{y}<br>Exécution: %{x}<br>Nb: %{z}<extra></extra>",
        colorbar=dict(tickfont=dict(color="#24292f"))
    ))
    fig.update_layout(
        title=dict(text="🔀 Maturité × Exécution (croisement)", font=dict(size=15, family="Syne")),
        xaxis_title="État d'exécution", yaxis_title="État de maturité",
        **PLOTLY_THEME, height=380
    )
    return fig


def plot_kpi_budgetise_programme(df_operations):
    def kpi_calc(col, nom):
        s = col.str.strip().str.upper()
        total = len(s.dropna())
        n_o = (s == "O").sum(); n_n = (s == "N").sum()
        return {"nom": nom, "total": total, "n_oui": int(n_o), "n_non": int(n_n),
                "p_oui": n_o/total*100 if total else 0, "p_non": n_n/total*100 if total else 0}

    kb = kpi_calc(df_operations["BUDGETISE"],  "Budgétisé")
    kp = kpi_calc(df_operations["PROGRAMME"],  "Programmé")

    cats  = ["Budgétisé","Programmé"]
    v_oui = [kb["n_oui"], kp["n_oui"]]
    v_non = [kb["n_non"], kp["n_non"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="OUI ✅", y=cats, x=v_oui, orientation="h",
        marker_color=PALETTE[1], text=v_oui, textposition="inside",
        textfont=dict(color="white", size=13, family="Syne"),
        hovertemplate="%{y} — OUI : %{x}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        name="NON ❌", y=cats, x=v_non, orientation="h",
        marker_color=PALETTE[3], text=v_non, textposition="inside",
        textfont=dict(color="white", size=13, family="Syne"),
        base=v_oui, hovertemplate="%{y} — NON : %{x}<extra></extra>"
    ))
    fig.update_layout(
        barmode="stack",
        title=dict(text="📋 Budgétisation et programmation des opérations", font=dict(size=15, family="Syne")),
        xaxis_title="Nombre d'opérations",
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#d0d7de"),
        **PLOTLY_THEME, height=300
    )
    return fig, kb, kp


# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 1rem;">
        <div style="font-size:2.5rem">🏗️</div>
        <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:1.2rem;
                    background:linear-gradient(135deg,#58a6ff,#bc8cff);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text;">MAGNUS</div>
        <div style="color:#8b949e; font-size:0.78rem; margin-top:0.2rem">Banque de Projets · Analyse IA</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📁 Source de données")

    uploaded_file = st.file_uploader(
        "Importer un nouveau PDF",
        type=["pdf"],
        help="Glissez-déposez votre fichier PDF de banque de projets ici."
    )

    if uploaded_file:
        st.success(f"✅ {uploaded_file.name}")
        use_default = False
    else:
        if os.path.exists(DEFAULT_PDF):
            st.info(f"📄 Fichier par défaut :\n`{DEFAULT_PDF}`")
            use_default = True
        else:
            st.warning("⚠️ Aucun fichier par défaut trouvé.\nImportez un PDF ci-dessus.")
            use_default = False

    st.markdown("---")

    run_pipeline = st.button(
        "🚀 Lancer l'analyse",
        type="primary",
        use_container_width=True,
        disabled=(not use_default and uploaded_file is None)
    )

    st.markdown("---")
    st.markdown("""
    <div style="color:#8b949e; font-size:0.75rem; line-height:1.6">
    <b style="color:#c9d1d9">Pipeline IA :</b><br>
    1️⃣ Extraction PDF<br>
    2️⃣ Nettoyage textes<br>
    3️⃣ Embeddings CamemBERT<br>
    4️⃣ Auto-encodeur IDEC<br>
    5️⃣ Méthode du coude<br>
    6️⃣ Clustering + nommage
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="magnus-title">Banque de Projets — Magnus Entreprise</div>
<div class="magnus-subtitle">
    Analyse intelligente du portefeuille · Clustering sémantique · Indicateurs financiers et opérationnels
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  GESTION DU PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state["results"] = None

progress_placeholder = st.empty()

if run_pipeline:
    # Déterminer le chemin du PDF
    if uploaded_file:
        tmp_path = f"/tmp/{uploaded_file.name}"
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_path = tmp_path
    else:
        pdf_path = DEFAULT_PDF

    with st.spinner(""):
        results, err = lancer_pipeline(pdf_path, progress_placeholder)

    progress_placeholder.empty()

    if err:
        st.error(f"❌ Erreur : {err}")
    else:
        st.session_state["results"] = results
        st.success("✅ Analyse terminée avec succès !")
        time.sleep(0.5)
        st.rerun()

# Chargement automatique si fichier par défaut dispo et pas encore analysé
elif st.session_state["results"] is None and use_default and os.path.exists(DEFAULT_PDF):
    with st.spinner("⏳ Chargement de l'analyse par défaut…"):
        results, err = lancer_pipeline(DEFAULT_PDF, progress_placeholder)
    progress_placeholder.empty()
    if not err:
        st.session_state["results"] = results
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
#  AFFICHAGE DES RÉSULTATS
# ──────────────────────────────────────────────────────────────────────────────

res = st.session_state.get("results")

if res is None:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; color:#8b949e;">
        <div style="font-size:4rem; margin-bottom:1rem">🏗️</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.4rem; color:#e6edf3; margin-bottom:0.5rem">
            Prêt à analyser votre portefeuille
        </div>
        <div style="font-size:0.9rem">
            Importez un fichier PDF via le panneau de gauche puis cliquez sur <b>Lancer l'analyse</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


df_projets    = res["df_projets"]
df_operations = res["df_operations"]
df_ops_final  = res["df_ops_final"]
K_OPT         = res["K_OPT"]
noms_clusters = res["noms_clusters"]
sil           = res["sil"]


# ──────────────────────────────────────────────────────────────────────────────
#  ONGLETS
# ──────────────────────────────────────────────────────────────────────────────
tab_synthese, tab_clusters, tab_finances, tab_operations, tab_modele, tab_donnees = st.tabs([
    "🏠 Synthèse",
    "🗂️ Projets & Groupes",
    "💰 Finances",
    "⚙️ Opérations",
    "🔬 Modèle IA",
    "📋 Données brutes",
])


# ══════════════════════════════════════════════════════════════════════════════
#  ONGLET 1 — SYNTHÈSE
# ══════════════════════════════════════════════════════════════════════════════
with tab_synthese:

    # KPI CARDS
    cout_total = df_operations["COUT_NUM"].sum()
    cout_moyen = df_operations["COUT_NUM"].mean()

    cols = st.columns(5)
    kpis = [
        ("Projets analysés",      len(df_projets),              "🏗️", "kpi-blue"),
        ("Opérations totales",    len(df_operations),           "📋", "kpi-green"),
        ("Budget total (KFCFA)",  f"{cout_total:,.0f}",         "💰", "kpi-purple"),
        ("Coût moyen (KFCFA)",    f"{cout_moyen:,.0f}",         "📊", "kpi-orange"),
        (f"Groupes identifiés",   K_OPT,                        "🎯", "kpi-teal"),
    ]
    for col, (label, value, icon, cls) in zip(cols, kpis):
        col.markdown(render_kpi(label, value, icon, cls), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # LIGNE 1
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.plotly_chart(plot_donut_execution(df_operations), use_container_width=True, key="synthese_donut_execution")

    with col_b:
        # Clusters bar
        cluster_c = df_projets["CLUSTER"].value_counts().sort_index()
        fig_cl = go.Figure()
        for c, cnt in cluster_c.items():
            nom = noms_clusters.get(int(c), f"Groupe {c}")
            fig_cl.add_trace(go.Bar(
                x=[f"Groupe {int(c)}"], y=[cnt],
                marker_color=PALETTE[int(c) % len(PALETTE)],
                name=nom, text=[cnt], textposition="outside",
                textfont=dict(color="#c9d1d9", size=13, family="Syne"),
                hovertemplate=f"<b>Groupe {int(c)} — {nom}</b><br>{cnt} projet(s)<extra></extra>"
            ))
        fig_cl.update_layout(
            title=dict(text=f"🎯 Répartition des projets par groupe (K = {K_OPT})", font=dict(size=15, family="Syne")),
            showlegend=False, **PLOTLY_THEME, height=420
        )
        fig_cl.update_yaxes(title_text="Nombre de projets")
        st.plotly_chart(fig_cl, use_container_width=True, key="synthese_clusters_bar")

    # LIGNE 2
    col_c, col_d = st.columns([1, 1])
    with col_c:
        st.plotly_chart(plot_histogrammes_etats(df_operations), use_container_width=True, key="synthese_histogrammes_etats")
    with col_d:
        fig_bp, kb, kp = plot_kpi_budgetise_programme(df_operations)
        st.plotly_chart(fig_bp, use_container_width=True, key="synthese_kpi_budgetise_programme")

        cols_kpi2 = st.columns(4)
        mini_kpis = [
            ("Budgétisé OUI",    f"{kb['p_oui']:.0f}%", "kpi-green"),
            ("Budgétisé NON",    f"{kb['p_non']:.0f}%", "kpi-orange"),
            ("Programmé OUI",    f"{kp['p_oui']:.0f}%", "kpi-green"),
            ("Programmé NON",    f"{kp['p_non']:.0f}%", "kpi-orange"),
        ]
        for col, (lbl, val, cls) in zip(cols_kpi2, mini_kpis):
            col.markdown(
                f'<div class="kpi-card {cls}" style="padding:0.8rem">'
                f'<div class="kpi-value" style="font-size:1.5rem">{val}</div>'
                f'<div class="kpi-label">{lbl}</div></div>',
                unsafe_allow_html=True
            )

    # INSIGHTS AUTOMATIQUES
    st.markdown('<div class="section-title">💡 Points clés à retenir</div>', unsafe_allow_html=True)

    exec_c = df_operations["ETAT_EXECUTION"].str.strip().value_counts()
    pct_acheve = exec_c.get("Achevé", 0) / len(df_operations) * 100
    gap_df = df_operations.dropna(subset=["COUT_NUM"]).copy()
    gap_df["NF"] = gap_df["ETAT_FINANCEMENT"].str.lower().str.strip().apply(
        lambda x: "non" in x and "acquis" not in x
    )
    pct_non_finance = gap_df["NF"].mean() * 100

    insights = []
    if pct_acheve < 30:
        insights.append(("warning", f"Seulement <b>{pct_acheve:.0f}%</b> des opérations sont achevées. Une grande partie du portefeuille est encore en cours ou non démarrée."))
    else:
        insights.append(("success", f"<b>{pct_acheve:.0f}%</b> des opérations sont marquées comme achevées."))
    if pct_non_finance > 20:
        insights.append(("warning", f"<b>{pct_non_finance:.0f}%</b> des opérations semblent non financées — un gap budgétaire significatif à surveiller."))
    insights.append(("info", f"Le pipeline IA a identifié <b>{K_OPT} groupes thématiques</b> de projets (score de cohérence Silhouette : <b>{sil:.3f}</b>)."))
    if kb["p_oui"] < 60:
        insights.append(("warning", f"Moins de 60% des opérations sont budgétisées (<b>{kb['p_oui']:.0f}%</b>). Cela peut impacter la réalisation du programme."))

    for typ, msg in insights:
        cls = "warning" if typ == "warning" else ("success" if typ == "success" else "")
        st.markdown(f'<div class="insight-box {cls}">{msg}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ONGLET 2 — PROJETS & GROUPES
# ══════════════════════════════════════════════════════════════════════════════
with tab_clusters:
    st.markdown('<div class="section-title">🗺️ Carte des projets</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
    Chaque point représente un projet. Les projets proches sur la carte partagent des intitulés similaires.
    Les couleurs indiquent le groupe d'appartenance. Survolez un point pour voir les détails.
    </div>""", unsafe_allow_html=True)

    st.plotly_chart(
        plot_clusters_scatter(df_projets, res["Z_2d"], res["methode_proj"], K_OPT, noms_clusters),
        use_container_width=True,
        key="clusters_scatter"
    )

    # Résumé des groupes
    st.markdown('<div class="section-title">📂 Composition des groupes</div>', unsafe_allow_html=True)
    for c in sorted(df_projets["CLUSTER"].unique()):
        c = int(c)
        grp = df_projets[df_projets["CLUSTER"] == c]
        nom = noms_clusters.get(c, f"Groupe {c}")
        conf_moy = grp["CONFIANCE"].mean()
        coul = PALETTE[c % len(PALETTE)]

        with st.expander(f"Groupe {c} — {nom}  ({len(grp)} projet(s)) · Confiance : {conf_moy:.0%}", expanded=(c == 0)):
            for _, r in grp.iterrows():
                st.markdown(
                    f"<div style='display:flex;gap:1rem;align-items:center;padding:0.5rem 0;"
                    f"border-bottom:1px solid #21262d'>"
                    f"<span style='font-family:monospace;font-size:0.82rem;color:{coul};min-width:130px'>{r['ID_PROJET']}</span>"
                    f"<span style='color:#c9d1d9;font-size:0.9rem'>{r['INTITULE_PROJET']}</span>"
                    f"<span style='margin-left:auto;font-size:0.8rem;color:#8b949e'>{r['CONFIANCE']:.0%}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # Heatmap Q
    st.markdown('<div class="section-title">🎯 Matrice de probabilité d\'appartenance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
    Cette matrice montre la probabilité que chaque projet appartienne à chaque groupe.
    Une valeur élevée (proche de 1) signifie que le projet est clairement dans ce groupe.
    Une valeur faible indique un projet à la frontière entre plusieurs groupes.
    </div>""", unsafe_allow_html=True)
    st.plotly_chart(
        plot_heatmap_Q(df_projets, res["Q_fin_np"], K_OPT),
        use_container_width=True,
        key="clusters_heatmap_q"
    )

    # Radar par cluster
    st.markdown('<div class="section-title">🕸️ Profil opérationnel par groupe</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_radar(df_ops_final, noms_clusters, K_OPT), use_container_width=True, key="clusters_radar")
    st.markdown("""
    <div class="insight-box">
    Chaque axe représente une dimension opérationnelle (0 = mauvais, 1 = excellent).
    Un groupe avec un polygone large et régulier présente un bon profil sur tous les critères.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ONGLET 3 — FINANCES
# ══════════════════════════════════════════════════════════════════════════════
with tab_finances:

    # KPI financiers
    df_c = df_operations.dropna(subset=["COUT_NUM"])
    c_total = df_c["COUT_NUM"].sum()
    c_max   = df_c["COUT_NUM"].max()
    c_min   = df_c["COUT_NUM"].min()
    c_med   = df_c["COUT_NUM"].median()

    cols_f = st.columns(4)
    fin_kpis = [
        ("Coût total portefeuille", f"{c_total:,.0f}", "💎", "kpi-purple"),
        ("Opération la + chère",    f"{c_max:,.0f}",   "📈", "kpi-orange"),
        ("Opération la - chère",    f"{c_min:,.0f}",   "📉", "kpi-green"),
        ("Coût médian",             f"{c_med:,.0f}",   "⚖️",  "kpi-blue"),
    ]
    for col, (label, value, icon, cls) in zip(cols_f, fin_kpis):
        col.markdown(render_kpi(f"{label}<br>(KFCFA)", value, icon, cls), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Top 5
    fig_top, top5p, top5m = plot_top5(df_operations)
    st.plotly_chart(fig_top, use_container_width=True, key="finances_top5")

    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown("**🔴 Les 5 opérations les plus coûteuses**")
        st.dataframe(
            top5p[["ID_OPERATION","INTITULE_OPERATION","COUT_NUM","ETAT_EXECUTION"]]
            .rename(columns={"COUT_NUM":"Coût (KFCFA)","ETAT_EXECUTION":"Exécution"})
            .style.format({"Coût (KFCFA)": "{:,.0f}"}),
            hide_index=True, use_container_width=True
        )
    with col_f:
        st.markdown("**🟢 Les 5 opérations les moins coûteuses**")
        st.dataframe(
            top5m[["ID_OPERATION","INTITULE_OPERATION","COUT_NUM","ETAT_EXECUTION"]]
            .rename(columns={"COUT_NUM":"Coût (KFCFA)","ETAT_EXECUTION":"Exécution"})
            .style.format({"Coût (KFCFA)": "{:,.0f}"}),
            hide_index=True, use_container_width=True
        )

    st.markdown("---")
    # Gap financement
    fig_gap, agg_fin = plot_gap_financement(df_operations)
    st.plotly_chart(fig_gap, use_container_width=True, key="finances_gap")

    # Résumé gap
    total_global = agg_fin["total"].sum()
    gap_non = agg_fin[agg_fin["RISQUE"] == "🔴 Non financé"]["total"].sum()
    gap_par = agg_fin[agg_fin["RISQUE"] == "⚠ Partiel"]["total"].sum()
    exposition = gap_non + gap_par

    col_g1, col_g2, col_g3 = st.columns(3)
    col_g1.markdown(render_kpi("Gap non financé (KFCFA)", f"{gap_non:,.0f}", "🔴", "kpi-orange"), unsafe_allow_html=True)
    col_g2.markdown(render_kpi("Gap partiel (KFCFA)", f"{gap_par:,.0f}", "⚠️", "kpi-purple"), unsafe_allow_html=True)
    col_g3.markdown(render_kpi("Exposition totale", f"{exposition/total_global*100:.0f}%", "📊", "kpi-blue"), unsafe_allow_html=True)

    st.markdown("---")
    # Boxplot
    st.plotly_chart(plot_boxplot_couts(df_operations), use_container_width=True, key="finances_boxplot_couts")
    st.markdown("""
    <div class="insight-box">
    Ce graphique montre comment les coûts sont répartis selon l'état d'avancement des travaux.
    La ligne centrale représente la valeur médiane (50% des opérations sont en dessous).
    Les points isolés sont des opérations atypiques (très chères ou très bon marché).
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    # Coûts par cluster
    st.plotly_chart(plot_couts_par_cluster(df_ops_final, noms_clusters, K_OPT), use_container_width=True, key="finances_couts_par_cluster")


# ══════════════════════════════════════════════════════════════════════════════
#  ONGLET 4 — OPÉRATIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_operations:

    st.plotly_chart(plot_histogrammes_etats(df_operations), use_container_width=True, key="operations_histogrammes_etats")
    st.plotly_chart(plot_donut_execution(df_operations), use_container_width=True, key="operations_donut_execution")

    st.markdown("---")
    st.plotly_chart(plot_croisement_maturite(df_operations), use_container_width=True, key="operations_croisement_maturite")
    st.markdown("""
    <div class="insight-box">
    Ce tableau croisé répond à la question : <b>les projets validés (VISA) sont-ils mieux exécutés ?</b>
    Chaque cellule indique combien d'opérations se trouvent à l'intersection de deux états.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    fig_bp2, kb2, kp2 = plot_kpi_budgetise_programme(df_operations)
    st.plotly_chart(fig_bp2, use_container_width=True, key="operations_kpi_budgetise_programme")

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("**Budgétisation**")
        st.markdown(f"""
        <div class="insight-box {'success' if kb2['p_oui'] >= 70 else 'warning'}">
        <b>{kb2['n_oui']}</b> opérations budgétisées ({kb2['p_oui']:.1f}%)<br>
        <b>{kb2['n_non']}</b> opérations non budgétisées ({kb2['p_non']:.1f}%)
        </div>""", unsafe_allow_html=True)
    with col_h2:
        st.markdown("**Programmation**")
        st.markdown(f"""
        <div class="insight-box {'success' if kp2['p_oui'] >= 70 else 'warning'}">
        <b>{kp2['n_oui']}</b> opérations programmées ({kp2['p_oui']:.1f}%)<br>
        <b>{kp2['n_non']}</b> opérations non programmées ({kp2['p_non']:.1f}%)
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  ONGLET 5 — MODÈLE IA
# ══════════════════════════════════════════════════════════════════════════════
with tab_modele:

    st.markdown("""
    <div class="insight-box">
    <b>Comment fonctionne l'intelligence artificielle ?</b><br>
    Le système utilise un modèle de langage (CamemBERT) pour comprendre le sens des intitulés de projets,
    puis un réseau de neurones (IDEC) pour regrouper automatiquement les projets similaires.
    Le nombre de groupes est déterminé automatiquement par la <b>méthode du coude</b>.
    </div>""", unsafe_allow_html=True)

    # Méthode du coude
    st.markdown('<div class="section-title">📐 Choix automatique du nombre de groupes</div>', unsafe_allow_html=True)
    st.plotly_chart(plot_elbow(res["k_vals"], res["inerties"], K_OPT), use_container_width=True, key="modele_elbow")
    st.markdown(f"""
    <div class="insight-box success">
    Le <b>coude de la courbe</b> se situe à <b>K = {K_OPT}</b>.
    C'est le point où ajouter un groupe supplémentaire n'améliore plus significativement la qualité du regroupement.
    <br><br>
    <b>Interprétation :</b> L'inertie mesure à quel point les projets sont dispersés à l'intérieur de chaque groupe.
    Plus elle est faible, plus les groupes sont compacts et cohérents.
    </div>""", unsafe_allow_html=True)

    # Courbes d'apprentissage
    st.markdown('<div class="section-title">📉 Apprentissage du modèle</div>', unsafe_allow_html=True)
    st.plotly_chart(
        plot_training_curves(res["hist_pre"], res["hist_total"], res["hist_mse_r"], res["hist_kl"]),
        use_container_width=True,
        key="modele_training_curves"
    )

    # Métriques qualité
    st.markdown('<div class="section-title">🏆 Qualité du regroupement</div>', unsafe_allow_html=True)
    db  = res["db"]
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        cls_sil = "kpi-green" if sil > 0.25 else "kpi-orange"
        st.markdown(render_kpi("Score Silhouette", f"{sil:.3f}", "⭐", cls_sil,
                               "✅ Bon" if sil > 0.25 else "⚠ Faible"), unsafe_allow_html=True)
        if sil > 0.50:
            msg = "Excellent — Les groupes sont très bien séparés et cohésifs."
        elif sil > 0.25:
            msg = "Acceptable — La structure est raisonnable pour ce type de données."
        else:
            msg = "Faible — Les groupes se chevauchent. Essayez un autre K."
        st.markdown(f'<div class="insight-box">📊 <b>Interprétation :</b> {msg}</div>', unsafe_allow_html=True)

    with col_m2:
        cls_db = "kpi-green" if db < 1.5 else "kpi-orange"
        st.markdown(render_kpi("Indice Davies-Bouldin", f"{db:.3f}", "🎯", cls_db,
                               "✅ Bon" if db < 1.5 else "⚠ À surveiller"), unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">📊 <b>Interprétation :</b> Plus cet indice est bas, plus les groupes sont
        distincts les uns des autres. Un score inférieur à 1.5 est considéré comme satisfaisant.
        </div>""", unsafe_allow_html=True)

    # Architecture technique
    with st.expander("⚙️ Détails techniques du modèle (pour les curieux)"):
        st.code(f"""
Modèle de langage : CamemBERT (camembert-base)
  - Dimension des vecteurs : 768
  - Stratégie : Mean pooling pondéré par le masque d'attention

Auto-encodeur (IDEC) :
  Encodeur : 768 → 256 → 128 → {HP['dim_latente']} (espace latent)
  Décodeur : {HP['dim_latente']} → 128 → 256 → 768

Hyperparamètres :
  - Dropout        : {HP['dropout']}
  - LR pré-entr.   : {HP['lr_pretrain']}
  - LR raffinement : {HP['lr_raffinement']}
  - Epochs pré     : {HP['epochs_pretrain']}
  - Epochs IDEC    : {HP['epochs_raffinement']}
  - Gamma (KL)     : {HP['gamma']}
  - Batch size     : {HP['batch_size']}

Sélection K : Méthode du coude (distance perpendiculaire max)
K retenu     : {K_OPT}
        """, language="yaml")


# ══════════════════════════════════════════════════════════════════════════════
#  ONGLET 6 — DONNÉES BRUTES
# ══════════════════════════════════════════════════════════════════════════════
with tab_donnees:

    st.markdown('<div class="section-title">🗃️ Projets</div>', unsafe_allow_html=True)

    search_p = st.text_input("🔍 Rechercher un projet (ID ou intitulé)", key="search_projets")
    df_p_display = df_projets.copy()
    df_p_display["NOM_GROUPE"] = df_p_display.apply(
        lambda r: f"Groupe {int(r['CLUSTER'])} — {noms_clusters.get(int(r['CLUSTER']),'')}",
        axis=1
    )
    df_p_display["CONFIANCE"] = df_p_display["CONFIANCE"].apply(lambda x: f"{x:.0%}")
    if search_p:
        mask = (
            df_p_display["ID_PROJET"].str.contains(search_p, case=False, na=False) |
            df_p_display["INTITULE_PROJET"].str.contains(search_p, case=False, na=False)
        )
        df_p_display = df_p_display[mask]

    st.dataframe(
        df_p_display[["ID_PROJET","INTITULE_PROJET","NOM_GROUPE","CONFIANCE"]],
        hide_index=True, use_container_width=True, height=350
    )

    st.markdown('<div class="section-title">📋 Opérations</div>', unsafe_allow_html=True)

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        etats_exec_opts = ["Tous"] + df_operations["ETAT_EXECUTION"].str.strip().unique().tolist()
        filtre_exec = st.selectbox("Filtrer par état d'exécution", etats_exec_opts)
    with col_filter2:
        search_o = st.text_input("🔍 Rechercher une opération", key="search_ops")

    df_o_display = df_operations.copy()
    if filtre_exec != "Tous":
        df_o_display = df_o_display[df_o_display["ETAT_EXECUTION"].str.strip() == filtre_exec]
    if search_o:
        mask_o = (
            df_o_display["ID_OPERATION"].str.contains(search_o, case=False, na=False) |
            df_o_display["INTITULE_OPERATION"].str.contains(search_o, case=False, na=False)
        )
        df_o_display = df_o_display[mask_o]

    cols_display = ["ID_OPERATION","INTITULE_OPERATION","COUT_NUM","ETAT_MATURITE",
                    "ETAT_EXECUTION","ETAT_FINANCEMENT","BUDGETISE","PROGRAMME"]
    st.dataframe(
        df_o_display[cols_display].rename(columns={"COUT_NUM":"Coût (KFCFA)"})
        .style.format({"Coût (KFCFA)": "{:,.0f}"}),
        hide_index=True, use_container_width=True, height=400
    )

    st.markdown("---")
    # Téléchargement
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_p = df_projets.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("⬇️ Télécharger les projets (CSV)", csv_p,
                           "projets_magnus.csv", "text/csv", use_container_width=True)
    with col_dl2:
        csv_o = df_operations.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("⬇️ Télécharger les opérations (CSV)", csv_o,
                           "operations_magnus.csv", "text/csv", use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#8b949e; font-size:0.78rem; padding:0.5rem 0 1rem">
    Magnus Entreprise · Banque de Projets · Pipeline IA : CamemBERT + IDEC + Méthode du Coude
</div>
""", unsafe_allow_html=True)
