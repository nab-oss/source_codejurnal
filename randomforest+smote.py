# ============================================================
# Judul:
# Klasifikasi Persetujuan Pinjaman Menggunakan Random Forest
# dengan Penanganan Ketidakseimbangan Kelas Berbasis SMOTE
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


warnings.filterwarnings("ignore")


# ============================================================
# 1. Konfigurasi Awal
# ============================================================

RANDOM_STATE = 42

DATA_PATH = "loan_risk_prediction_dataset.csv"

OUTPUT_DIR = "output_loan_rf_smote"
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. Fungsi Bantuan
# ============================================================

# ── Tema akademik global ────────────────────────────────────────────────────
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

ACEDEMIC_PALETTE = ["#2E4057", "#D62839", "#4A7C59", "#F4A261", "#6A4C93"]
ACEDEMIC_CMAP   = "Blues"
ACEDEMIC_DIV    = "RdBu_r"

plt.rcParams.update({
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "#F8F9FA",
    "axes.edgecolor"    : "#CCCCCC",
    "axes.linewidth"    : 0.8,
    "axes.grid"         : True,
    "grid.color"        : "#E0E0E0",
    "grid.linewidth"    : 0.6,
    "grid.linestyle"    : "--",
    "axes.titlesize"    : 13,
    "axes.titleweight"  : "bold",
    "axes.titlepad"     : 10,
    "axes.labelsize"    : 11,
    "axes.labelweight"  : "bold",
    "axes.labelpad"     : 6,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
    "legend.title_fontsize": 10,
    "legend.frameon"    : True,
    "legend.framealpha" : 0.9,
    "legend.edgecolor"  : "#CCCCCC",
    "figure.dpi"        : 150,
    "savefig.facecolor" : "white",
    "font.family"       : "DejaVu Sans",
})
# ────────────────────────────────────────────────────────────────────────────


def save_plot(filename):
    path = os.path.join(FIGURE_DIR, filename)
    plt.tight_layout(pad=1.5)
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def make_onehot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def evaluate_model(model_name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    result = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob)
    }

    report_df = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).transpose()

    return result, report_df, y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred, title, filename):
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.grid(False)  # Nonaktifkan grid agar tidak menutupi kotak CM
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Tidak Disetujui", "Disetujui"]
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > thresh else "#2E4057"
            )

    ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
           xticklabels=classes, yticklabels=classes)
    ax.set_xlabel("Prediksi", fontsize=11, fontweight="bold")
    ax.set_ylabel("Aktual", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    fig.patch.set_facecolor("white")
    save_plot(filename)


def get_feature_names(preprocessor, numeric_features, categorical_features):
    feature_names = []

    feature_names.extend(numeric_features)

    cat_pipeline = preprocessor.named_transformers_["cat"]
    onehot = cat_pipeline.named_steps["onehot"]

    try:
        cat_names = onehot.get_feature_names_out(categorical_features)
    except AttributeError:
        cat_names = onehot.get_feature_names(categorical_features)

    feature_names.extend(list(cat_names))

    return feature_names


# ============================================================
# 3. Load Dataset
# ============================================================

df = pd.read_csv(DATA_PATH)

print("Dataset berhasil dibaca.")
print("Ukuran dataset:", df.shape)
print(df.head())


# ============================================================
# 4. Validasi Kolom
# ============================================================

target_col = "LoanApproved"

expected_columns = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "YearsExperience",
    "Gender",
    "Education",
    "City",
    "EmploymentType",
    "LoanApproved"
]

missing_columns = [col for col in expected_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"Kolom berikut tidak ditemukan: {missing_columns}")

if target_col not in df.columns:
    raise ValueError("Kolom target LoanApproved tidak ditemukan.")


# ============================================================
# 5. Data Understanding
# ============================================================

dataset_shape = pd.DataFrame({
    "Keterangan": ["Jumlah Baris", "Jumlah Kolom"],
    "Nilai": [df.shape[0], df.shape[1]]
})
dataset_shape.to_csv(os.path.join(TABLE_DIR, "01_dataset_shape.csv"), index=False)

data_info = pd.DataFrame({
    "Nama_Kolom": df.columns,
    "Tipe_Data": df.dtypes.astype(str).values,
    "Jumlah_Missing_Value": df.isnull().sum().values,
    "Persentase_Missing_Value": (df.isnull().mean().values * 100).round(2),
    "Jumlah_Unique": df.nunique().values
})
data_info.to_csv(os.path.join(TABLE_DIR, "02_data_info.csv"), index=False)

target_distribution = df[target_col].value_counts().reset_index()
target_distribution.columns = ["Kelas", "Jumlah"]
target_distribution["Persentase"] = (
    target_distribution["Jumlah"] / len(df) * 100
).round(2)
target_distribution.to_csv(
    os.path.join(TABLE_DIR, "03_target_distribution.csv"),
    index=False
)

duplicate_count = pd.DataFrame({
    "Keterangan": ["Jumlah Data Duplikat"],
    "Nilai": [df.duplicated().sum()]
})
duplicate_count.to_csv(os.path.join(TABLE_DIR, "04_duplicate_count.csv"), index=False)

print("\nInformasi data disimpan ke folder tables.")


# ============================================================
# 6. Deskripsi Variabel Penelitian
# ============================================================

variable_description = pd.DataFrame({
    "Variabel": [
        "Age",
        "Income",
        "LoanAmount",
        "CreditScore",
        "YearsExperience",
        "Gender",
        "Education",
        "City",
        "EmploymentType",
        "LoanApproved"
    ],
    "Jenis_Data": [
        "Numerik",
        "Numerik",
        "Numerik",
        "Numerik",
        "Numerik",
        "Kategorikal",
        "Kategorikal",
        "Kategorikal",
        "Kategorikal",
        "Target"
    ],
    "Peran": [
        "Fitur",
        "Fitur",
        "Fitur",
        "Fitur",
        "Fitur",
        "Fitur",
        "Fitur",
        "Fitur",
        "Fitur",
        "Label"
    ],
    "Keterangan": [
        "Usia pemohon pinjaman",
        "Pendapatan pemohon",
        "Jumlah pinjaman yang diajukan",
        "Skor kredit pemohon",
        "Lama pengalaman kerja",
        "Jenis kelamin pemohon",
        "Tingkat pendidikan pemohon",
        "Kota domisili pemohon",
        "Jenis pekerjaan pemohon",
        "Status persetujuan pinjaman, 0 tidak disetujui dan 1 disetujui"
    ]
})

variable_description.to_csv(
    os.path.join(TABLE_DIR, "05_variable_description.csv"),
    index=False
)


# ============================================================
# 7. Exploratory Data Analysis
# ============================================================

numeric_features = [
    "Age",
    "Income",
    "LoanAmount",
    "CreditScore",
    "YearsExperience"
]

categorical_features = [
    "Gender",
    "Education",
    "City",
    "EmploymentType"
]


# 7.1 Grafik distribusi target
fig, ax = plt.subplots(figsize=(6, 5))
counts = df[target_col].value_counts().sort_index()
bars = ax.bar(
    ["Tidak Disetujui (0)", "Disetujui (1)"],
    counts.values,
    color=["#2E4057", "#D62839"],
    edgecolor="white", linewidth=1.5, width=0.5
)
for bar, val in zip(bars, counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + counts.max() * 0.01,
        f"{val:,}\n({val/len(df)*100:.1f}%)",
        ha="center", va="bottom", fontsize=10, fontweight="bold", color="#2E4057"
    )
ax.set_title("Distribusi Kelas LoanApproved", fontsize=13, fontweight="bold")
ax.set_xlabel("Kelas LoanApproved", fontsize=11, fontweight="bold")
ax.set_ylabel("Jumlah Data", fontsize=11, fontweight="bold")
ax.set_ylim(0, counts.max() * 1.18)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.patch.set_facecolor("white")
save_plot("01_distribusi_target.png")


# 7.2 Grafik missing value
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
if len(missing_values) > 0:
    colors = plt.cm.Blues_r(np.linspace(0.3, 0.8, len(missing_values)))
    bars = ax.bar(missing_values.index, missing_values.values, color=colors, edgecolor="white", linewidth=1)
    for bar, val in zip(bars, missing_values.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(val), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Jumlah Missing Value per Kolom", fontsize=13, fontweight="bold")
    ax.set_xlabel("Kolom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Jumlah Missing Value", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
else:
    ax.text(0.5, 0.5, "Tidak Ada Missing Value",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color="#4A7C59", transform=ax.transAxes)
    ax.set_title("Missing Value per Kolom", fontsize=13, fontweight="bold")
    ax.tick_params(left=False, bottom=False)
    ax.set_xticks([]); ax.set_yticks([])
fig.patch.set_facecolor("white")
save_plot("02_missing_value.png")


# 7.3 Histogram fitur numerik
for col in numeric_features:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(data=df, x=col, kde=True, bins=30,
                 color="#2E4057", edgecolor="white", linewidth=0.5,
                 line_kws={"color": "#D62839", "linewidth": 2},
                 ax=ax)
    ax.set_title(f"Distribusi {col}", fontsize=13, fontweight="bold")
    ax.set_xlabel(col, fontsize=11, fontweight="bold")
    ax.set_ylabel("Frekuensi", fontsize=11, fontweight="bold")
    mean_val = df[col].mean()
    ax.axvline(mean_val, color="#F4A261", linewidth=2, linestyle="--",
               label=f"Mean = {mean_val:.2f}")
    ax.legend(fontsize=9)
    fig.patch.set_facecolor("white")
    save_plot(f"03_histogram_{col}.png")


# 7.4 Boxplot fitur numerik terhadap target
for col in numeric_features:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=df, x=target_col, y=col,
        palette=["#2E4057", "#D62839"],
        width=0.5, linewidth=1.2,
        flierprops=dict(marker="o", markersize=3, alpha=0.5,
                        markerfacecolor="#F4A261", markeredgecolor="none"),
        ax=ax
    )
    ax.set_xticklabels(["Tidak Disetujui (0)", "Disetujui (1)"])
    ax.set_title(f"Boxplot {col} berdasarkan Status Pinjaman",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Status Pinjaman", fontsize=11, fontweight="bold")
    ax.set_ylabel(col, fontsize=11, fontweight="bold")
    fig.patch.set_facecolor("white")
    save_plot(f"04_boxplot_{col}_by_target.png")


# 7.5 Heatmap korelasi numerik
fig, ax = plt.subplots(figsize=(8, 6))
corr = df[numeric_features + [target_col]].corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True
sns.heatmap(
    corr, annot=True, cmap="RdBu_r", fmt=".2f",
    vmin=-1, vmax=1, center=0,
    linewidths=0.5, linecolor="white",
    annot_kws={"size": 9, "weight": "bold"},
    square=True, ax=ax
)
ax.set_title("Heatmap Korelasi Fitur Numerik", fontsize=13, fontweight="bold")
ax.tick_params(axis="x", rotation=30)
ax.tick_params(axis="y", rotation=0)
fig.patch.set_facecolor("white")
save_plot("05_heatmap_korelasi.png")


# 7.6 Distribusi fitur kategorikal terhadap target
for col in categorical_features:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=df, x=col, hue=target_col,
        palette=["#2E4057", "#D62839"],
        edgecolor="white", linewidth=0.8, ax=ax
    )
    ax.set_title(f"Distribusi {col} berdasarkan Status Pinjaman",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel(col, fontsize=11, fontweight="bold")
    ax.set_ylabel("Jumlah Data", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, ["Tidak Disetujui (0)", "Disetujui (1)"],
              title="Status Pinjaman", fontsize=9, title_fontsize=9,
              bbox_to_anchor=(1.02, 1), loc='upper left')  # Pindahkan legend keluar agar tidak menimpa data
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.patch.set_facecolor("white")
    save_plot(f"06_countplot_{col}_by_target.png")


# 7.7 Approval rate per kategori
for col in categorical_features:
    approval_rate = (
        df.groupby(col)[target_col]
        .mean()
        .reset_index()
        .sort_values(target_col, ascending=False)
    )

    approval_rate.to_csv(
        os.path.join(TABLE_DIR, f"06_approval_rate_by_{col}.csv"),
        index=False
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    n = len(approval_rate)
    colors = plt.cm.Blues(np.linspace(0.4, 0.85, n))
    sns.barplot(
        data=approval_rate, x=col, y=target_col,
        palette=colors.tolist(), edgecolor="white", linewidth=0.8, ax=ax
    )
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center", va="bottom", fontsize=8, fontweight="bold", color="#2E4057"
        )
    ax.set_title(f"Rata-rata Approval Rate berdasarkan {col}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel(col, fontsize=11, fontweight="bold")
    ax.set_ylabel("Rata-rata Approval Rate", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.patch.set_facecolor("white")
    save_plot(f"07_approval_rate_by_{col}.png")


# 7.8 Statistik deskriptif
df[numeric_features].describe().transpose().to_csv(
    os.path.join(TABLE_DIR, "07_descriptive_statistics_numeric.csv")
)


# ============================================================
# 8. Pembersihan Data
# ============================================================

df_clean = df.copy()

before_duplicate = len(df_clean)
df_clean = df_clean.drop_duplicates()
after_duplicate = len(df_clean)

cleaning_summary = pd.DataFrame({
    "Proses": ["Hapus Duplikasi"],
    "Jumlah_Sebelum": [before_duplicate],
    "Jumlah_Sesudah": [after_duplicate],
    "Jumlah_Dihapus": [before_duplicate - after_duplicate]
})
cleaning_summary.to_csv(
    os.path.join(TABLE_DIR, "08_cleaning_summary.csv"),
    index=False
)


# ============================================================
# 9. Split Fitur dan Target
# ============================================================

X = df_clean.drop(columns=[target_col])
y = df_clean[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

split_distribution = pd.DataFrame({
    "Dataset": ["Train", "Train", "Test", "Test"],
    "Kelas": [
        0,
        1,
        0,
        1
    ],
    "Jumlah": [
        (y_train == 0).sum(),
        (y_train == 1).sum(),
        (y_test == 0).sum(),
        (y_test == 1).sum()
    ]
})
split_distribution["Persentase"] = split_distribution.groupby("Dataset")["Jumlah"].transform(
    lambda x: (x / x.sum() * 100).round(2)
)
split_distribution.to_csv(
    os.path.join(TABLE_DIR, "09_train_test_distribution.csv"),
    index=False
)


# ============================================================
# 10. Preprocessing Pipeline
# ============================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", make_onehot_encoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="drop"
)

preprocessing_summary = pd.DataFrame({
    "Jenis_Fitur": ["Numerik", "Kategorikal"],
    "Fitur": [
        ", ".join(numeric_features),
        ", ".join(categorical_features)
    ],
    "Perlakuan": [
        "Missing value diisi median",
        "Missing value diisi modus, lalu One Hot Encoding"
    ],
    "Alasan": [
        "Median lebih stabil terhadap nilai ekstrem",
        "Model membutuhkan input numerik"
    ]
})
preprocessing_summary.to_csv(
    os.path.join(TABLE_DIR, "10_preprocessing_summary.csv"),
    index=False
)


# ============================================================
# 11. Baseline Model Random Forest Tanpa SMOTE
# ============================================================

baseline_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=1
    ))
])

baseline_model.fit(X_train, y_train)

baseline_result, baseline_report, baseline_pred, baseline_prob = evaluate_model(
    "Random Forest Tanpa SMOTE",
    baseline_model,
    X_test,
    y_test
)

baseline_report.to_csv(
    os.path.join(TABLE_DIR, "11_classification_report_baseline.csv")
)

plot_confusion_matrix(
    y_test,
    baseline_pred,
    "Confusion Matrix Random Forest Tanpa SMOTE",
    "08_confusion_matrix_baseline.png"
)


# ============================================================
# 12. Random Forest dengan SMOTE
# ============================================================

smote_model = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=1
    ))
])

smote_model.fit(X_train, y_train)

smote_result, smote_report, smote_pred, smote_prob = evaluate_model(
    "Random Forest dengan SMOTE",
    smote_model,
    X_test,
    y_test
)

smote_report.to_csv(
    os.path.join(TABLE_DIR, "12_classification_report_smote.csv")
)

plot_confusion_matrix(
    y_test,
    smote_pred,
    "Confusion Matrix Random Forest dengan SMOTE",
    "09_confusion_matrix_smote.png"
)


# ============================================================
# 13. Distribusi Kelas Sebelum dan Sesudah SMOTE
# ============================================================

preprocessor_for_smote_check = clone(preprocessor)
X_train_processed = preprocessor_for_smote_check.fit_transform(X_train)

smote_checker = SMOTE(random_state=RANDOM_STATE)
X_train_smote_check, y_train_smote_check = smote_checker.fit_resample(
    X_train_processed,
    y_train
)

class_distribution_smote = pd.DataFrame({
    "Kondisi": [
        "Sebelum SMOTE",
        "Sebelum SMOTE",
        "Sesudah SMOTE",
        "Sesudah SMOTE"
    ],
    "Kelas": [
        0,
        1,
        0,
        1
    ],
    "Jumlah": [
        (y_train == 0).sum(),
        (y_train == 1).sum(),
        (y_train_smote_check == 0).sum(),
        (y_train_smote_check == 1).sum()
    ]
})

class_distribution_smote["Persentase"] = class_distribution_smote.groupby("Kondisi")["Jumlah"].transform(
    lambda x: (x / x.sum() * 100).round(2)
)

class_distribution_smote.to_csv(
    os.path.join(TABLE_DIR, "13_class_distribution_before_after_smote.csv"),
    index=False
)

fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(
    data=class_distribution_smote,
    x="Kondisi", y="Jumlah", hue="Kelas",
    palette=["#2E4057", "#D62839"],
    edgecolor="white", linewidth=0.8, ax=ax
)
for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(
            f"{int(p.get_height()):,}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#2E4057"
        )
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ["Kelas 0 (Tidak Disetujui)", "Kelas 1 (Disetujui)"],
          title="Kelas", fontsize=9, title_fontsize=9,
          bbox_to_anchor=(1.02, 1), loc='upper left')  # Pindahkan legend keluar agar tidak menimpa data
ax.set_title("Distribusi Kelas Sebelum dan Sesudah SMOTE",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Kondisi", fontsize=11, fontweight="bold")
ax.set_ylabel("Jumlah Data", fontsize=11, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.patch.set_facecolor("white")
save_plot("10_distribusi_sebelum_sesudah_smote.png")


# ============================================================
# 14. Perbandingan Awal Baseline dan SMOTE
# ============================================================

initial_comparison = pd.DataFrame([baseline_result, smote_result])
initial_comparison.to_csv(
    os.path.join(TABLE_DIR, "14_initial_model_comparison.csv"),
    index=False
)

fig, ax = plt.subplots(figsize=(9, 5))
comparison_melted = initial_comparison.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"],
    var_name="Metric",
    value_name="Score"
)
sns.barplot(
    data=comparison_melted, x="Metric", y="Score", hue="Model",
    palette=["#2E4057", "#D62839"],
    edgecolor="white", linewidth=0.8, ax=ax
)
for p in ax.patches:
    if p.get_height() > 0.01:
        ax.annotate(
            f"{p.get_height():.3f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center", va="bottom", fontsize=7, fontweight="bold", color="#2E4057"
        )
ax.set_title("Perbandingan Performa Model Awal", fontsize=13, fontweight="bold")
ax.set_xlabel("Metrik Evaluasi", fontsize=11, fontweight="bold")
ax.set_ylabel("Skor", fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.12)
ax.tick_params(axis="x", rotation=15)
ax.legend(title="Model", fontsize=9, title_fontsize=9,
          bbox_to_anchor=(1.02, 1), loc='upper left')  # Pindahkan legend keluar agar tidak menimpa label
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
fig.patch.set_facecolor("white")
save_plot("11_perbandingan_model_awal.png")


# ============================================================
# 15. ROC Curve dan Precision Recall Curve Model Awal
# ============================================================

fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(y_test, baseline_prob, name="Tanpa SMOTE",
                                  color="#2E4057", ax=ax)
RocCurveDisplay.from_predictions(y_test, smote_prob, name="Dengan SMOTE",
                                  color="#D62839", ax=ax)
ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Random (AUC = 0.500)")
ax.set_title("ROC Curve — Model Awal", fontsize=13, fontweight="bold")
ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
ax.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
fig.patch.set_facecolor("white")
save_plot("12_roc_curve_model_awal.png")

fig, ax = plt.subplots(figsize=(7, 6))
PrecisionRecallDisplay.from_predictions(y_test, baseline_prob, name="Tanpa SMOTE",
                                         color="#2E4057", ax=ax)
PrecisionRecallDisplay.from_predictions(y_test, smote_prob, name="Dengan SMOTE",
                                         color="#D62839", ax=ax)
ax.set_title("Precision-Recall Curve — Model Awal", fontsize=13, fontweight="bold")
ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower left")
fig.patch.set_facecolor("white")
save_plot("13_precision_recall_curve_model_awal.png")


# ============================================================
# 16. Hyperparameter Tuning Model Random Forest dengan SMOTE
# ============================================================

tuning_pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("classifier", RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=1          # gunakan 1 agar tidak looping warning di Python 3.14 + Windows
    ))
])

param_distributions = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [None, 5, 10, 15, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__bootstrap": [True, False]
}

cv = StratifiedKFold(
    n_splits=3,       # dikurangi 5→3 agar lebih cepat
    shuffle=True,
    random_state=RANDOM_STATE
)

random_search = RandomizedSearchCV(
    estimator=tuning_pipeline,
    param_distributions=param_distributions,
    n_iter=15,        # dikurangi 30→15 agar lebih cepat
    scoring="f1",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=1,         # gunakan 1 agar tidak looping warning di Python 3.14 + Windows
    verbose=1,
    return_train_score=True
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

best_params = pd.DataFrame({
    "Parameter": list(random_search.best_params_.keys()),
    "Nilai_Terbaik": list(random_search.best_params_.values())
})
best_params.to_csv(
    os.path.join(TABLE_DIR, "15_best_hyperparameters.csv"),
    index=False
)

cv_results = pd.DataFrame(random_search.cv_results_)
cv_results.to_csv(
    os.path.join(TABLE_DIR, "16_random_search_cv_results.csv"),
    index=False
)


# ============================================================
# 17. Evaluasi Model Terbaik
# ============================================================

best_result, best_report, best_pred, best_prob = evaluate_model(
    "Random Forest SMOTE Tuning",
    best_model,
    X_test,
    y_test
)

best_report.to_csv(
    os.path.join(TABLE_DIR, "17_classification_report_best_model.csv")
)

plot_confusion_matrix(
    y_test,
    best_pred,
    "Confusion Matrix Model Terbaik",
    "14_confusion_matrix_best_model.png"
)


# ============================================================
# 18. Perbandingan Semua Model
# ============================================================

final_comparison = pd.DataFrame([
    baseline_result,
    smote_result,
    best_result
])

final_comparison.to_csv(
    os.path.join(TABLE_DIR, "18_final_model_comparison.csv"),
    index=False
)

fig, ax = plt.subplots(figsize=(10, 5))
final_melted = final_comparison.melt(
    id_vars="Model",
    value_vars=["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"],
    var_name="Metric",
    value_name="Score"
)

sns.barplot(
    data=final_melted, x="Metric", y="Score", hue="Model",
    palette=["#2E4057", "#D62839", "#4A7C59"],
    edgecolor="white", linewidth=0.8, ax=ax
)
for p in ax.patches:
    if p.get_height() > 0.01:
        ax.annotate(
            f"{p.get_height():.3f}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center", va="bottom", fontsize=7, fontweight="bold", color="#1a1a2e"
        )
ax.set_title("Perbandingan Performa Seluruh Model", fontsize=13, fontweight="bold")
ax.set_xlabel("Metrik Evaluasi", fontsize=11, fontweight="bold")
ax.set_ylabel("Skor", fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.15)
ax.tick_params(axis="x", rotation=15)
ax.legend(title="Model", fontsize=9, title_fontsize=9,
          bbox_to_anchor=(1.02, 1), loc='upper left')  # Pindahkan legend keluar agar tidak menimpa label
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
fig.patch.set_facecolor("white")
save_plot("15_perbandingan_seluruh_model.png")


fig, ax = plt.subplots(figsize=(7, 6))
RocCurveDisplay.from_predictions(y_test, baseline_prob, name="Tanpa SMOTE",
                                  color="#2E4057", ax=ax)
RocCurveDisplay.from_predictions(y_test, smote_prob, name="Dengan SMOTE",
                                  color="#D62839", ax=ax)
RocCurveDisplay.from_predictions(y_test, best_prob, name="SMOTE + Tuning",
                                  color="#4A7C59", ax=ax)
ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Random (AUC = 0.500)")
ax.set_title("ROC Curve — Seluruh Model", fontsize=13, fontweight="bold")
ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
ax.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
fig.patch.set_facecolor("white")
save_plot("16_roc_curve_seluruh_model.png")


fig, ax = plt.subplots(figsize=(7, 6))
PrecisionRecallDisplay.from_predictions(y_test, baseline_prob, name="Tanpa SMOTE",
                                         color="#2E4057", ax=ax)
PrecisionRecallDisplay.from_predictions(y_test, smote_prob, name="Dengan SMOTE",
                                         color="#D62839", ax=ax)
PrecisionRecallDisplay.from_predictions(y_test, best_prob, name="SMOTE + Tuning",
                                         color="#4A7C59", ax=ax)
ax.set_title("Precision-Recall Curve — Seluruh Model", fontsize=13, fontweight="bold")
ax.set_xlabel("Recall", fontsize=11, fontweight="bold")
ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="lower left")
fig.patch.set_facecolor("white")
save_plot("17_precision_recall_curve_seluruh_model.png")


# ============================================================
# 19. Feature Importance Model Terbaik
# ============================================================

best_preprocessor = best_model.named_steps["preprocessor"]
best_classifier = best_model.named_steps["classifier"]

feature_names = get_feature_names(
    best_preprocessor,
    numeric_features,
    categorical_features
)

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": best_classifier.feature_importances_
}).sort_values("Importance", ascending=False)

feature_importance.to_csv(
    os.path.join(TABLE_DIR, "19_feature_importance.csv"),
    index=False
)

top_feature_importance = feature_importance.head(15)

fig, ax = plt.subplots(figsize=(9, 7))
n_feat = len(top_feature_importance)
colors = plt.cm.Blues_r(np.linspace(0.25, 0.85, n_feat))
bars = ax.barh(
    top_feature_importance["Feature"],
    top_feature_importance["Importance"],
    color=colors, edgecolor="white", linewidth=0.8
)
for bar, val in zip(bars, top_feature_importance["Importance"]):
    ax.text(
        val + 0.001, bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}", va="center", ha="left",
        fontsize=8, fontweight="bold", color="#2E4057"
    )
ax.invert_yaxis()
ax.set_title("Top 15 Feature Importance — Model Terbaik (RF + SMOTE + Tuning)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score (Mean Decrease Impurity)",
              fontsize=11, fontweight="bold")
ax.set_ylabel("Fitur", fontsize=11, fontweight="bold")
ax.set_xlim(0, top_feature_importance["Importance"].max() * 1.18)
fig.patch.set_facecolor("white")
save_plot("18_top_15_feature_importance.png")


# ============================================================
# 20. Simpan Model dan Objek Penting
# ============================================================

joblib.dump(
    baseline_model,
    os.path.join(MODEL_DIR, "baseline_random_forest.pkl")
)

joblib.dump(
    smote_model,
    os.path.join(MODEL_DIR, "random_forest_smote.pkl")
)

joblib.dump(
    best_model,
    os.path.join(MODEL_DIR, "best_random_forest_smote_tuning.pkl")
)


# ============================================================
# 21. Ringkasan Eksperimen untuk Artikel Jurnal
# ============================================================

best_model_name = final_comparison.sort_values(
    by="F1_Score",
    ascending=False
).iloc[0]["Model"]

best_f1 = final_comparison.sort_values(
    by="F1_Score",
    ascending=False
).iloc[0]["F1_Score"]

best_auc = final_comparison.sort_values(
    by="F1_Score",
    ascending=False
).iloc[0]["ROC_AUC"]

top_5_features = feature_importance.head(5)["Feature"].tolist()

research_summary = pd.DataFrame({
    "Aspek": [
        "Jumlah Data",
        "Jumlah Fitur",
        "Target",
        "Distribusi Kelas 0",
        "Distribusi Kelas 1",
        "Model Terbaik Berdasarkan F1 Score",
        "F1 Score Model Terbaik",
        "ROC AUC Model Terbaik",
        "Top 5 Feature Importance"
    ],
    "Hasil": [
        df_clean.shape[0],
        X.shape[1],
        target_col,
        int((y == 0).sum()),
        int((y == 1).sum()),
        best_model_name,
        round(best_f1, 4),
        round(best_auc, 4),
        ", ".join(top_5_features)
    ]
})

research_summary.to_csv(
    os.path.join(TABLE_DIR, "20_research_summary.csv"),
    index=False
)


# ============================================================
# 22. Teks Interpretasi Otomatis
# ============================================================

interpretation_text = f"""
RINGKASAN HASIL EKSPERIMEN

Dataset berisi {df_clean.shape[0]} data dan {X.shape[1]} fitur.
Target penelitian adalah {target_col}.

Distribusi target:
Kelas 0: {(y == 0).sum()} data.
Kelas 1: {(y == 1).sum()} data.

Model yang diuji:
1. Random Forest tanpa SMOTE.
2. Random Forest dengan SMOTE.
3. Random Forest dengan SMOTE dan hyperparameter tuning.

Model terbaik berdasarkan F1 Score adalah {best_model_name}.
Nilai F1 Score model terbaik adalah {best_f1:.4f}.
Nilai ROC AUC model terbaik adalah {best_auc:.4f}.

Lima fitur paling berpengaruh berdasarkan feature importance:
{", ".join(top_5_features)}

Catatan pembahasan:
Jika recall kelas 1 meningkat setelah SMOTE, maka SMOTE membantu model mengenali kelas minoritas.
Jika accuracy turun sedikit tetapi F1 Score atau recall kelas 1 meningkat, hasil tersebut tetap relevan untuk dataset tidak seimbang.
Untuk artikel jurnal, prioritaskan pembahasan pada F1 Score, recall kelas 1, ROC AUC, confusion matrix, dan feature importance.
"""

with open(os.path.join(OUTPUT_DIR, "ringkasan_hasil_eksperimen.txt"), "w", encoding="utf-8") as file:
    file.write(interpretation_text)


# ============================================================
# 23. Cetak Hasil Akhir
# ============================================================

print("\n============================================================")
print("EKSPERIMEN SELESAI")
print("============================================================")

print("\nPerbandingan Model:")
print(final_comparison)

print("\nParameter Terbaik:")
print(best_params)

print("\nTop 10 Feature Importance:")
print(feature_importance.head(10))

print("\nSemua output tersimpan di folder:")
print(OUTPUT_DIR)

print("\nFile penting:")
print("1. output_loan_rf_smote/tables/18_final_model_comparison.csv")
print("2. output_loan_rf_smote/tables/19_feature_importance.csv")
print("3. output_loan_rf_smote/tables/20_research_summary.csv")
print("4. output_loan_rf_smote/figures/")
print("5. output_loan_rf_smote/models/best_random_forest_smote_tuning.pkl")
print("6. output_loan_rf_smote/ringkasan_hasil_eksperimen.txt")