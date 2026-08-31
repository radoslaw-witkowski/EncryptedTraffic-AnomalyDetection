# =========================================================
# realtime_full_demo.py
# Near-real-time anomaly detection + anomaly family prediction
# =========================================================

import os
import time
import glob
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
import subprocess
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

PCAP_DIR = BASE_DIR / "data" / "realtime_stream"
TMP_DIR = BASE_DIR / "data" / "realtime_tmp"

FEATURE_SCHEMA_PATH = BASE_DIR / "feature_schema.csv"
SCALER_PATH = BASE_DIR / "standard_scaler.pkl"

AE_MODEL_PATH = BASE_DIR / "models" / "best_autoencoder_final.pth"
FAMILY_CLASSIFIER_PATH = BASE_DIR / "models" / "family_classifier_rf.pkl"

SLEEP_TIME = 2

# 95th percentile of reconstruction errors on the normal training dataset
THRESHOLD = 0.219239

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# MODEL AE
# =========================================================

class ImprovedAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# =========================================================
# LOAD FILES
# =========================================================

feature_columns = pd.read_csv(
    FEATURE_SCHEMA_PATH
)["feature"].tolist()

scaler = joblib.load(
    SCALER_PATH
)

family_classifier = joblib.load(
    FAMILY_CLASSIFIER_PATH
)

input_dim = len(feature_columns)

ae = ImprovedAutoencoder(input_dim).to(DEVICE)
ae.load_state_dict(
    torch.load(
    AE_MODEL_PATH,
    map_location=DEVICE
)
)

ae.eval()

print("=== REALTIME FULL DEMO ===")
print("Device:", DEVICE)
print("Liczba cech:", input_dim)


print("Threshold:", round(float(THRESHOLD), 6))

# =========================================================
# HELPERS
# =========================================================

RAW_COLUMNS = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "proto",
    "conn_state",
    "orig_pkts",
    "resp_pkts",
]

NUMERIC_COLUMNS = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "orig_pkts",
    "resp_pkts",
]

def read_zeek_conn(path):

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header = None
    rows = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if line.startswith("#fields"):
            header = line.split("\t")[1:]
            continue

        if line.startswith("#"):
            continue

        parts = line.split("\t")

        if header and len(parts) == len(header):
            rows.append(parts)

    if header is None:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=header)

def preprocess(df):

    if df.empty:
        return pd.DataFrame(columns=feature_columns)

    df = df[RAW_COLUMNS].copy()

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(
            df[col].replace("-", 0),
            errors="coerce"
        ).fillna(0)

    for col in ["duration", "orig_bytes", "resp_bytes"]:
        df[col] = np.log1p(df[col])

    df = pd.get_dummies(
        df,
        columns=["proto", "conn_state"],
        drop_first=True
    )

    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    scaled = scaler.transform(df)

    return pd.DataFrame(
        scaled,
        columns=feature_columns
    )

def run_zeek(pcap_path):
    name = os.path.basename(pcap_path).replace(".", "_")
    outdir = os.path.join(TMP_DIR, name)
    os.makedirs(outdir, exist_ok=True)

    # Remove old Zeek logs
    for file in os.listdir(outdir):
        if file.endswith(".log"):
            os.remove(os.path.join(outdir, file))

    pcap_abs = Path(pcap_path).resolve()
    outdir_abs = Path(outdir).resolve()

    # Convert Windows paths dynamically to WSL paths
    pcap_wsl = subprocess.run(
        ["wsl", "wslpath", "-a", str(pcap_abs)],
        capture_output=True,
        text=True,
        check=True
    ).stdout.strip()

    outdir_wsl = subprocess.run(
        ["wsl", "wslpath", "-a", str(outdir_abs)],
        capture_output=True,
        text=True,
        check=True
    ).stdout.strip()

    cmd = [
        "wsl",
        "bash",
        "-lc",
        f'cd "{outdir_wsl}" && /opt/zeek/bin/zeek -C -r "{pcap_wsl}"'
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print("[ZEEK ERROR]")
        print(result.stderr)

    return os.path.join(outdir, "conn.log")
# =========================================================
# MAIN LOOP
# =========================================================

os.makedirs(TMP_DIR, exist_ok=True)

processed = set()

print("\nStart monitorowania folderu:")
print(PCAP_DIR)

flow_global = 0

while True:

    pcaps = sorted(
        glob.glob(os.path.join(PCAP_DIR, "*.pcapng")) +
        glob.glob(os.path.join(PCAP_DIR, "*.pcap"))
    )

    for pcap in pcaps:

        if pcap in processed:
            continue

        print("\n" + "="*70)
        print("NOWY PLIK:", pcap)

        start_time = time.time()

        conn_path = run_zeek(pcap)

        if not os.path.exists(conn_path):
            print("Brak conn.log")
            processed.add(pcap)
            continue

        raw_df = read_zeek_conn(conn_path)

        if raw_df.empty:
            print("Pusty conn.log")
            processed.add(pcap)
            continue

        proc_df = preprocess(raw_df)

        x = torch.tensor(
            proc_df.values,
            dtype=torch.float32
        ).to(DEVICE)

        with torch.no_grad():

            embeddings = ae.encoder(x)

            recon = ae(x)

            errors = torch.max(
                (x - recon) ** 2,
                dim=1
            ).values

        errors_np = errors.cpu().numpy()
        embeddings_np = embeddings.cpu().numpy()

        alerts = 0

        for i in range(len(errors_np)):

            flow_global += 1

            score = float(errors_np[i])

            decision = (
                "ANOMALIA"
                if score > THRESHOLD
                else "OK"
            )

            proto = raw_df.iloc[i]["proto"]
            state = raw_df.iloc[i]["conn_state"]

            if decision == "ANOMALIA":

                alerts += 1

                pred = family_classifier.predict(
                    embeddings_np[i].reshape(1, -1)
                )[0]

                probs = family_classifier.predict_proba(
                    embeddings_np[i].reshape(1, -1)
                )[0]

                confidence = float(np.max(probs))

                print(
                    f"[ALERT] "
                    f"flow={flow_global:06d} | "
                    f"proto={proto} | "
                    f"state={state} | "
                    f"score={score:.6f} | "
                    f"predicted={pred} | "
                    f"confidence={confidence:.2f}"
                )

            else:

                print(
                    f"[ OK ] "
                    f"flow={flow_global:06d} | "
                    f"proto={proto} | "
                    f"state={state} | "
                    f"score={score:.6f}"
                )

        elapsed = time.time() - start_time

        throughput = len(errors_np) / max(elapsed, 0.0001)

        print("\n--- PODSUMOWANIE ---")
        print("Flowy:", len(errors_np))
        print("Alerty:", alerts)
        print("Czas:", round(elapsed, 3), "s")
        print("Throughput:", round(throughput, 2), "flow/s")

        processed.add(pcap)

    time.sleep(SLEEP_TIME)