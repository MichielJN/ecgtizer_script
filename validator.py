import numpy as np
import pandas as pd
from pathlib import Path
import math

# === PARAMETERS ===
REF_CSV = Path(r"[original timeseries]")
DIG_CSV = Path(r"[digitized timeseries]")

FS = 500               # geschatte samplefrequentie (Hz) – pas aan indien nodig
MAX_SHIFT_SEC = 0.5    # maximaal 0,5s verschuiven, zoals in de Challenge
MIN_SAMPLES = 500      # minimaal aantal overlappende samples per lead
# ===================


def load_signals(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # eerste kolom weggooien als het tijd is
    if df.columns[0].lower() in ["time", "t", "timestamp"]:
        df = df.drop(columns=df.columns[0])
    return df


def best_alignment_metrics(ref: np.ndarray,
                           dig: np.ndarray,
                           fs: int = FS,
                           max_shift_sec: float = MAX_SHIFT_SEC,
                           min_samples: int = MIN_SAMPLES) -> dict:
    """
    Zoek over verschuivingen in de tijd (lags) naar de beste uitlijning.
    Voor elke lag:
      - selecteer overlappende stukken
      - verwijder NaN's
      - haal het gemiddelde (baseline) eraf
      - zoek beste schaalfactor a zodat x ≈ a * y
      - bereken SNR in dB en correlatie

    Retourneert de lag met hoogste SNR.
    """

    n = min(len(ref), len(dig))
    ref = ref[:n].astype(float)
    dig = dig[:n].astype(float)

    max_shift = int(max_shift_sec * fs)
    best_snr = -math.inf
    best_pcc = math.nan
    best_lag = 0
    best_n = 0
    best_cov = 0.0

    for lag in range(-max_shift, max_shift + 1):
        if lag < 0:
            x = ref[-lag:]
            y = dig[:len(x)]
        else:
            x = ref[:n - lag]
            y = dig[lag:lag + len(x)]

        if len(x) < min_samples:
            continue

        mask = ~np.isnan(x) & ~np.isnan(y)
        m = int(mask.sum())
        if m < min_samples:
            continue

        x_m = x[mask]
        y_m = y[mask]

        # baseline eraf
        x0 = x_m - np.mean(x_m)
        y0 = y_m - np.mean(y_m)

        # als y0 bijna nul is, overslaan
        denom = float(np.sum(y0 ** 2))
        if denom <= 0:
            continue

        # optimale schaalfactor a zodat SSE minimaal wordt
        a = float(np.sum(x0 * y0) / denom)
        y_hat = a * y0

        # signaal- en foutenergie
        sxx = float(np.sum(x0 ** 2))
        sse = float(np.sum((x0 - y_hat) ** 2))
        if sxx <= 0 or sse <= 0:
            continue

        snr = 10.0 * math.log10(sxx / sse)

        # PCC na schaal/baseline-correctie
        pcc = float(np.corrcoef(x0, y_hat)[0, 1])

        coverage = m / float(n)

        if snr > best_snr:
            best_snr = snr
            best_pcc = pcc
            best_lag = lag
            best_n = m
            best_cov = coverage

    if best_n == 0:
        return {
            "snr": math.nan,
            "raw_snr": math.nan,
            "pcc": math.nan,
            "lag": 0,
            "n": 0,
            "coverage": 0.0,
            "scale": math.nan,
        }

    # eenvoudige dekking-straf: vermenigvuldig met coverage
    snr_eff = best_snr * best_cov

    return {
        "snr": snr_eff,        # gestrafte SNR
        "raw_snr": best_snr,   # ruwe SNR zonder dekkingstraf
        "pcc": best_pcc,
        "lag": best_lag,
        "n": best_n,
        "coverage": best_cov,
        # schaalfactor is impliciet, maar je kunt hem bewaren als je wilt
    }


def main():
    ref_df = load_signals(REF_CSV)
    dig_df = load_signals(DIG_CSV)

    common_leads = sorted(set(ref_df.columns) & set(dig_df.columns))
    if not common_leads:
        print("Geen overlappende lead-namen tussen de twee CSV's.")
        print("Ref leads:", list(ref_df.columns))
        print("Dig leads:", list(dig_df.columns))
        return

    print("Lead | lag  | SNR_dB  | rawSNR  |  PCC      |   n   | cov")
    print("-----+------+---------+---------+----------+-------+------")

    snr_values = []
    for lead in common_leads:
        ref = ref_df[lead].to_numpy(dtype=float)
        dig = dig_df[lead].to_numpy(dtype=float)

        m = best_alignment_metrics(ref, dig)

        snr_str = "nan" if math.isnan(m["snr"]) else f"{m['snr']:.2f}"
        raw_str = "nan" if math.isnan(m["raw_snr"]) else f"{m['raw_snr']:.2f}"
        pcc_str = "nan" if math.isnan(m["pcc"]) else f"{m['pcc']:.4f}"
        cov_str = f"{m['coverage']:.2f}"

        if not math.isnan(m["snr"]):
            snr_values.append(m["snr"])

        print(
            f"{lead:4s} | {m['lag']:4d} | {snr_str:7s} | {raw_str:7s} | "
            f"{pcc_str:8s} | {m['n']:5d} | {cov_str}"
        )

    if snr_values:
        mean_snr = sum(snr_values) / len(snr_values)
        print(f"\nGemiddelde SNR over geldige leads: {mean_snr:.2f} dB")


if __name__ == "__main__":
    main()
