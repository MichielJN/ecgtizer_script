import os
import sys
import argparse
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

import torch
import pdf2image
from pdf2image import convert_from_path as _orig_convert_from_path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_script_dir() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path(sys.argv[0]).resolve().parent


SCRIPT_DIR = get_script_dir()


def resolve_from_script_dir(p: str | Path) -> Path:
    p = Path(p)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return p.resolve()



POPLER_BIN = r"C:\Program Files\poppler\poppler-24.08.0\Library\bin"

if POPLER_BIN and POPLER_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = os.environ.get("PATH", "") + ";" + POPLER_BIN

Image.MAX_IMAGE_PIXELS = None


def _convert_from_path_with_poppler(*args, **kwargs):
    """Altijd met poppler_path=POPLER_BIN werken."""
    if "poppler_path" not in kwargs:
        kwargs["poppler_path"] = POPLER_BIN
    return _orig_convert_from_path(*args, **kwargs)


pdf2image.convert_from_path = _convert_from_path_with_poppler

from ecgtizer.ecgtizer import ECGtizer 


def local_name(tag: str) -> str:
    """Geeft de lokale tagnaam terug, zonder namespace."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def iter_sequences(root):
    """Itereert over alle <sequence>-elementen (ongeacht namespace)."""
    for el in root.iter():
        if local_name(el.tag) == "sequence":
            yield el


def find_child_by_localname(parent, name: str):
    """Zoekt eerste kind met gegeven lokale naam."""
    for child in parent:
        if local_name(child.tag) == name:
            return child
    return None


def get_sequence_code(seq) -> str:
    """
    Haalt de 'code' uit een <sequence>:

      <sequence>
        <code code="MDC_ECG_LEAD_I" .../>
        <value>...</value>
      </sequence>
    """
    for child in seq:
        if local_name(child.tag) == "code":
            return child.attrib.get("code", "")
    return ""


def png_to_pdf(png_path: Path, pdf_path: Path, dpi: int):
    """
    Zet een PNG-afbeelding om naar een één-pagina PDF
    (met simpele auto-crop der witte marge).
    """
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    Image.MAX_IMAGE_PIXELS = None

    img = Image.open(png_path).convert("RGB")

    try:
        gray = img.convert("L")
        mask = gray.point(lambda p: 255 if p < 250 else 0)
        bbox = mask.getbbox()
        if bbox:
            img = img.crop(bbox)
    except Exception:
        pass

    img.save(pdf_path, resolution=dpi)


def summarize_ecg(ecg: ECGtizer, label: str = ""):
    """Drukt korte info over gevonden leads af."""
    leads = getattr(ecg, "extracted_lead", None)

    if not isinstance(leads, dict):
        print(f"[INFO] {label} ECGtizer-extractie bevat geen 'extracted_lead'-dict.")
        return

    lead_names = sorted(leads.keys())
    print(
        f"[INFO] {label} ECGtizer vond {len(lead_names)} lead(s): "
        + ", ".join(lead_names),
    )

    lengths = {}
    for name, arr in leads.items():
        try:
            lengths[name] = len(arr)
        except TypeError:
            pass

    if lengths:
        min_len = min(lengths.values())
        max_len = max(lengths.values())
        print(
            f"[INFO] {label} lengte per lead (samples): "
            f"min={min_len}, max={max_len}",
        )


def run_ecgtizer_multi(
    pdf_path: Path,
    base_dpi: int,
    base_method: str,
    debug: bool,
) -> ECGtizer:
    """
    Probeert ECGtizer met meerdere combinaties van DPI en extractiemethode.

    DPI-kandidaten: base_dpi, 500, 300, 400
    Methoden: base_method, fragmented, full, lazy
    """
    dpi_candidates: list[int] = []
    for d in (base_dpi, 500, 300, 400):
        if d is not None and d > 0 and d not in dpi_candidates:
            dpi_candidates.append(d)

    method_candidates: list[str] = []
    for m in (base_method, "fragmented", "full", "lazy"):
        if m is not None and m not in method_candidates:
            method_candidates.append(m)

    last_exc: Exception | None = None

    for dpi in dpi_candidates:
        for m in method_candidates:
            try:
                print(
                    f"[INFO] ECGtizer voor {pdf_path} met methode '{m}' "
                    f"en dpi={dpi}",
                )
                ecg = ECGtizer(
                    str(pdf_path),
                    dpi=dpi,
                    extraction_method=m,
                    verbose=debug,
                    DEBUG=debug,
                )
                print(
                    f"[INFO] ECGtizer slaagde met methode '{m}' en dpi={dpi}",
                )
                summarize_ecg(ecg, label="(na extractie)")
                return ecg
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                print(
                    f"[WAARSCHUWING] Methode '{m}' faalde "
                    f"voor {pdf_path} bij dpi={dpi}: {msg}",
                )
                if "index 0 is out of bounds" in msg:
                    print(
                        "[TIP] Lege interne array in ECGtizer; "
                        "mogelijk geen bruikbare leads gevonden "
                        "(DPI/layout der ECG wijkt af).",
                    )
                if debug:
                    traceback.print_exc()
                last_exc = e

    assert last_exc is not None
    raise last_exc

def parse_aecg_xml(xml_path: Path):
    """
    Leest ECGtizer-XML (HL7 AnnotatedECG) en geeft terug:
      - tijdvector (seconden, op basis van TIME_ABSOLUTE increment)
      - dict van leads: naam -> numpy-array (µV)
      - dt (seconden)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    dt = None
    for seq in iter_sequences(root):
        code = get_sequence_code(seq)
        if code == "TIME_ABSOLUTE":
            val = find_child_by_localname(seq, "value")
            if val is None:
                continue
            incr = find_child_by_localname(val, "increment")
            if incr is not None:
                dt = float(incr.attrib.get("value"))
                break

    if dt is None:
        raise ValueError("Geen TIME_ABSOLUTE increment (dt) gevonden in XML.")

    leads: dict[str, np.ndarray] = {}

    for seq in iter_sequences(root):
        code = get_sequence_code(seq)
        if not code.startswith("MDC_ECG_LEAD_"):
            continue

        lead_name = code.replace("MDC_ECG_LEAD_", "")

        val = find_child_by_localname(seq, "value")
        if val is None:
            continue

        origin_el = find_child_by_localname(val, "origin")
        scale_el = find_child_by_localname(val, "scale")
        digits_el = find_child_by_localname(val, "digits")
        if digits_el is None:
            continue

        origin = float(origin_el.attrib.get("value")) if origin_el is not None else 0.0
        scale = float(scale_el.attrib.get("value")) if scale_el is not None else 1.0

        digits_str = (digits_el.text or "").strip()
        if not digits_str:
            continue

        digits = np.fromstring(digits_str, sep=" ", dtype=float)

        values_uv = origin + scale * digits
        leads[lead_name] = values_uv

    if not leads:
        raise ValueError("Geen ECG-leads (MDC_ECG_LEAD_*) gevonden in XML.")

    max_len = max(len(v) for v in leads.values())
    t = np.arange(max_len) * dt

    return t, leads, dt


def export_xml_to_csv_like_template(
    xml_path: Path,
    csv_path: Path,
    template_csv_path: Path,
    *,
    add_time_if_template_has_it: bool = True,
    time_col_candidates: tuple[str, ...] = ("t", "time", "tijd", "t_s"),
    units: str = "uV",  # "uV" of "mV"
    float_format: str | None = None,  # bv. "%.6f"
) -> pd.DataFrame:
    """
    Schrijft CSV in *exact* dezelfde kolommen+volgorde als template_csv_path.

    - Indien template een tijdkolom heeft (bv. t_s), vullen wij die uit dt.
    - Leaddata zetten wij op kolommen met dezelfde naam als in template.
    - units="mV" deelt door 1000 (van µV naar mV).
    """
    t, leads, _dt = parse_aecg_xml(xml_path)

    template_df0 = pd.read_csv(template_csv_path, nrows=0)
    template_cols = list(template_df0.columns)

    time_col = None
    if add_time_if_template_has_it:
        for cand in time_col_candidates:
            if cand in template_cols:
                time_col = cand
                break
    try:
        template_n = sum(1 for _ in open(template_csv_path, "rb")) - 1 
        if template_n <= 0:
            template_n = None
    except Exception:
        template_n = None

    out_len = len(t)
    if template_n is not None:
        out_len = min(out_len, template_n)

    out = {}
    for col in template_cols:
        out[col] = np.full(out_len, np.nan, dtype=float)

    if time_col is not None:
        out[time_col][:] = t[:out_len]


    for col in template_cols:
        if col == time_col:
            continue
        if col in leads:
            vals = leads[col][:out_len].astype(float, copy=False)
            if units.lower() == "mv":
                vals = vals / 1000.0
            out[col][:len(vals)] = vals

    df = pd.DataFrame(out, columns=template_cols)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format=float_format)
    print(f"[INFO] CSV (template-stijl) opgeslagen te: {csv_path}")
    return df



def parse_xml_lead(xml_path: Path, lead_name: str):
    """
    Haalt één lead (bv. I) uit ECGtizer-XML en geeft (t, waarden_µV) terug.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    dt = None
    for seq in iter_sequences(root):
        code = get_sequence_code(seq)
        if code == "TIME_ABSOLUTE":
            val = find_child_by_localname(seq, "value")
            if val is None:
                continue
            incr = find_child_by_localname(val, "increment")
            if incr is not None:
                dt = float(incr.attrib.get("value"))
                break

    if dt is None:
        raise ValueError("Geen TIME_ABSOLUTE increment (dt) gevonden in XML.")

    values_uv = None

    for seq in iter_sequences(root):
        code = get_sequence_code(seq)
        if code != f"MDC_ECG_LEAD_{lead_name}":
            continue

        val = find_child_by_localname(seq, "value")
        if val is None:
            continue

        origin_el = find_child_by_localname(val, "origin")
        scale_el = find_child_by_localname(val, "scale")
        digits_el = find_child_by_localname(val, "digits")
        if digits_el is None:
            continue

        origin = float(origin_el.attrib.get("value")) if origin_el is not None else 0.0
        scale = float(scale_el.attrib.get("value")) if scale_el is not None else 1.0

        digits_str = (digits_el.text or "").strip()
        if not digits_str:
            continue

        digits = np.fromstring(digits_str, sep=" ", dtype=float)
        values_uv = origin + scale * digits
        break

    if values_uv is None:
        raise ValueError(f"Lead {lead_name} niet gevonden in XML.")

    t = np.arange(len(values_uv)) * dt
    return t, values_uv


def compare_xml_with_csv(
    xml_path: Path,
    csv_path: Path,
    lead_name: str,
    max_samples: int = 5000,
    show_plots: bool = True,
    save_plots_root: Path | None = None,
    base_name: str | None = None,
):
    print(f"[INFO] Vergelijk XML vs CSV voor lead {lead_name}:")
    print(f"       XML: {xml_path}")
    print(f"       CSV: {csv_path}")

    t_xml, lead_xml = parse_xml_lead(xml_path, lead_name)

    df_csv = pd.read_csv(csv_path)
    if lead_name not in df_csv.columns:
        print(f"[WAARSCHUWING] Lead '{lead_name}' niet gevonden als kolom in CSV.")
        print(f"[INFO] Beschikbare kolommen in CSV: {list(df_csv.columns)}")
        return

    lead_csv = df_csv[lead_name].to_numpy()

    n = min(len(lead_xml), len(lead_csv), max_samples)
    if n == 0:
        print("[WAARSCHUWING] Geen overlappende samples om te vergelijken.")
        return

    t = t_xml[:n]
    y_xml = lead_xml[:n]
    y_csv = lead_csv[:n]

    if base_name is None:
        base_name = xml_path.stem

    fig1 = plt.figure()
    plt.plot(t, y_csv, label="CSV")
    plt.plot(t, y_xml, linestyle="--", label="XML (ECGtizer)")
    plt.xlabel("tijd (s)")
    plt.ylabel("amplitude (µV)")
    plt.title(f"Lead {lead_name}: CSV vs ECGtizer-XML (eerste {n} samples)")
    plt.legend()
    plt.tight_layout()

    zoom_n = min(500, n)
    fig2 = plt.figure()
    plt.plot(t[:zoom_n], y_csv[:zoom_n], label="CSV")
    plt.plot(t[:zoom_n], y_xml[:zoom_n], linestyle="--", label="XML (ECGtizer)")
    plt.xlabel("tijd (s)")
    plt.ylabel("amplitude (µV)")
    plt.title(f"Lead {lead_name}: zoom eerste {zoom_n} samples")
    plt.legend()
    plt.tight_layout()

    if save_plots_root is not None:
        save_plots_root.mkdir(parents=True, exist_ok=True)
        fig1_path = save_plots_root / f"{base_name}_lead{lead_name}_compare_full.png"
        fig2_path = save_plots_root / f"{base_name}_lead{lead_name}_compare_zoom.png"
        fig1.savefig(fig1_path)
        fig2.savefig(fig2_path)
        print(f"[INFO] Plots opgeslagen te:\n       {fig1_path}\n       {fig2_path}")

    if show_plots:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)

def digitize_one_image(
    png_path: Path,
    input_root: Path,
    tmp_root: Path,
    output_root: Path,
    dpi: int,
    method: str,
    model_path: str | None,
    device: torch.device,
    debug: bool,
    compare_csv_root: Path | None,
    compare_lead: str,
    max_samples_compare: int,
    show_plots: bool,
    plots_root: Path | None,
    keep_xml: bool,
) -> bool:
    try:
        rel = png_path.relative_to(input_root)
    except ValueError:
        rel = Path(png_path.name)

    pdf_path = (tmp_root / rel).with_suffix(".pdf")
    xml_path = (tmp_root / rel).with_suffix(".xml")
    csv_path = (output_root / rel).with_suffix(".csv")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        png_to_pdf(png_path, pdf_path, dpi)
    except Exception as e:  # noqa: BLE001
        print(f"[WAARSCHUWING] Kon PNG→PDF niet uitvoeren voor {png_path}: {e}")
        if debug:
            traceback.print_exc()
        return False
 
    try:
        ecg = run_ecgtizer_multi(
            pdf_path=pdf_path,
            base_dpi=dpi,
            base_method=method,
            debug=debug,
        )
    except Exception as e:
        print(f"[WAARSCHUWING] ECGtizer faalde bij {png_path}: {e}")
        if debug:
            traceback.print_exc()
        return False

    if model_path:
        try:
            print(
                f"[INFO] Completion toepasse op {png_path} "
                f"met model '{model_path}' en device '{device}'",
            )
            ecg.completion(model_path=model_path, device=device)
            summarize_ecg(ecg, label="(na completion)")
        except Exception as e:  # noqa: BLE001
            print(f"[WAARSCHUWING] Completion faalde bij {png_path}: {e}")
            if debug:
                traceback.print_exc()

    try:
        ecg.save_xml(str(xml_path))
        print(f"[INFO] XML (tusschen) opgeslagen te: {xml_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[WAARSCHUWING] Kon XML niet opslaan voor {png_path}: {e}")
        if debug:
            traceback.print_exc()
        return False

    try:
        template_csv = compare_csv_root / f"{png_path.stem.split('-')[0]}.csv" if compare_csv_root else None
        if template_csv is not None and template_csv.exists():
            export_xml_to_csv_like_template(
                xml_path=xml_path,
                csv_path=csv_path,
                template_csv_path=template_csv,
                units="uV",
                float_format=None,
            )
        else:
            export_xml_to_csv(xml_path, csv_path)
    except Exception as e:  # noqa: BLE001
        print(f"[WAARSCHUWING] Kon CSV niet maken voor {xml_path}: {e}")
        if debug:
            traceback.print_exc()
        return False


    if compare_csv_root is not None:
        stem = png_path.stem
        base_id = stem.split("-")[0]
        orig_csv_path = compare_csv_root / f"{base_id}.csv"

        if orig_csv_path.exists():
            print(f"[INFO] Originele CSV gevonden: {orig_csv_path}")
            try:
                compare_xml_with_csv(
                    xml_path=xml_path,
                    csv_path=orig_csv_path,
                    lead_name=compare_lead,
                    max_samples=max_samples_compare,
                    show_plots=show_plots,
                    save_plots_root=plots_root,
                    base_name=stem,
                )
            except Exception as e:
                print(
                    f"[WAARSCHUWING] Vergelijking XML/CSV faalde voor {png_path}: {e}",
                )
                if debug:
                    traceback.print_exc()
        else:
            print(
                f"[INFO] Geen originele CSV gevonden voor {png_path} "
                f"({orig_csv_path} ontbreekt).",
            )
    if not keep_xml:
        try:
            xml_path.unlink(missing_ok=True)
        except Exception:
            pass

    return True

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-gewijze ECG-digitisatie met ECGtizer (CSV-uitvoer; paden naast het script).",
    )

    parser.add_argument(
        "--input_root",
        type=str,
        default="train",
        help="Map met ECG-afbeeldingen (relatief t.o.v. scriptmap; standaard: train).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="digitized_csv",
        help="Map waar CSV-uitvoer komt (relatief t.o.v. scriptmap; standaard: digitized_csv).",
    )
    parser.add_argument(
        "--tmp_root",
        type=str,
        default="tmp_ecgtizer_pdfs",
        help="Tijdelijke map voor PDF-bestanden en tusschen-XML (relatief t.o.v. scriptmap).",
    )
    parser.add_argument(
        "--keep_xml",
        action="store_true",
        help="Behoudt tusschen-XML in tmp_root (standaard: neen; zij worden verwijderd).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=500,
        help=(
            "Voorkeurs-DPI-resolutie voor PDF-rendering (standaard: 500). "
            "Script probeert bij mislukking ook andere DPI-waarden."
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        default="fragmented",
        choices=["full", "lazy", "fragmented"],
        help=(
            "Voorkeurs-ECGtizer-extractiemethode. "
            "Bij falen worden ook de andere methoden geprobeerd."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "Pad naar het U-Net-model (ECGrecover) voor completion. "
            "Relatief t.o.v. scriptmap, tenzij absolute weg."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch-device: 'auto', 'cpu' of bijvoorbeeld 'cuda'.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="Glob-pattern voor afbeeldingen (standaard: *.png).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Zet ECGtizer in DEBUG/verbose-modus (meer logs en tracebacks).",
    )

    parser.add_argument(
        "--compare_csv_root",
        type=str,
        default=".",
        help=(
            "Map met originele CSV-bestanden (relatief t.o.v. scriptmap; standaard: .). "
            "Leeg laten om vergelijking uit te zetten."
        ),
    )
    parser.add_argument(
        "--compare_lead",
        type=str,
        default="I",
        help="Lead-naam voor vergelijking (standaard: I).",
    )
    parser.add_argument(
        "--max_samples_compare",
        type=int,
        default=5000,
        help="Maximaal aantal samples om te tonen in vergelijkingsplots.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Geen plots tonen (wel opslaan indien --plots_root gezet).",
    )
    parser.add_argument(
        "--plots_root",
        type=str,
        default="Plots",
        help="Map om vergelijkingsplots in op te slaan (relatief t.o.v. scriptmap; standaard: Plots).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_root = resolve_from_script_dir(args.input_root)
    output_root = resolve_from_script_dir(args.output_root)
    tmp_root = resolve_from_script_dir(args.tmp_root)

    compare_csv_root = (
        resolve_from_script_dir(args.compare_csv_root) if args.compare_csv_root else None
    )
    plots_root = resolve_from_script_dir(args.plots_root) if args.plots_root else None

    model_path = args.model_path
    if model_path:
        model_path = str(resolve_from_script_dir(model_path))

    show_plots = not args.no_plots

    if not input_root.exists():
        raise SystemExit(f"Input-root bestaat niet: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    png_files = sorted(input_root.glob("**/" + args.pattern))
    if not png_files:
        raise SystemExit(
            f"Geen afbeeldingen gevonden onder {input_root} met patroon {args.pattern}",
        )

    print(f"Gevonden afbeeldingen: {len(png_files)}")
    print(f"Voorbeeld: {png_files[0]}")

    ok_count = 0
    fail_count = 0

    for png_path in tqdm(png_files, desc="ECG's digitaliserende"):
        success = digitize_one_image(
            png_path=png_path,
            input_root=input_root,
            tmp_root=tmp_root,
            output_root=output_root,
            dpi=args.dpi,
            method=args.method,
            model_path=model_path,
            device=device,
            debug=args.debug,
            compare_csv_root=compare_csv_root,
            compare_lead=args.compare_lead,
            max_samples_compare=args.max_samples_compare,
            show_plots=show_plots,
            plots_root=plots_root,
            keep_xml=args.keep_xml,
        )
        if success:
            ok_count += 1
        else:
            fail_count += 1

    print(
        f"Voltooid. Succesvol: {ok_count}, mislukt: {fail_count}. "
        f"CSV-bestanden staan in: {output_root}",
    )
    if compare_csv_root is not None:
        print(
            "Vergelijking met originele CSV werd geprobeerd voor passende IDs "
            f"onder: {compare_csv_root}",
        )
    if args.keep_xml:
        print(f"Tusschen-XML (indien gemaakt) bleef behouden in: {tmp_root}")


if __name__ == "__main__":
    main()
