#!/usr/bin/env python3
"""
inaturalist_scraper.py - Scrape Alabama snake photos from iNaturalist (CS 499 project)

Uses the official API via `pyinaturalist` (not HTML scraping). Only keeps
photos with a usable license (CC0/CC-BY/CC-BY-SA/CC-BY-NC/CC-BY-NC-SA),
downloads them per-species, and writes manifest.csv with attribution info.

manifest.csv / labels.csv are MERGED on each run (not overwritten), so you can
scrape species in separate runs without losing earlier records.

Install:  pip install pyinaturalist requests
Run:      python3 inaturalist_scraper.py --output-dir ./data --max-per-species 150
Test one: python3 inaturalist_scraper.py --species "Agkistrodon contortrix" --max-per-species 10
Mimics:   python3 inaturalist_scraper.py --output-dir data/Output --first-photo-only \
              --max-per-species 400 \
              --species "Lampropeltis elapsoides,Lampropeltis triangulum,Nerodia sipedon,Nerodia erythrogaster,Heterodon platirhinos,Pantherophis guttatus" \
              --place "Alabama, US|Georgia, US|Florida, US|Mississippi, US|Tennessee, US"
"""

import argparse, csv, os, sys, time
import requests
from pyinaturalist import get_observations, get_taxa, get_places_autocomplete

REQUEST_DELAY = 1.1
LICENSES = ["CC0", "CC-BY", "CC-BY-SA", "CC-BY-NC", "CC-BY-NC-SA"]

# (common_name, scientific_name, category)
# category: "venomous" | "mimic" (target look-alike) | "other" (catch-all /
# "unknown species" bucket -- every other snake known to occur in Alabama,
# collected in smaller numbers just so the classifier has *something* to
# fall back on instead of forcing a confident wrong answer on species it
# was never trained to recognize).
SPECIES = [
    # ---- 6 venomous target species (matches ADCNR/Outdoor Alabama's official count) ----
    ("Copperhead", "Agkistrodon contortrix", "venomous"),
    ("Cottonmouth", "Agkistrodon piscivorus", "venomous"),
    ("Timber Rattlesnake", "Crotalus horridus", "venomous"),
    ("Pygmy Rattlesnake", "Sistrurus miliarius", "venomous"),
    ("Eastern Diamondback Rattlesnake", "Crotalus adamanteus", "venomous"),
    ("Eastern Coral Snake", "Micrurus fulvius", "venomous"),

    # ---- 6 curated look-alike / mimic species ----
    ("Scarlet Kingsnake", "Lampropeltis elapsoides", "mimic"),
    ("Milk Snake", "Lampropeltis triangulum", "mimic"),
    ("Common Watersnake", "Nerodia sipedon", "mimic"),
    ("Plain-bellied Watersnake", "Nerodia erythrogaster", "mimic"),
    ("Eastern Hognose Snake", "Heterodon platirhinos", "mimic"),
    ("Corn Snake", "Pantherophis guttatus", "mimic"),

    # ---- "other" catch-all: remaining non-venomous species known in Alabama ----
    ("Worm Snake", "Carphophis amoenus", "other"),
    ("Scarlet Snake", "Cemophora coccinea", "other"),
    ("Black Racer", "Coluber constrictor", "other"),
    ("Ringneck Snake", "Diadophis punctatus", "other"),
    ("Eastern Indigo Snake", "Drymarchon couperi", "other"),
    ("Mud Snake", "Farancia abacura", "other"),
    ("Rainbow Snake", "Farancia erytrogramma", "other"),
    ("Southern Hognose Snake", "Heterodon simus", "other"),
    ("Prairie Kingsnake", "Lampropeltis calligaster", "other"),
    ("Mole Kingsnake", "Lampropeltis rhombomaculata", "other"),
    ("Eastern Kingsnake", "Lampropeltis getula", "other"),
    ("Black Kingsnake", "Lampropeltis nigra", "other"),
    ("Eastern Coachwhip", "Masticophis flagellum", "other"),
    ("Gulf Salt Marsh Snake", "Nerodia clarkii", "other"),
    ("Mississippi Green Watersnake", "Nerodia cyclopion", "other"),
    ("Banded Watersnake", "Nerodia fasciata", "other"),
    ("Florida Green Watersnake", "Nerodia floridana", "other"),
    ("Diamondback Watersnake", "Nerodia rhombifer", "other"),
    ("Brown Watersnake", "Nerodia taxispilota", "other"),
    ("Rough Green Snake", "Opheodrys aestivus", "other"),
    ("Eastern Rat Snake", "Pantherophis alleghaniensis", "other"),
    ("Pine Snake", "Pituophis melanoleucus", "other"),
    ("Glossy Crayfish Snake", "Liodytes rigida", "other"),
    ("Queen Snake", "Regina septemvittata", "other"),
    ("Pine Woods Snake", "Rhadinaea flavilata", "other"),
    ("Swamp Snake", "Liodytes pygaea", "other"),
    ("Brown Snake", "Storeria dekayi", "other"),
    ("Redbelly Snake", "Storeria occipitomaculata", "other"),
    ("Crowned Snake", "Tantilla coronata", "other"),
    ("Ribbon Snake", "Thamnophis saurita", "other"),
    ("Garter Snake", "Thamnophis sirtalis", "other"),
    ("Rough Earth Snake", "Virginia striatula", "other"),
    ("Smooth Earth Snake", "Virginia valeriae", "other"),
]


def get_place_id(name):
    r = get_places_autocomplete(q=name).get("results", [])
    if not r:
        print(f"  place not found, skipping: {name}")
        return None
    print(f"Place: {r[0]['display_name']} (id={r[0]['id']})")
    return r[0]["id"]


def get_place_ids(place_arg):
    """Accepts one place, or several separated by '|', e.g.
    'Alabama, US|Georgia, US|Florida, US'. Returns a list of place_ids."""
    ids = []
    for p in place_arg.split("|"):
        p = p.strip()
        if not p:
            continue
        pid = get_place_id(p)
        if pid:
            ids.append(pid)
    return ids


def get_taxon_id(sci_name):
    for t in get_taxa(q=sci_name, rank="species", per_page=5).get("results", []):
        if t["name"].lower() == sci_name.lower():
            return t["id"]
    return None


def fetch_observations(taxon_id, place_ids, max_n, quality, alive_only=False):
    out, id_above = [], None
    while len(out) < max_n:
        kwargs = dict(taxon_id=taxon_id, place_id=place_ids, photos=True,
                      photo_license=LICENSES, per_page=min(100, max_n - len(out)),
                      order_by="id", order="asc")
        if quality and quality != "any":
            kwargs["quality_grade"] = quality
        if alive_only:
            # iNaturalist's "Alive or Dead" annotation: term_id=17, value 18=Alive.
            # Only catches observations someone has actually annotated -- it
            # will not find every live-snake photo, but it will reliably
            # exclude anything tagged Dead (roadkill, museum specimens, etc.)
            kwargs["term_id"] = 17
            kwargs["term_value_id"] = 18
        if id_above:
            kwargs["id_above"] = id_above
        time.sleep(REQUEST_DELAY)
        page = get_observations(**kwargs).get("results", [])
        if not page:
            break
        out.extend(page)
        id_above = page[-1]["id"]
        if len(page) < kwargs["per_page"]:
            break
    return out[:max_n]


def photo_records(obs, name, sci_name, category):
    loc = obs.get("location")
    if isinstance(loc, (list, tuple)) and len(loc) == 2:
        lat, lng = loc
    elif isinstance(loc, str) and "," in loc:
        lat, lng = loc.split(",")[:2]
    else:
        lat, lng = None, None
    for p in obs.get("photos", []):
        if not p.get("license_code"):
            continue  # no license = all rights reserved, skip
        yield {
            "common_name": name, "scientific_name": sci_name, "category": category,
            "observation_id": obs["id"], "photo_id": p["id"],
            "license": p["license_code"], "attribution": p.get("attribution", ""),
            "observed_on": obs.get("observed_on"), "lat": lat, "lng": lng,
            "url": p.get("url", "").replace("square", "medium"),
            "inat_link": f"https://www.inaturalist.org/observations/{obs['id']}",
        }


def download(url, path, session):
    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return True
    except requests.RequestException as e:
        print(f"  download failed: {e}")
        return False


def sanitize_filename(name):
    return name.replace(" ", "_").replace("/", "-")


def merge_csv(path, new_rows, fieldnames, key):
    """Merge new rows into an existing CSV instead of overwriting it.
    Rows with the same key are replaced by the new version."""
    existing = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8-sig") as f:
            existing = list(csv.DictReader(f))
    new_keys = {str(r[key]) for r in new_rows}
    merged = [r for r in existing if str(r.get(key)) not in new_keys] + new_rows
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)
    return len(merged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="./inat_data")
    ap.add_argument("--place", default="Alabama, US",
                     help="one place, or several separated by '|', e.g. "
                          "'Alabama, US|Georgia, US|Florida, US|Mississippi, US|Tennessee, US'")
    ap.add_argument("--max-per-species", type=int, default=150,
                     help="max observations per venomous/mimic species (default 150)")
    ap.add_argument("--max-per-other", type=int, default=25,
                     help="max observations per 'other' catch-all species (default 25, kept small on purpose)")
    ap.add_argument("--quality-grade", default="research", choices=["research", "needs_id", "casual", "any"])
    ap.add_argument("--species", help="comma-separated scientific names to limit the run")
    ap.add_argument("--skip-other", action="store_true", help="skip the 'other' catch-all species entirely")
    ap.add_argument("--alive-only", action="store_true",
                     help="only keep observations annotated 'Alive' on iNaturalist (filters out "
                          "roadkill/dead specimens, but only catches observations someone actually "
                          "annotated -- expect fewer results, not a perfect filter)")
    ap.add_argument("--first-photo-only", action="store_true",
                     help="download only the first usable photo per observation "
                          "(avoids the same snake landing in both train and test)")
    args = ap.parse_args()

    species = SPECIES
    if args.skip_other:
        species = [s for s in species if s[2] != "other"]
    if args.species:
        wanted = {s.strip().lower() for s in args.species.split(",")}
        species = [s for s in species if s[1].lower() in wanted]
        if not species:
            sys.exit("None of the --species names matched the SPECIES list -- check spelling.")

    img_dir = os.path.join(args.output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    place_ids = get_place_ids(args.place)
    if not place_ids:
        sys.exit("No valid places found -- check --place spelling.")
    session = requests.Session()
    session.headers["User-Agent"] = "UAH-CS499-SnakeID-Project/1.0"

    rows = []
    summary = []
    for name, sci, category in species:
        print(f"=== {name} ({sci}) ===")
        taxon_id = get_taxon_id(sci)
        if not taxon_id:
            print("  taxon not found, skipping")
            summary.append((sci, 0, 0))
            continue

        cap = args.max_per_other if category == "other" else args.max_per_species
        obs_list = fetch_observations(taxon_id, place_ids, cap, args.quality_grade, args.alive_only)
        print(f"  {len(obs_list)} observations")

        species_dir = os.path.join(img_dir, sci.replace(" ", "_"))
        os.makedirs(species_dir, exist_ok=True)

        n = 0
        for obs in obs_list:
            for rec in photo_records(obs, name, sci, category):
                n += 1
                fname = f"{sanitize_filename(name)}{n}.jpg"   # e.g. Corn_Snake1.jpg
                path = os.path.join(species_dir, fname)
                if download(rec["url"], path, session):
                    rec["local_path"] = os.path.relpath(path, args.output_dir)
                    rows.append(rec)
                    time.sleep(0.3)
                    if args.first_photo_only:
                        break   # one photo per observation
                else:
                    n -= 1  # download failed, don't burn a number
                    time.sleep(0.3)
        print(f"  downloaded {n} images")
        summary.append((sci, len(obs_list), n))

    if rows:
        manifest = os.path.join(args.output_dir, "manifest.csv")
        total = merge_csv(manifest, rows, list(rows[0].keys()), key="photo_id")

        # Lean labels.csv for labeling/training tools -- just the columns
        # you actually need to point a labeling tool or dataloader at a file
        # and know its ground-truth class. Attribution/license stays in
        # manifest.csv only.
        label_cols = ["local_path", "scientific_name", "common_name", "category", "observation_id"]
        labels = os.path.join(args.output_dir, "labels.csv")
        merge_csv(labels, rows, label_cols, key="local_path")

        print("\n===== Summary =====")
        for sci, n_obs, n_img in summary:
            print(f"  {sci:30s} {n_obs:4d} observations  {n_img:4d} images")
        print(f"\nDone: {len(rows)} new images ({total} total in manifest).")
        print(f"  Manifest: {manifest}\n  Labels:   {labels}")
    else:
        print("No images downloaded.")


if __name__ == "__main__":
    main()