"""Pirelli's compound nomination for every race (hard/medium/soft as C-numbers), from the race's English Wikipedia
article → reference/compounds.csv, with the sentence it came from and the article URL. Races whose article has no
such sentence are written with empty compounds, to be filled by hand from another source (and marked as such)."""
import glob
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

API = "https://en.wikipedia.org/w/api.php?action=parse&prop=wikitext&format=json&formatversion=2&redirects=1&page="
UA = {"User-Agent": "undercut-research/0.1 (student sports-analytics project)"}


CACHE = "data/wiki"


def wikitext(title):
    """The article's wikitext, cached in data/wiki/; waits between requests and backs off when rate-limited."""
    path = f"{CACHE}/{title.replace(' ', '_').replace('/', '_')}.json"
    if os.path.exists(path):
        return json.load(open(path)).get("parse", {}).get("wikitext", "")
    req = urllib.request.Request(API + urllib.parse.quote(title.replace(" ", "_")), headers=UA)
    for wait in (5, 15, 45, 90):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(wait)
    else:
        raise RuntimeError(f"rate-limited on {title}")
    os.makedirs(CACHE, exist_ok=True)
    json.dump(body, open(path, "w"))
    time.sleep(2)
    return body.get("parse", {}).get("wikitext", "")


def nomination(text):
    """The first sentence naming Pirelli (or 'tyre compounds') and three C-numbers; returns (sorted numbers, sentence)."""
    text = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", " ", text, flags=re.S)  # citations would join sentences
    for s in re.split(r"(?<=[.!?])\s+|\n", text):
        nums = sorted({int(x) for x in re.findall(r"\bC([0-6])\b", s)})
        if len(nums) == 3 and re.search(r"Pirelli|compound", s):
            return nums, re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", s).strip()
    return None, ""


def build():
    rows = []
    for f in sorted(glob.glob("data/laps/*_info.json")):
        info = json.load(open(f))
        title = f"{info['Year']} {info['Event']}"
        nums, sentence = nomination(wikitext(title))
        rows.append(dict(race_id=info["race_id"], year=info["Year"], event=info["Event"],
                         hard=nums[0] if nums else None, medium=nums[1] if nums else None,
                         soft=nums[2] if nums else None, source="wikipedia" if nums else "",
                         url="https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                         evidence=sentence))
        print(info["race_id"], title, nums, flush=True)
    pd.DataFrame(rows).to_csv("reference/compounds.csv", index=False)


if __name__ == "__main__":
    build()
