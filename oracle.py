# oracle.py
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# ---------- 1) DATA ----------
# TODO: create a small list[tuple[str, str]] where each tuple is (text, label)
# Start with ~3-4 short lines per show for: Naruto, JoJo, AOT, One Piece, Demon Slayer
DATA: List[Tuple[str, str]] = [
    # ("shadow clones ramen training", "Naruto"),
    # ("summon my stand ora ora", "JoJo"),
    # ...
    ("Dattebayo","Naruto"),
    ("For those who don't believe in themselves, hardwork is worthless!", "Naruto"), #Might Guy,
    ("That is my ninja way!", "Naruto"),
    ("Believe it!", "Naruto"),
    ("Hard work is worthless for those that don't believe in themselves", "Naruto"),
    ("Im gonna be Hokage", "Naruto"),
    ("Those who break the rules are scum, but those who abandon their friends are worse than scum.", "Naruto"),

    ("Daga, kotowaru.", "Jojo's Bizarre Adventure"),
    ("I refuse", "Jojo's Bizarre Adventure"),
    ("I reject my humanity Jojo!", "Jojo's Bizarre Adventure"),
    ("It was me, DIO!", "Jojo's Bizarre Adventure"),
    ("ORA ORA ORA ORA!", "Jojo's Bizarre Adventure"),
    ("MUDA MUDA MUDA MUDA!", "Jojo's Bizarre Adventure"),
    ("Your next line is...", "Jojo's Bizarre Adventure"),

    ("Im gonna be the king of the Pirates!", "One Piece"),
    ("I want to live!", "One Piece"),
    ("Nothing Happened.", "One Piece"),
    ("People's dreams... never end!", "One Piece"),
    ("I’m going to be the world’s greatest swordsman!", "One Piece"),

    ("It's over 9000!", "Dragon Ball"),
    ("Kamehameha!", "Dragon Ball"),
    ("I am the Prince of all Saiyans", "Dragon Ball"),
    ("This isn't even my final form", "Dragon Ball"),
    ("Final Flash!", "Dragon Ball"),

    ("I'll become the Wizard King!", "Black Clover"),
    ("Surpass your limits. Right now.", "Black Clover"), #- Yami Sukehiro
    ("My magic is never giving up!", "Black Clover"), # - Asta
    ("I’ll surpass you, Asta.", "Black Clover"), # - Yuno
    ("I’ll become someone who can protect everyone!", "Black Clover"), # - Asta
    ("Black Bulls, move out!", "Black Clover"), # - Yami Sukehiro

    ("If you have time to fantasize about a beautiful end, then live beautifully until the end.", "Gintama"), # - Gintoki Sakata
    ("The country? The world? Those are big words. I just want to protect what’s in front of me.", "Gintama"), # - Gintoki Sakata
    ("Even if we’re Shonen Jump protagonists, we still get hit by cars.", "Gintama"), #- Gintoki Sakata
    ("You yourself have to change first, or nothing will change for you.", "Gintama"), # - Gintoki Sakata

    ("Go beyond! Plus Ultra!", "My Hero Academia"), # - All Might
    ("I am here!", "My Hero Academia"), # - All Might
    ("You can become a hero.", "My Hero Academia"), # - All Might
    ("United States of Smash!", "My Hero Academia"), # - All Might

    ("If you win, you live. If you lose, you die. If you don’t fight, you can’t win!", "Attack on Titan"), # - Eren Yeager
    ("This world is cruel and merciless, but it’s also very beautiful.", "Attack on Titan"), # - Mikasa Ackerman
    ("Shinzō wo sasageyo!", "Attack on Titan"), # - Erwin Smith / Survey Corps
    ("On that day, humanity received a grim reminder…", "Attack on Titan"), # - Narration
    ("I want to keep moving forward until I destroy my enemies.", "Attack on Titan"), # - Eren Yeager

    ("You should enjoy the little detours. That’s where you’ll find the things more important than what you want.", "Hunter × Hunter"), # - Ging Freecss
    ("If I ignore a friend I have the ability to help, wouldn’t I be betraying him?", "Hunter × Hunter"), # - Killua Zoldyck
    ("Bungee Gum possesses the properties of both rubber and gum.", "Hunter × Hunter"),# - Hisoka Morow
    ("Thank you for a truly fun fight.", "Hunter × Hunter"), # - Isaac Netero
    ("My name is Meruem.", "Hunter × Hunter") # - Meruem
    
]

LABELS = sorted({lbl for _, lbl in DATA}) #{} indicate a set comprehension, adds lbl each itteration to set

def show_label_counts():
    c = Counter(lbl for _, lbl in DATA)
    print("Label counts:", dict(c))

# ---------- 2) MODEL PIPELINE ----------
# We’ll add scikit-learn bits after we confirm the dataset is fine.

def main():
    show_label_counts()
    # TODO later: call train(), eval(), then launch REPL

if __name__ == "__main__":
    main()
