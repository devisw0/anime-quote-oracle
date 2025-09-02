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


# DESIGN GUIDELINES FOR DISTINCTIVE ANIME QUOTES:
# 1. Use character names (Goku, Luffy, Naruto, Deku, Eren, Jotaro)
# 2. Use unique terminology (Sharingan, Kamehameha, Devil Fruit, Stand, Quirk, Titan)
# 3. Use location names (Konoha, Grand Line, UA Academy, Wall Maria)
# 4. Avoid generic action words (fight, power, strong, battle)
# 5. Avoid common sentence structures ("I will become...")
# 6. Include unique sounds/catchphrases (Dattebayo, Ora, Plus Ultra)


DATA: List[Tuple[str, str]] = [
    ("Dattebayo", "Naruto"),
    ("Sasuke activated his Sharingan", "Naruto"),
    ("Hokage protects Konoha village", "Naruto"),
    ("Rasengan spiral chakra technique", "Naruto"),
    ("Nine-tails Kurama fox demon", "Naruto"),
    ("Akatsuki organization hunts bijuu", "Naruto"),
    ("Chidori lightning blade jutsu", "Naruto"),
    ("Byakugan Hyuga clan bloodline", "Naruto"),
    ("Hidden Leaf genin graduation", "Naruto"),
    ("Itachi Uchiha clan massacre", "Naruto"),
    ("Sage Mode natural energy", "Naruto"),
    ("Kakashi sensei teaches team", "Naruto"),
    ("Orochimaru snake sannin villain", "Naruto"),
    ("Hinata loves Naruto Uzumaki", "Naruto"),
    ("Shikamaru shadow jutsu strategy", "Naruto"),
    ("Gaara sand village kazekage", "Naruto"),
    ("Tsunade fifth hokage gambler", "Naruto"),
    ("Jiraiya pervy sage author", "Naruto"),
    ("Madara Uchiha final boss", "Naruto"),
    ("Boruto next generation sequel", "Naruto"),
    
    # Dragon Ball - Focus on: Saiyan, Goku, Vegeta, ki, Kamehameha, planet names
    ("Goku transforms Super Saiyan", "Dragon Ball"),
    ("Vegeta Prince of Saiyans", "Dragon Ball"),
    ("Kamehameha energy wave attack", "Dragon Ball"),
    ("Frieza destroyed Planet Vegeta", "Dragon Ball"),
    ("Piccolo Namekian green warrior", "Dragon Ball"),
    ("Gohan half-saiyan hybrid", "Dragon Ball"),
    ("Trunks future timeline warrior", "Dragon Ball"),
    ("Cell android perfect form", "Dragon Ball"),
    ("Majin Buu pink destruction", "Dragon Ball"),
    ("Beerus God of Destruction", "Dragon Ball"),
    ("Whis angel martial arts", "Dragon Ball"),
    ("Bulma Capsule Corp scientist", "Dragon Ball"),
    ("Krillin bald human fighter", "Dragon Ball"),
    ("Master Roshi turtle hermit", "Dragon Ball"),
    ("Shenron dragon grants wishes", "Dragon Ball"),
    ("Yamcha baseball desert bandit", "Dragon Ball"),
    ("Tien three-eyed martial artist", "Dragon Ball"),
    ("Chiaotzu small psychic fighter", "Dragon Ball"),
    ("Raditz Goku brother saiyan", "Dragon Ball"),
    ("Nappa bald saiyan elite", "Dragon Ball"),
    
    # One Piece - Focus on: Luffy, pirates, Devil Fruit, Grand Line, crew names
    ("Luffy rubber Gomu Gomu", "One Piece"),
    ("Zoro three sword santoryu", "One Piece"),
    ("Nami navigator weather witch", "One Piece"),
    ("Sanji black leg cook", "One Piece"),
    ("Chopper reindeer doctor", "One Piece"),
    ("Robin archaeology Ohara survivor", "One Piece"),
    ("Franky cyborg shipwright", "One Piece"),
    ("Brook skeleton musician", "One Piece"),
    ("Jinbe fishman helmsman", "One Piece"),
    ("Shanks red hair yonko", "One Piece"),
    ("Whitebeard strongest man alive", "One Piece"),
    ("Blackbeard darkness logia fruit", "One Piece"),
    ("Ace fire fist portgas", "One Piece"),
    ("Sabo revolutionary army chief", "One Piece"),
    ("Kaido strongest creature beast", "One Piece"),
    ("Big Mom charlotte linlin", "One Piece"),
    ("Doflamingo string heavenly demon", "One Piece"),
    ("Crocodile sand warlord baroque", "One Piece"),
    ("Enel lightning god skypiea", "One Piece"),
    ("Going Merry thousand sunny", "One Piece"),
    
    # My Hero Academia - Focus on: Deku, All Might, Quirks, UA Academy, hero names
    ("Deku One For All", "My Hero Academia"),
    ("All Might Symbol of Peace", "My Hero Academia"),
    ("Bakugo explosion quirk kacchan", "My Hero Academia"),
    ("Todoroki fire ice quirk", "My Hero Academia"),
    ("Iida engine quirk ingenium", "My Hero Academia"),
    ("Uraraka zero gravity quirk", "My Hero Academia"),
    ("Aizawa erasure quirk eraser", "My Hero Academia"),
    ("Shigaraki decay quirk villain", "My Hero Academia"),
    ("UA Academy hero course", "My Hero Academia"),
    ("All For One villain master", "My Hero Academia"),
    ("Endeavor flame hero number", "My Hero Academia"),
    ("Hawks wing hero commission", "My Hero Academia"),
    ("Mirko rabbit hero", "My Hero Academia"),
    ("Best Jeanist fiber hero", "My Hero Academia"),
    ("Midnight sleep quirk hero", "My Hero Academia"),
    ("Present Mic voice quirk", "My Hero Academia"),
    ("Recovery Girl healing quirk", "My Hero Academia"),
    ("Nezu principal intelligent animal", "My Hero Academia"),
    ("Toga blood transform quirk", "My Hero Academia"),
    ("Dabi blue flame quirk", "My Hero Academia"),
    
    # Attack on Titan - Focus on: Eren, titans, walls, Survey Corps, Eldian/Marley
    ("Eren Yeager founding titan", "Attack on Titan"),
    ("Mikasa Ackerman scarf red", "Attack on Titan"),
    ("Armin Colossal titan intelligence", "Attack on Titan"),
    ("Levi Ackerman humanity strongest", "Attack on Titan"),
    ("Survey Corps wings freedom", "Attack on Titan"),
    ("Wall Maria Rose Sina", "Attack on Titan"),
    ("Colossal Titan Bertholdt", "Attack on Titan"),
    ("Armored Titan Reiner", "Attack on Titan"),
    ("Beast Titan Zeke monkey", "Attack on Titan"),
    ("Female Titan Annie crystal", "Attack on Titan"),
    ("Jaw Titan Porco Marcel", "Attack on Titan"),
    ("Cart Titan Pieck quadruped", "Attack on Titan"),
    ("War Hammer Titan crystal", "Attack on Titan"),
    ("Eldian devil island paradis", "Attack on Titan"),
    ("Marley warrior program", "Attack on Titan"),
    ("Paths coordinate ymir fritz", "Attack on Titan"),
    ("Rumbling wall titans march", "Attack on Titan"),
    ("ODM gear maneuver equipment", "Attack on Titan"),
    ("Shiganshina district hometown", "Attack on Titan"),
    ("Grisha Yeager basement key", "Attack on Titan"),
    
    # Jojo - Focus on: Stand names, Ora/Muda, Joestar, DIO, bizarre
    ("Jotaro Star Platinum ora", "Jojo's Bizarre Adventure"),
    ("DIO World time stop", "Jojo's Bizarre Adventure"),
    ("Joseph Hermit Purple hamon", "Jojo's Bizarre Adventure"),
    ("Josuke Crazy Diamond restoration", "Jojo's Bizarre Adventure"),
    ("Giorno Golden Experience requiem", "Jojo's Bizarre Adventure"),
    ("Jolyne Stone Free string", "Jojo's Bizarre Adventure"),
    ("Jonathan hamon breathing ripple", "Jojo's Bizarre Adventure"),
    ("Kakyoin Hierophant Green emerald", "Jojo's Bizarre Adventure"),
    ("Polnareff Silver Chariot sword", "Jojo's Bizarre Adventure"),
    ("Avdol Magician Red fire", "Jojo's Bizarre Adventure"),
    ("Iggy Fool sand dog", "Jojo's Bizarre Adventure"),
    ("Kira Yoshikage Killer Queen", "Jojo's Bizarre Adventure"),
    ("Diavolo King Crimson time", "Jojo's Bizarre Adventure"),
    ("Pucci Made in Heaven acceleration", "Jojo's Bizarre Adventure"),
    ("Valentine Dirty Deeds Done", "Jojo's Bizarre Adventure"),
    ("Speedwagon Foundation research organization", "Jojo's Bizarre Adventure"),
    ("Pillar Men ultimate lifeform", "Jojo's Bizarre Adventure"),
    ("Arrow stand virus evolution", "Jojo's Bizarre Adventure"),
    ("Bizarre adventure Joestar bloodline", "Jojo's Bizarre Adventure"),
    ("Muda muda muda useless", "Jojo's Bizarre Adventure"),
]

# QUALITY CHECK GUIDELINES:
# - Every quote should contain at least 2 unique identifiers for its anime
# - No quote should be comprehensible without anime knowledge  
# - Character names should appear in most quotes
# - Avoid quotes that could logically belong to multiple anime
# - Test: "Could someone guess the anime from this quote alone?"

LABELS = sorted({lbl for _, lbl in DATA}) #{} indicate a set comprehension, adds lbl each itteration to set

def show_label_counts():
    c = Counter(lbl for _, lbl in DATA)
    print("Label counts:", dict(c))

# ---------- 2) MODEL PIPELINE ----------
# We’ll add scikit-learn bits after we confirm the dataset is fine.


def train_and_eval(DATA):

    #scikit: create object -> fit -> predict
    
    quotelist = [quote for quote,_ in DATA]
    print(len(quotelist))
    animelist = [animename for _,animename in DATA]
    print(len(animelist))

    #splitting to training and testing data
    X_train, X_test, y_train, y_test = train_test_split(quotelist,animelist,test_size=0.25,stratify=animelist, random_state=42 )
    print(f"The lenght training Data for our X is {len(X_train)} and for the test data it is {len(X_test)}")
    print(f"The count of our animes in the Training set is {Counter(y_train)} and for Testing set is {Counter(y_test)}")

    #vectorizing our x (rows or the quotes) and turns it into a matrix (vectorized quote , unigram/bigram and the cell values contain a number of how important the term is to the quote)
    myvectorization = TfidfVectorizer(ngram_range=(1,2), max_df=0.95, min_df = 1, stop_words=None)

    mymodel = LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced')

    pipe = Pipeline([("tfidf", myvectorization), ("clf", mymodel)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    print(accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, digits=3))
    #Precision asks: "When the model predicted this anime, how often was it right?"
    #Recall: "Of all the actual Dragon Ball quotes in the test set, how many did the model successfully identify?"

    # After training, check what the model learned
    def analyze_predictions(pipe, X_test, y_test):
        predictions = pipe.predict(X_test)
        probabilities = pipe.predict_proba(X_test)
        
        print("Detailed predictions:")
        for i, (quote, actual, predicted) in enumerate(zip(X_test, y_test, predictions)):
            max_prob = max(probabilities[i])
            print(f"Quote: '{quote[:30]}...'")
            print(f"  Actual: {actual}, Predicted: {predicted}, Confidence: {max_prob:.3f}")
            if actual != predicted:
                print(f"  ERROR!")
            print()

    # Add this after your current training code
    analyze_predictions(pipe, X_test, y_test)


def main():
    show_label_counts()
    # TODO later: call train(), eval(), then launch REPL

if __name__ == "__main__":
    # main()
    train_and_eval(DATA)