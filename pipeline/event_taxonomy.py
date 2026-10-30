"""Life situations Mind.ai looks for in entries.

Each event:
    label      shown on the Insights page ("Exams")
    phrase     used in alert sentences ("an exam")
    category   groups events (academics, work, social, ...)
    triggers   words/phrases matched on their LEMMA, so "exams", "exam's" and
               "exam" all match "exam". "word:NOUN" only matches that part of
               speech ("went for a run" yes, "ran out of time" no).
    negation_ok  the trigger already contains a negation ("couldn't sleep"),
                 so don't treat "not" as cancelling the event.

Design rule: events are SITUATIONS, never emotion words. "lonely" or "stressed"
would correlate with sadness/fear trivially and teach the user nothing.
Emotion words are what the emotion model measures; events are what causes them.
"""

EVENTS = {
    # ---------------- academics ----------------
    "exam": dict(label="Exams", phrase="an exam", category="academics",
                 triggers=["exam", "examination", "test:NOUN", "paper:NOUN", "viva", "quiz", "midterm",
                           "unit test", "oral exam", "practical exam", "end sem", "endsem", "semester exam"]),
    "assignment": dict(label="Assignments and deadlines", phrase="a deadline", category="academics",
                       triggers=["assignment", "homework", "deadline", "submission", "due date", "submit:VERB"]),
    "results": dict(label="Results and marks", phrase="your results", category="academics",
                    triggers=["result", "marks", "grade:NOUN", "cgpa", "sgpa", "scorecard", "report card"]),
    "presentation": dict(label="Presentations", phrase="a presentation", category="academics",
                         triggers=["presentation", "seminar", "ppt", "demo", "pitch:NOUN", "speech:NOUN"]),
    "classes": dict(label="College and classes", phrase="classes", category="academics",
                    triggers=["lecture", "class:NOUN", "college", "lab:NOUN", "practical:NOUN", "professor", "teacher"]),
    "project": dict(label="Projects", phrase="project work", category="academics",
                    triggers=["project", "hackathon", "thesis", "research"]),

    # ---------------- work ----------------
    "interview": dict(label="Interviews and placements", phrase="an interview", category="work",
                      triggers=["interview", "placement", "recruiter", "aptitude test", "job application", "hr round"]),
    "work": dict(label="Work", phrase="work", category="work",
                 triggers=["office", "shift:NOUN", "job", "workplace", "overtime", "internship", "startup"]),
    "meeting": dict(label="Meetings", phrase="a meeting", category="work",
                    triggers=["meeting", "standup", "stand-up", "client", "call with the client", "review meeting"]),
    "boss": dict(label="Boss and seniors", phrase="your boss", category="work",
                 triggers=["boss", "manager", "senior:NOUN", "team lead", "supervisor"]),

    # ---------------- people ----------------
    "family": dict(label="Family", phrase="family time", category="people",
                   triggers=["mom", "mum", "mother", "dad", "father", "parent", "family", "brother", "sister",
                             "sibling", "grandma", "grandmother", "grandpa", "grandfather", "cousin", "relative:NOUN",
                             "aai", "baba", "papa", "mummy"]),
    "friends": dict(label="Time with friends", phrase="plans with friends", category="people",
                    triggers=["friend", "hangout", "hang out", "party:NOUN", "catch up", "get-together", "squad", "gang:NOUN"]),
    "partner": dict(label="Partner", phrase="time with your partner", category="people",
                    triggers=["girlfriend", "boyfriend", "partner", "wife", "husband", "fiance", "fiancee", "crush:NOUN"]),
    "roommate": dict(label="Roommates", phrase="your roommate", category="people",
                     triggers=["roommate", "roomie", "flatmate", "room partner"]),
    "conflict": dict(label="Arguments and conflict", phrase="a difficult conversation", category="people",
                     triggers=["argument", "fight", "argue", "quarrel", "yell", "shout:VERB", "scold", "breakup",
                               "break up", "misunderstanding", "conflict", "insult"]),

    # ---------------- body ----------------
    "exercise": dict(label="Exercise", phrase="a workout", category="health",
                     triggers=["gym", "workout", "work out", "run:NOUN", "jog", "jogging", "yoga", "exercise",
                               "walk:NOUN", "swim", "swimming", "cricket", "football", "badminton", "cycling",
                               "cycle:VERB", "dance:VERB", "dancing", "trek", "hike"]),
    "poor_sleep": dict(label="Poor sleep", phrase="a short night", category="health", negation_ok=True,
                       triggers=["could not sleep", "can not sleep", "did not sleep", "insomnia", "sleep late",
                                 "sleepless", "stay up", "all nighter", "all-nighter", "barely sleep",
                                 "no sleep", "up all night", "late night"]),
    "good_sleep": dict(label="Good sleep", phrase="a good night's sleep", category="health",
                       triggers=["sleep well", "good sleep", "nap:NOUN", "well rested", "sleep in", "full night"]),
    "illness": dict(label="Feeling unwell", phrase="feeling unwell", category="health",
                    triggers=["sick", "fever", "headache", "migraine", "cough", "doctor", "hospital",
                              "ill", "flu", "stomach ache", "body ache", "injury", "injured"]),

    # ---------------- daily life ----------------
    "commute": dict(label="Commute and travel", phrase="a long commute", category="daily",
                    triggers=["traffic", "commute", "local train", "train:NOUN", "bus:NOUN", "metro", "rickshaw",
                              "cab:NOUN", "uber", "ola", "crowd", "crowded"]),
    "money": dict(label="Money", phrase="money matters", category="daily",
                  triggers=["money", "rent:NOUN", "bill:NOUN", "fee", "fees", "salary", "loan", "expense",
                            "emi", "budget", "pocket money", "broke"]),
    "screen_time": dict(label="Phone and social media", phrase="screen time", category="daily",
                        triggers=["instagram", "youtube", "reel", "scroll:VERB", "scrolling", "social media",
                                  "phone:NOUN", "screen time", "netflix binge", "doomscroll"]),
    "chores": dict(label="Chores", phrase="chores", category="daily",
                   triggers=["chore", "laundry", "cleaning", "clean the", "grocery", "groceries", "dishes",
                             "household work", "cooking for"]),

    # ---------------- things that restore ----------------
    "hobby": dict(label="Hobbies", phrase="a hobby", category="leisure",
                  triggers=["guitar", "piano", "sing:VERB", "singing", "draw:VERB", "drawing", "painting", "paint:VERB",
                            "sketch", "cook:VERB", "bake", "baking", "reading", "novel", "write poetry",
                            "photography", "gardening", "music", "playlist"]),
    "entertainment": dict(label="Movies, shows and games", phrase="a movie night", category="leisure",
                          triggers=["movie", "film:NOUN", "series", "episode", "show:NOUN", "anime",
                                    "video game", "gaming", "concert", "standup comedy"]),
    "outing": dict(label="Trips and outings", phrase="an outing", category="leisure",
                   triggers=["trip", "travel:NOUN", "vacation", "holiday", "beach", "outing:NOUN", "temple",
                             "mall", "cafe", "restaurant", "picnic", "fort", "drive:NOUN", "festival"]),
    "nature": dict(label="Time outdoors", phrase="time outdoors", category="leisure",
                   triggers=["park:NOUN", "garden:NOUN", "sunset", "sunrise", "rain:NOUN", "sea:NOUN", "nature",
                             "fresh air", "terrace"]),
    "me_time": dict(label="Time to yourself", phrase="time to yourself", category="leisure",
                    triggers=["meditate", "meditation", "journal:VERB", "journaling", "me time", "self care",
                              "self-care", "relax:VERB", "rest:NOUN", "day off"]),
}

CATEGORY_LABELS = {
    "academics": "Studies", "work": "Work", "people": "People", "health": "Body and sleep",
    "daily": "Daily life", "leisure": "Leisure", "topic": "Your recurring topics",
}

# Nouns never worth surfacing as a personal "topic" (too generic, time words, or emotion words).
GENERIC_NOUNS = set("""
day time thing today tomorrow yesterday morning afternoon evening night week weekend month year hour minute
second moment lot bit way people person everyone someone anyone everything something anything nothing feeling
feelings mood life kind sort type place stuff part end start beginning idea reason thought thoughts mind head
heart self side point case fact matter question answer problem issue situation chance number piece name word
anxiety stress happiness sadness anger fear joy tension worry excitement pressure frustration panic peace calm
relief guilt shame change plan guy guys girl girls bro dude stuff work-life regret disappointment confidence motivation energy emotion emotions entry journal diary
today's tonight home house room bed lunch dinner breakfast food water tea coffee chai thing things""".split())
