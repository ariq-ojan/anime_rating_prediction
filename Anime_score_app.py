import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

st.title("📺 MyAnimeList Score Predictor")

#Load Model
@st.cache_resource
def load_model():
    with open("Anime_Score_model.sav", "rb") as f:
        model = pickle.load(f)
    return model

class MultiLabelBinarizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
        self.mlb.fit(X)
        return self

    def transform(self, X):
        return self.mlb.transform(X)

model = load_model()

genre_options = [
    "Action", "Adventure", "Avant Garde", "Award Winning", "Boys Love",
    "Comedy", "Drama", "Ecchi", "Erotica", "Fantasy", "Girls Love",
    "Gourmet", "Hentai", "Horror", "Mystery", "Romance", "Sci-Fi",
    "Slice of Life", "Sports", "Supernatural", "Suspense",'Unknown'
]

genres = st.multiselect(
    "Select up to 4 Genres (Choose unknown if it's not on the list)",
    genre_options
)
if len(genres) > 4:
    st.error("You can select up to **4 genres only**.")
    st.stop()
elif len(genres) == 0:
    st.error("**You have to select genre**.")
    st.stop()

studio = st.selectbox("Studio (Choose unknown if it's not on the list)", ['100studio', '10Gauge', 
               '2:10 Animation', '3D', '5 Inc.', '717 Animation Studio', 
               '7doc', '81 Produce', '8bit', 'A-1 Pictures', 'A-Real', 'A.C.G.T.', 'ABJ COMPANY', 'ACC Production', 
               'ACiD FiLM', 'AIC', 'AIC ASTA', 'AIC Build', 'AIC Classic', 'AIC Frontier', 'AIC PLUS+', 'AIC Project', 
               'AIC Spirits', 'AIC Takarazuka', 'AION Studio', 'AMGA', 'APPP', 'AQUA ARIS', 'ARCUS', 'ARECT', 
               'ASK Animation Studio', 'AT-2', 'AXsiZ', 'Academy Productions', 'Acca effe', 'Actas', 'Adonero', 
               'Aeonium', 'Agent 21', 'Ai Yume Mai', 'Aiko', 'Aiti St.', 'Ajia-do', 'Akatsuki', 'Albacrow', 
               'Alice Production', 'Alke', 'Alpha Animation', 'Amarcord', 'An DerCen', 'Anima', 'Anima&Co.', 
               'Animaruya', 'Animation 21', 'Animation 501', 'Animation Planet', 'Animation Staff Room', 'Anime Antenna Iinkai', 
               'Anime Beans', 'Anime R', 'Anime Tokyo', 'Anon Pictures', 'Anpro', 'Arch', 'Arcs Create', 'Ark', 'Arms', 
               'Artland', 'Artmic', 'Arvo Animation', 'Asahi Production', 'Ascension', 'Ashi Productions', 'Asmik Ace', 
               'Atelier Pontdarc', 'Atorie A.B.C.', 'Atti Production', 'Au Praxinoscope', 'Aubec', 'Aurora Animation', 
               'Avaco Creative Studios', 'Azeta Pictures', 'B&T', 'B.CMAY PICTURES', 'BUG FILMS', 'BUILD DREAM', 'BYMENT', 
               'Bakken Record', 'Bandai Namco Pictures', 'Barnum Studio', 'BeSTACK', 'Beat Frog', 'Bee Media', 'Bee Train', 
               'Beijing Enlight Pictures', 'Bibury Animation Studios', 'Big Bang', 'Big Firebird Culture', 'Big Wing', 'Blade', 
               'BloomZ', 'Blue Cat', 'Blue Note', 'Blue bread', 'Bones', 'Bones Film', 'Bouncy', 'Boyan Pictures', 
               "Brain's Base", 'BreakBottle', 'Bridge', 'Buemon', 'Buyuu', 'C and R', 'C&S Production', 'C-Station', 'C2C', 
               'CALF', 'CCTV Animation Group', 'CG Year', 'CLAP', 'CMC Media', 'Cafe de Jeilhouse', 'Chaos Project', 
               'Charaction', "Children's Playground Entertainment", 'Chongzhuo Animation', 'Chosen', 'ChuChu', 'Circle Tribute', 
               'Circus Production', 'Climax Studio', 'Cloud Art', 'Cloud Culture', 'Cloud Hearts', 'CloverWorks', 
               'CoMix Wave Films', 'Coastline Animation Studio', 'Collaboration Works', 'Colored Pencil Animation', 
               'Colored Pencil Animation Japan', 'Comma Studio', 'Concept Films', 'Connect', 'Craftar Studios', 
               'Creators Dot Com', 'Creators in Pack', 'CyberConnect2', 'Cyclone Graphics', 'CygamesPictures', 'D & D Pictures', 
               'D.A.S.T Corporation', 'D.ROCK-ART', 'DAX Production', 'DC Impression Vision', 'DLE', 'DMM.futureworks', 
               'DR Movie', 'DRAWIZ', 'Daewon Media', 'Dai-Ichi Douga', 'Dancing CG Studio', 'DandeLion Animation Studio', 
               'Dangun Pictures', 'Datama Film', 'Daume', 'David Production', 'Dawn Animation', 'Deck', 'Decovocal', 
               'Delight Animation', 'Digital Dream Studios', 'Digital Frontier', 'Digital Media Lab', 'Digital Network Animation', 
               'Diomedéa', 'Directions', 'Djinn Power', 'Doga Kobo', 'Dongwoo A&E', 'Dream Entertainment', 'Drive', 'Durufix', 
               'Dyna Method', 'Dynamo Pictures', 'E&G Films', 'E&H Production', 'EKACHI EPILKA', 'EMT Squared', 'ENGI', 
               'EOEO System', 'EOTA', 'East Fish Studio', 'Echo', 'Echoes', 'Egg', 'Eiken', 'Ekura Animal', 'ElectromagneticWave', 
               'Elias', 'Emon', 'Encourage Films', 'Enishiya', 'Enzo Animation', 'Eshoya Honpo', 'Ether Kitten', 
               'Executive Decision', 'Ezόla', 'Fanworks', 'Felix Film', 'Fengyun Animation', 'Fenz', 'Fever Creations', 
               'Fifth Avenue', 'Flat Studio', 'Flavors Soft', 'Flint Sugar', 'Flying Ship Studio', 'Foch Film', 'Fortes', 
               'Four Some', 'Front Line', 'Front Wing', 'Frontier Works', 'Fugaku', 'Fuji TV', 'Fukushima Gaina', 'Funny Flux', 
               'Future Planet', 'G&G Entertainment', 'G-Lam', 'G-angle', 'G.P Entertainment', 'GARDEN Culture', 'GARDEN LODGE', 
               'GRIZZLY', 'Gaina', 'Gainax', 'Gainax Kyoto', 'Gakken', 'Gakken Eigakyoku', 'Gallop', 'Gambit', 'Garyuu Studio', 
               'Gathering', 'Gear Studio', 'Geek Toys', 'Gekkou', 'Geno Studio', 'Ginga Teikoku', 'Ginga Ya', 'GoHands', 'Gonzo', 
               'Gosay Studio', 'Graphinica', 'Gravity Well', 'Green Monster Team', 'Group TAC', 'Grouper Productions', 'Gunners',
               'Guton Animation Studio', 'Gyorai Eizo Inc.', 'HAL Film Maker', 'HMCH', 'HOTZIPANG', 'HS Pictures Studio', 
               'Hai An Xian Donghua Gongzuo Shi', 'Hand to Mouse.', 'Haoliners Animation League', 'Happy Elements', 'Hayabusa Film',
               'Heart & Soul Animation', 'Heewon Entertainment', 'High Energy Studio', 'Himajin Planning', 'Hololive Production', 
               'Hong Ying Animation', 'Hoods Entertainment', 'Horannabi', 'Hoso Seisaku Doga', 'Hotline', 'HuaDream', 
               'HuaMei Animation', 'Hurray!', 'I & A', 'I was a Ballerina', 'I-move', 'I.Gzwei', 'I.Toon', 'IKK Room', 'ILCA', 
               'IMAGICA Lab.', 'Iconix Entertainment', 'Idea Factory', 'Ijigen Tokyo', 'Image House', 'Image Kei', 
               'Imageworks Studio', 'Imagi', 'Imagica', 'Imagica Digitalscape', 'Imagica Infos', 'Imagin', 'Imagineer', 
               'Indeprox', 'Inugoya', 'Ishibashi Planning', 'Ishikawa Pro', 'Ishimori Entertainment', 'Itasca Studio', 
               'Iyasakadou Film', 'J.C.Staff', 'J.K.I', 'JCF', 'JOF', 'Japan Vistec', 'Jichitai Anime', 'Jinnis Animation Studios', 
               'Joicy Studio', 'Joker Films', 'Jumondou', 'Jumonji', 'K-Factory', 'KAGAYA Studio', 'KIZAWA Studio', 
               'KKC Animation Production', 'KOO-KI', 'KSS', 'Kaca Entertainment', 'Kachidoki Studio', 'Kachigarasu', 
               'Kaeruotoko Shokai', 'Kami Kukan', 'Kamikaze Douga', 'Kamio Japan', 'Kanaban Graphics', 'Kaname Productions', 
               'Kantou Douga Kai', 'Karaku', 'Karasfilms', 'Kate Arrow', 'Kazami Gakuen Koushiki Douga-bu', 'Kazuki Production', 
               'Keica', 'Kenji Studio', 'Kent House', 'Keyring', 'Khara', 'Kinema Citrus', 'Kitty Film Mitaka Studio', 
               'Knack Productions', 'Kojiro Shishido Animation Works', 'Kokusai Eigasha', 'Kung Fu Frog Animation', 
               'Kuri Jikken Manga Koubou', 'Kyoto Animation', 'Kyushu Network Animation', 'LAN Studio', 'LICO', 'LIDENFILMS', 
               'LMD', 'LX Animation Studio', 'LandQ studios', 'Lapin Track', 'Larx Entertainment', 'Lay-duce', 
               'Le-joy Animation Studio', 'Lerche', 'Lesprit', 'Liber', 'Liberty Animation Studio', 'Lide', 'Life Work', 
               'Light Chaser Animation Studios', 'Lingsanwu Animation', 'Live2D Creative Studio', 'Liyu Culture', 'Lyrics', 
               'L²Studio', 'M&M', 'M.S.C', 'MAPPA', 'MASTER LIGHTS', 'MB planning', 'MI', 'MK Pictures', 'MMT Technology', 
               'Maboroshi Koubou', 'Madhouse', 'Magic Bus', 'Maho Film', 'Majin', 'Makaria', 'Making Animation', 'Manglobe', 
               'Manpuku Jinja', 'Maro Studio', 'Marui Group', 'Marvy Jack', 'Marza Animation Planet', 'Media Factory', 'Medo', 
               'Melissa', 'Meltdown', 'Meruhensha', 'Mikimoto Production', 'Mili Pictures', 'Milky Cartoon', 'Millepensee', 
               'Minami Machi Bugyousho', 'Mirai Film', 'MoMo Production', 'Mokai Technology', 'Momoi Planning', 'MooGoo', 
               'Mook Animation', 'Mook DLE', 'Motion Magic', 'Mouse', 'Mousou Senka', 'Movic', 'Mushi Production', 'NAZ', 'NHK', 
               'NHK Enterprises', 'Namu Animation', 'Neft Film', 'Nekonigashi Inc.', 'Network Kouenji Studio', 'New Deer', 
               'New Generation', 'Next Media Animation', 'Nexus', 'Niceboat Animation', 'Nihikime no Dozeu', 'Nihon Hoso Eigasha', 
               'Nippon Animation', 'Nippon Ramayana Film', 'Nippon TV Douga', 'No Side', 'Nomad', 'Noovo', 'Nur', 'Nut', 
               'OB Planning', 'OLM', 'OLM Digital', 'ORADA COMPANY', 'ORCEN', 'ORENDA', 'OZ', 'Ocon Studio', 'October Media', 
               'Oddjob', 'Odolttogi', 'Office AO', 'Office DCI', 'Office No. 8', 'Office Take Off', 'Office TakeOut', 
               'Oh! Production', 'Okumaza', 'Okuruto Noboru', 'Onion Studio', 'Opera House', 'Orange', 'Ordet', 
               'Oriental Creative Color', 'Original Force', 'Otogi Production', 'Oxybot', 'Oz Inc.', 'P core', 'P.A. Works', 
               'PHANTOM', 'PINE JAM', 'PP Project', 'PPM', 'PRA', 'Painting Dream', 'Palm Studio', 'Panda Factory', 
               'Panda Tower Studio', 'Panmedia', 'Paper Animation', 'Paper Plane Animation Studio', 'Passion Paint Animation', 
               'Passione', 'Pastel', 'Pb Animation', 'Peak Hunt', 'Pepper Conpanna', 'Phoenix Entertainment', 'Picograph', 'Picona', 
               'Picture Magic', 'Pie in the sky', 'Pierrot Films', 'Pierrot Plus', 'Piko Studio', 'Pink Cat', 'Piso Studio', 'Planet', 
               'Planet Cartoon', 'Platinum Vision', 'Plum', 'Plus Heads', 'PoRO', 'Point Pictures', 'Pollyanna Graphics', 
               'Polygon Pictures', 'Potato House', 'Primastea', 'PrimeTime', 'Production +h.', 'Production D.M.H', 'Production I.G', 
               'Production IMS', 'Production Reed', 'Project No.9', 'Project Team Muu', 'Project Team Sarah', 'Public & Basic', 
               'Public Enemies', 'Puzzle Animation Studio Limited', 'Qianqi Animation', 'Qingxiang Culture', 'Qiyuan Yinghua', 'Quad', 
               'Qualia Animation', 'Qubic Pictures', 'Quyue Technology', 'Qzil.la', 'RG Animation Studios', 'ROLL2', 'Rabbit Gate', 
               'Rabbit Machine', 'Radix', 'Raiose', 'Red Dog Culture House', 'Remic', 'Revoroot', 'Rikuentai', 'Ripple Film', 'Ripromo', 
               'Rising Force', 'Robot Communications', 'Rocen', "Rock'n Roll Mountain", 'Rockwell Eyes', 'Ruo Hong Culture', "Ryuu M's", 
               'S.o.K', 'SAMG Entertainment', 'SANZIGEN', 'SBS TV Production', 'SELFISH', 'SIDO LIMITED', 'SILVER LINK.', 'SJYNEXCUS', 
               "STUDIO6'oN", 'STUDIOK110', 'Saetta', 'Saigo no Shudan', 'Sakura Create', 'Samsara Animation Studio', 'San-X', 'Sanctuary', 
               'Sanrio', 'Sanrio Digital', 'Sasayuri', 'Satelight', 'Schoolzone', 'Science SARU', 'Scooter Films', 'Seoul Movie', 'Seven', 
               'Seven Arcs', 'Seven Arcs Pictures', 'Seven Stone Entertainment', 'Shaft', 'Shanghai Animation Film Studio', 'Sharefun Studio', 
               'Shengguang Knight Culture', 'Shengying Animation', 'Shenman Entertainment', 'Shin-Ei Animation', 'Shindeban Film', 
               'Shinjukuza', 'Shinkuukan', 'Shion', 'Shirogumi', 'Shochiku Animation Institute', 'Shogakukan Music & Digital Entertainment', 
               'Shuiniu Dongman', 'Shuka', 'Shura', 'Shykeumo Animation Studio', 'Signal.MD', 'Silver', 'Skyloong', 'Sofix', 'Soft Garage', 
               'Soigne', 'Sola Digital Arts', 'Sonsan Kikaku', 'Sotsu', 'Sovat Theater', 'Space Neko Company', 'Space-X', 
               'Sparkly Key Animation Studio', 'Sparky Animation', 'Spell Bound', 'Spooky graphic', 'Square Enix Visual Works', 
               'Square Pictures', 'Staple Entertainment', 'Starry Cube', 'Steamworks', "Steve N' Steven", 'Stingray', 
               'Strawberry Meets Pictures', 'Studio 1st', 'Studio 3Hz', 'Studio 4°C', 'Studio 88', 'Studio 9 Maiami', 'Studio A-CAT', 
               'Studio Animal', 'Studio BAZOOKA', 'Studio Barcelona', 'Studio Bind', 'Studio Bingo', 'Studio Binzo', 'Studio Blanc.',
               'Studio Bogey', 'Studio Boogie Nights', 'Studio CANDY BOX', 'Studio Chizu', 'Studio Coa', 'Studio Colorido', 'Studio Comet', 
               'Studio Core', 'Studio Crocodile', 'Studio DURIAN', 'Studio Dadashow', 'Studio Daisy', 'Studio Deen', 'Studio Dolphin Night', 
               'Studio Egg', 'Studio Eight Color', 'Studio Elle', 'Studio Eromatick', 'Studio Fantasia', 'Studio Flad', 'Studio Flag', 
               'Studio Fusion', 'Studio G-1Neo', 'Studio GOONEYS', 'Studio Gadget', 'Studio Ghibli', 'Studio Gokumi', 'Studio Gram', 
               'Studio Hibari', 'Studio Himalaya', 'Studio Hokiboshi', 'Studio Izena', 'Studio Jam', 'Studio Junio', 'Studio KAI', 
               'Studio Kafka', 'Studio KeepFire', 'Studio Khronos', 'Studio Kikan', 'Studio Kingyoiro', 'Studio Korumi', 'Studio Kyuuma', 
               'Studio LEO', 'Studio Live', 'Studio M2', 'Studio March', 'Studio Massket', 'Studio Matrix', 'Studio Meditation With a Pencil', 
               'Studio Mir', 'Studio Moe', 'Studio Moriken', 'Studio N', 'Studio Nanahoshi', 'Studio Nuck', 'Studio OX', 'Studio Outrigger', 
               'Studio Palette', 'Studio Pierrot', 'Studio Pivote', 'Studio Placebo', 'Studio Polon', 'Studio Ponoc', 'Studio Ppuri', 
               'Studio Prokion', 'Studio PuYUKAI', 'Studio Ranmaru', 'Studio Rikka', 'Studio Shelter', 'Studio Sign', 'Studio Signal', 
               'Studio Signpost', 'Studio Sota', 'Studio Soul', 'Studio Take Off', 'Studio Ten', 'Studio Ten Carat', 'Studio Tumble', 
               'Studio UGOKI', 'Studio Unicorn', 'Studio VOLN', 'Studio W.Baba', 'Studio Wombat', 'Studio World', 'Studio Z5', 'Studio Zero', 
               'Sublimation', 'Success Corp.', 'Sugar Boy', 'Sumomo Film', 'Sunflowers', 'Sunny Gapen', 'Sunny Side Up', 'Sunrise', 
               'Sunrise Beyond', 'Sunwoo Entertainment', 'Suoyi Technology', 'Super Normal Studio', 'Suzuki Mirano', 'SynergySP', 'T-Rex', 
               'T.P.O', 'TCJ', 'TEC', 'TMS Entertainment', 'TNK', 'TOHO animation STUDIO', 'TROYCA', 'TUBA', 'TV Douga', 'TYMOTE', 
               'TYO Animations', 'Taikong Works', 'Takahashi Studio', 'Takun Manga Box', 'Tama Production', 'Tamura Shigeru Studio', 
               'Tang Kirin Culture', 'Tatsunoko Production', 'Team OneOne', 'Team TillDawn', 'Team YokkyuFuman', 'Tear Studio', 
               'Teatro Nishi Tokyo Studio', 'Tecarat', 'Teddy', 'Telecom Animation Film', 'Telescreen', 'Tezuka Productions', 
               'The Answer Studio', 'Thundray', 'Tianshi Wenhua', 'Toei Animation', 'Toei Video', 'Toho Interactive Animation', 
               'Tokyo Kids', 'Tokyo Movie', 'Tokyo Movie Shinsha', 'Tokyo TV Douga', 'Tomason', 'Tomovies', 'Tomoyasu Murata Company', 
               'Tonari Animation', 'Tong Mingxuan Studio', 'Tonko House', 'Topcraft', 'Toyo Links Corporation', 'Trans Arts', 
               'Transcendence Picture', 'Trash Studio', 'TriF Studio', 'Triangle Staff', 'Trigger', 'Trinet Entertainment', 'Triple A', 
               'Triple X', 'Tryforce', 'Tsubo Production', 'Tsuburaya Productions', 'Tsuchida Productions', 'Tsukimidou', 
               'Tsumugi Akita Animation Lab', 'TthunDer Animation', 'Twenty First', 'Twilight Studio', 'Twilight Town', 'TypeZero', 
               'Typhoon Graphics', 'UKA', 'UWAN Pictures', 'UchuPeople', 'Ultra Super Pictures', 'Unend', 'Unknown', 'Urban Product', 
               'Usagi Ou', 'VCRWORKS', 'Valkyria', 'Vasoon Animation', 'Vega Entertainment', 'Venet', 'Viewworks', 'Village Studio', 
               'Visual 80', 'Voil', 'Volca', 'W-Toon Studio', 'WAO World', 'Wako Productions', 'Wawayu Animation', 'White Fox', 'Wise Guy', 
               'Wit Studio', 'Wolf Smoke Studio', 'Wolfsbane', 'Wonder Cat Animation', 'Wong Ping Animation Lab', 'Wulifang', 'XEBEC M2', 
               'Xebec', 'Xiaoming Taiji', 'Xing Yi Kai Chen', 'Xuni Ying Ye', 'Y.O.U.C', 'YHKT Entertainment', 'YURUPPE Inc.', 
               'Yamamura Animation', 'Yamato Works', 'Yamiken', 'Yaoyorozu', 'Yasuda Genshou Studio by Xenotoon', 'Yi Chen Animation', 
               'Yien Animation Studio', 'Yokohama Animation Laboratory', 'Yostar Pictures', 'Yumeta Company', 'Zelico Film', 'Zero-G', 
               'Zero-G Room', 'Zexcs', 'Ziine Studio', 'Zuiyo', 'Zyc', 'animate Film', 'asread.', 'asurafilm', 'domerica', 'drop', 'dwarf', 
               'evg', 'feel.', 'happyproject', 'helo.inc', 'iDRAGONS Creative Studio', 'indigo line', 'lxtl', 'monofilmo', 'pH Studio', 
               'production doA', 'soket', 'studio MOTHER', 'ufotable'])
type = st.selectbox('type',[
    'TV', 'OVA', 'ONA', 'Special', 'TV Special', 'Movie'
])
episodes = st.number_input("Episodes", min_value=1, max_value=26)

sourceDefault = 'Manga'
useSource = st.toggle("Use Source", value=True)
if useSource:
    source = st.selectbox('Source',[
        'Manga', 'Visual novel', 'Light novel', 'Novel', 'Original',
        'Web manga', '4-koma manga', 'Web novel', 'Game', 'Other', 'Book',
        'Mixed media', 'Music', 'Picture book', 'Unknown', 'Card game',
        'Radio'
    ])
else:
    source = sourceDefault

ratingRaw = [
        'PG-13 - Teens 13 or older', 'G - All Ages', 'Rx - Hentai', 
        'R - 17+ (violence & profanity)', 'PG - Children', 'R+ - Mild Nudity'
]

def cleanForDisplay(opt):
        if opt.startswith("PG-13"):
            return "PG - 13"
        elif opt.startswith("G"):
            return "G - All"
        elif opt.startswith("Rx"):
            return "Rx - Hentai"
        elif opt.startswith("R - 17+"):
            return "R - 17+"
        elif opt.startswith("PG - Children"):
            return "PG - Child"
        elif opt.startswith("R+"):
            return "R+ - Mild Nudity"
        else:
             return opt
displayOption = [cleanForDisplay(o) for o in ratingRaw]
rating = st.selectbox('Rating', displayOption)  
popularity = st.number_input("popularity", min_value=1, max_value=22225)

DurationDefault = 23
useSDuration = st.toggle("Use Duration", value=True)
if useSDuration:
    duration = st.number_input("Duration Minutes", min_value=1, max_value=168)
else:
    duration = DurationDefault
favorites = st.number_input("Favorites", min_value=0.0, max_value=236798.00)
scoredby = st.number_input("Scored By", min_value=102.0, max_value=2980783.00)
members = st.number_input("Members", min_value=209.00, max_value=4231885.00)

if st.button("Predict"):
    df = pd.DataFrame([{
        'genres': genres,
        'type': type,
        'episodes': episodes,
        'source': source,
        'rating': rating,
        'popularity': popularity,
        'duration_minutes': duration,
        'main_studio': studio,
        'log_favorites': np.log1p(favorites),
        'log_scored_by': np.log1p(scoredby),
        'log_members': np.log1p(members),
        'engagement_ratio': (favorites/members),
        'score_density': (scoredby/members),
        'popularity_inverse': (1/popularity),
    }])

    pred = model.predict(df)[0]
    st.write("Predicted Score:", pred)

    rmse = 0.375146
    mape = 4.373426

    # Range prediksi
    lower = max(1, pred - rmse)
    upper = min(10, pred + rmse)

    st.subheader("Predicted Score")
    st.write(f"### **{pred:.2f}**")

    st.markdown("### Prediction Range (±RMSE)")
    st.write(
        f"""
        Based on model, prediction range are:

        #### **{lower:.2f} - {upper:.2f}**

        So,the anime's actual score value approximately around:
        ± **{rmse:.3f} point** from prediction.
        """
    )

    st.markdown("### Model Justification")
    st.write(
        f"""
        **RMSE: `{rmse:.3f}`**  
        The average prediction error around **0.37 point**, so keep in mind that it might 
        not be a very accurate score.

        **MAPE: `{mape:.2f}%`**  
        Provides an indication that the relative error of the model around **4.37%**.
        Which means the model has an accuracy of more than **95%**.
        """

    )
